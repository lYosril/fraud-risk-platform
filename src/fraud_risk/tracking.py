"""MLflow experiment tracking for training runs.

Responsibility
    Own the boundary between the training code and MLflow: build a tracking configuration,
    start and finish a run, record parameters/tags/metrics/artifacts, and mark a run failed
    when training or artifact logging raises. MLflow is imported lazily so importing the
    training module (or the configuration) never contacts a tracking server.

What is recorded
    - Training configuration and model parameters, including seed, cost policy, grid sizes,
      calibration folds and compute settings.
    - Split strategy, split seed, fractions, row counts and position digests - never rows.
    - Feature columns, their dtypes and a schema digest.
    - Validation and test metrics of the selected model, plus its threshold. The test split
      is reported but never used for selection.
    - The local artifact bundle (model, comparison, metrics and run records) as run artifacts.

What is deliberately not recorded
    Complete datasets, raw transaction rows, feature snapshots, credentials or database DSNs.
    The tracking URI is redacted before it is written into a local record. Model registration
    is *not* performed here; publishing or promoting a model is a separate reviewed step.

Reproducible identity
    :func:`build_run_fingerprint` hashes the seed, split record, feature schema, training
    configuration and git revision. The resulting fingerprint is recorded in ``run.json``,
    attached as an MLflow tag and used in the run name. MLflow still assigns its own random
    ``run_id``; the fingerprint is the reproducible identity, not a replacement for it.

Status
    Implemented for the training stage. Registry aliases and deployment remain M5 work; no
    model is registered by this module.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit, urlunsplit

if TYPE_CHECKING:
    from fraud_risk.config import Settings

logger = logging.getLogger(__name__)

REGISTRY_NOTE = "not performed; model registration is a separate reviewed step"
MAX_FAILURE_TAG_LENGTH = 1000


class TrackingError(ValueError):
    """The run cannot be tracked honestly or the tracking backend cannot be reached."""


@dataclass(frozen=True)
class TrackingConfig:
    """Where a training run is recorded.

    ``tracking_uri`` is any URI MLflow accepts. The development default is a local SQLite
    store so no server is required. ``artifact_location`` is optional and only applied when
    the experiment is first created; it is useful for tests and for keeping artifacts beside
    a local database.
    """

    tracking_uri: str
    experiment_name: str
    artifact_location: str | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("tracking_uri", self.tracking_uri),
            ("experiment_name", self.experiment_name),
        ):
            if not isinstance(value, str) or not value.strip():
                msg = f"{name} must be a non-empty string (got {value!r})"
                raise TrackingError(msg)
            object.__setattr__(self, name, value.strip())
        if self.artifact_location is not None:
            if not isinstance(self.artifact_location, str) or not self.artifact_location.strip():
                msg = (
                    "artifact_location must be a non-empty string or None "
                    f"(got {self.artifact_location!r})"
                )
                raise TrackingError(msg)
            object.__setattr__(self, "artifact_location", self.artifact_location.strip())

    @classmethod
    def from_settings(cls, settings: Settings) -> TrackingConfig:
        """Build the tracking configuration from the single settings object."""
        return cls(
            tracking_uri=settings.mlflow_tracking_uri,
            experiment_name=settings.mlflow_experiment_name,
        )

    @property
    def safe_tracking_uri(self) -> str:
        """Tracking URI with userinfo and query parameters removed for local records."""
        return redact_uri(self.tracking_uri)


@dataclass(frozen=True)
class TrackingRecord:
    """The identity and location of a real MLflow run."""

    run_id: str
    experiment_id: str
    experiment_name: str
    tracking_uri: str
    run_name: str
    run_fingerprint: str
    artifact_uri: str | None
    registry: str = REGISTRY_NOTE

    def to_dict(self) -> dict[str, Any]:
        """Serialise the record without credentials or query strings."""
        return {
            "enabled": True,
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "tracking_uri": self.tracking_uri,
            "run_name": self.run_name,
            "run_fingerprint": self.run_fingerprint,
            "artifact_uri": self.artifact_uri,
            "registry": self.registry,
        }


@dataclass
class TrackingSession:
    """An active MLflow run. The client is intentionally not exposed."""

    record: TrackingRecord
    _client: Any = field(repr=False)
    _finished: bool = False

    def log_params(self, params: Mapping[str, object]) -> None:
        """Record string, numeric or JSON-serialisable parameters."""
        self._require_open()
        for key, value in params.items():
            if not isinstance(key, str) or not key:
                msg = f"parameter names must be non-empty strings (got {key!r})"
                raise TrackingError(msg)
            try:
                self._client.log_param(self.record.run_id, key, _param_value(value))
            except Exception as exc:
                msg = f"could not log parameter {key!r} to MLflow: {exc}"
                raise TrackingError(msg) from exc

    def log_metrics(self, metrics: Mapping[str, float]) -> None:
        """Record finite numeric metrics."""
        self._require_open()
        for key, value in metrics.items():
            if not isinstance(key, str) or not key:
                msg = f"metric names must be non-empty strings (got {key!r})"
                raise TrackingError(msg)
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                msg = f"metric {key!r} must be numeric (got {value!r})"
                raise TrackingError(msg) from exc
            if not math.isfinite(number):
                msg = f"metric {key!r} must be finite (got {value!r})"
                raise TrackingError(msg)
            try:
                self._client.log_metric(self.record.run_id, key, number)
            except Exception as exc:
                msg = f"could not log metric {key!r} to MLflow: {exc}"
                raise TrackingError(msg) from exc

    def log_artifacts(
        self,
        paths: Sequence[Path],
        *,
        artifact_path: str | None = None,
    ) -> None:
        """Upload local files as run artifacts; dataset files must never be passed here."""
        self._require_open()
        for raw_path in paths:
            path = Path(raw_path)
            if not path.is_file():
                msg = f"artifact does not exist or is not a file: {path}"
                raise TrackingError(msg)
            try:
                self._client.log_artifact(
                    self.record.run_id,
                    str(path),
                    artifact_path=artifact_path,
                )
            except Exception as exc:
                msg = f"could not log artifact {path} to MLflow: {exc}"
                raise TrackingError(msg) from exc

    def complete(self) -> None:
        """Mark the run finished successfully."""
        if self._finished:
            return
        try:
            self._client.set_tag(self.record.run_id, "training.status", "completed")
            self._client.set_terminated(self.record.run_id, "FINISHED")
        except Exception as exc:
            msg = f"could not mark MLflow run {self.record.run_id} as finished: {exc}"
            raise TrackingError(msg) from exc
        self._finished = True

    def fail(self, error: BaseException) -> None:
        """Best-effort terminal FAILED state; never mask the original training error."""
        if self._finished:
            return
        try:
            self._client.set_tag(self.record.run_id, "training.status", "failed")
            self._client.set_tag(self.record.run_id, "failure.type", type(error).__name__)
            self._client.set_tag(
                self.record.run_id,
                "failure.message",
                _truncate(str(error), MAX_FAILURE_TAG_LENGTH),
            )
        except Exception as exc:  # tracking must not replace the real failure
            logger.warning(
                "could not attach failure tags to MLflow run %s: %s",
                self.record.run_id,
                exc,
            )
        try:
            self._client.set_terminated(self.record.run_id, "FAILED")
        except Exception as exc:  # tracking must not replace the real failure
            logger.warning(
                "could not mark MLflow run %s as failed: %s",
                self.record.run_id,
                exc,
            )
        self._finished = True

    def _require_open(self) -> None:
        if self._finished:
            msg = f"MLflow run {self.record.run_id} is already finished"
            raise TrackingError(msg)


@contextmanager
def track_experiment(
    config: TrackingConfig,
    *,
    run_name: str,
    run_fingerprint: str,
    tags: Mapping[str, object] | None = None,
    params: Mapping[str, object] | None = None,
) -> Iterator[TrackingSession]:
    """Start, yield and terminate one MLflow run, marking it FAILED on any error."""
    session = start_tracking(
        config,
        run_name=run_name,
        run_fingerprint=run_fingerprint,
        tags=tags,
        params=params,
    )
    try:
        yield session
        session.complete()
    except BaseException as exc:
        session.fail(exc)
        raise


def start_tracking(
    config: TrackingConfig,
    *,
    run_name: str,
    run_fingerprint: str,
    tags: Mapping[str, object] | None = None,
    params: Mapping[str, object] | None = None,
) -> TrackingSession:
    """Create or reuse the experiment and start a real MLflow run."""
    if not isinstance(run_name, str) or not run_name.strip():
        msg = f"run_name must be a non-empty string (got {run_name!r})"
        raise TrackingError(msg)
    if not isinstance(run_fingerprint, str) or not run_fingerprint.strip():
        msg = "run_fingerprint must be a non-empty string"
        raise TrackingError(msg)

    session: TrackingSession | None = None
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=config.tracking_uri)
        experiment = client.get_experiment_by_name(config.experiment_name)
        if experiment is None:
            experiment_id = client.create_experiment(
                config.experiment_name,
                artifact_location=config.artifact_location,
            )
        else:
            experiment_id = experiment.experiment_id

        tag_values = {str(key): _tag_value(value) for key, value in (tags or {}).items()}
        tag_values.setdefault("run.fingerprint", run_fingerprint)
        tag_values.setdefault("tracking.registry", REGISTRY_NOTE)
        run = client.create_run(
            experiment_id,
            run_name=run_name.strip(),
            tags=tag_values,
        )
        artifact_uri = getattr(run.info, "artifact_uri", None)
        record = TrackingRecord(
            run_id=run.info.run_id,
            experiment_id=experiment_id,
            experiment_name=config.experiment_name,
            tracking_uri=config.safe_tracking_uri,
            run_name=run_name.strip(),
            run_fingerprint=run_fingerprint,
            artifact_uri=redact_uri(str(artifact_uri)) if artifact_uri else None,
        )
        session = TrackingSession(record=record, _client=client)
        if params:
            session.log_params(params)
        return session
    except Exception as exc:
        if session is not None:
            session.fail(exc)
        if isinstance(exc, TrackingError):
            raise
        msg = f"could not start MLflow tracking at {config.safe_tracking_uri}: {exc}"
        raise TrackingError(msg) from exc


def build_run_fingerprint(
    *,
    seed: int,
    split: Mapping[str, Any],
    feature_columns: Sequence[str],
    config: Mapping[str, Any],
    git_revision: str | None,
) -> str:
    """Hash the run's reproducible inputs into a stable identifier.

    Timestamps, fit durations and MLflow's own run id are excluded on purpose: the same
    data partition, schema, configuration, seed and code revision produce the same value.
    """
    payload = {
        "seed": seed,
        "split": dict(split),
        "feature_columns": list(feature_columns),
        "config": dict(config),
        "git_revision": git_revision,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_run_name(experiment_name: str, fingerprint: str) -> str:
    """A deterministic run name; MLflow still assigns its own run id."""
    return f"{experiment_name}-{fingerprint[:12]}"


def redact_uri(uri: str) -> str:
    """Remove userinfo and query strings from a URI before it is recorded."""
    try:
        parts = urlsplit(uri)
    except ValueError:
        return uri
    if not parts.scheme or not parts.netloc:
        return uri
    host = parts.hostname or ""
    if not host:
        return urlunsplit((parts.scheme, "", parts.path, "", ""))
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    try:
        port = parts.port
    except ValueError:
        port = None
    netloc = f"{host}:{port}" if port is not None else host
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))


def _param_value(value: object) -> str | int | float:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (str, int, float)):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _tag_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (str, int, float)):
        return str(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


__all__ = [
    "REGISTRY_NOTE",
    "TrackingConfig",
    "TrackingError",
    "TrackingRecord",
    "TrackingSession",
    "build_run_fingerprint",
    "build_run_name",
    "redact_uri",
    "start_tracking",
    "track_experiment",
]
