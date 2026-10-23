"""Compact reference profiles built from the training split.

Responsibility
    Describe the training distribution in enough detail to compute drift later, without
    storing a copy of the training data:

    - Reference quantile bin edges and bin shares per feature.
    - A small sorted quantile sample per feature and for the model score, used to estimate
      the KS statistic.
    - Missing-value rate, row count and the model identity that produced the scores.

The profile is written as JSON by the monitoring CLI and loaded by the scheduled report run.
It is an aggregate artefact: no transaction identifier, label or full feature row is present.

Reference construction rule
    The training split must be produced by the same split strategy and seed as the model
    training run. The CLI records both values so a later run can be audited. A profile built
    from validation or test rows is not a valid reference and must not be used.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from fraud_risk.monitoring.drift import (
    DEFAULT_BIN_COUNT,
    DEFAULT_QUANTILE_SAMPLE_SIZE,
    bin_proportions,
    finite_values,
    quantile_edges,
)

REFERENCE_SCHEMA_VERSION = 1
SCORE_DISTRIBUTION_NAME = "risk_score"


class ReferenceProfileError(ValueError):
    """A reference profile is missing, malformed or inconsistent."""


@dataclass(frozen=True, slots=True)
class DistributionReference:
    """Compact description of one numeric reference distribution."""

    name: str
    count: int
    missing_rate: float
    mean: float
    bin_edges: tuple[float, ...]
    bin_proportions: tuple[float, ...]
    quantile_sample: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-ready representation."""
        return {
            "name": self.name,
            "count": self.count,
            "missing_rate": self.missing_rate,
            "mean": self.mean,
            "bin_edges": list(self.bin_edges),
            "bin_proportions": list(self.bin_proportions),
            "quantile_sample": list(self.quantile_sample),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        """Validate and parse one distribution from JSON."""
        name = _require_str(payload, "name")
        count = _require_int(payload, "count", minimum=1)
        missing_rate = _require_float(payload, "missing_rate", minimum=0.0, maximum=1.0)
        mean = _require_float(payload, "mean")
        edges = tuple(
            _finite_float(value, f"{name}.bin_edges")
            for value in _require_list(payload, "bin_edges")
        )
        if any(right <= left for left, right in itertools.pairwise(edges)):
            msg = f"{name}.bin_edges must be strictly increasing"
            raise ReferenceProfileError(msg)
        proportions = tuple(
            _require_float_value(value, f"{name}.bin_proportions", minimum=0.0, maximum=1.0)
            for value in _require_list(payload, "bin_proportions")
        )
        if len(proportions) != len(edges) + 1:
            msg = (
                f"{name}.bin_proportions must have len(bin_edges) + 1 entries "
                f"(got {len(proportions)} for {len(edges)} edges)"
            )
            raise ReferenceProfileError(msg)
        if not math.isclose(sum(proportions), 1.0, rel_tol=0.0, abs_tol=1e-8):
            msg = f"{name}.bin_proportions must sum to 1"
            raise ReferenceProfileError(msg)
        sample = tuple(
            _finite_float(value, f"{name}.quantile_sample")
            for value in _require_list(payload, "quantile_sample")
        )
        if not sample:
            msg = f"{name}.quantile_sample must not be empty"
            raise ReferenceProfileError(msg)
        if any(right < left for left, right in itertools.pairwise(sample)):
            msg = f"{name}.quantile_sample must be sorted"
            raise ReferenceProfileError(msg)
        return cls(
            name=name,
            count=count,
            missing_rate=missing_rate,
            mean=mean,
            bin_edges=edges,
            bin_proportions=proportions,
            quantile_sample=sample,
        )


@dataclass(frozen=True, slots=True)
class ReferenceProfile:
    """Aggregate training reference used by every drift report."""

    generated_at: datetime
    reference_id: str
    source: str
    model_version: str
    model_sha256: str
    threshold: float
    split_strategy: str
    split_seed: int
    training_rows: int
    scored_rows: int
    sample_method: str
    features: tuple[DistributionReference, ...]
    score: DistributionReference
    schema_version: int = REFERENCE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-ready representation."""
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at.astimezone(UTC).isoformat(),
            "reference_id": self.reference_id,
            "source": self.source,
            "model_version": self.model_version,
            "model_sha256": self.model_sha256,
            "threshold": self.threshold,
            "split": {
                "strategy": self.split_strategy,
                "seed": self.split_seed,
                "training_rows": self.training_rows,
                "scored_rows": self.scored_rows,
                "sample_method": self.sample_method,
            },
            "features": [item.to_dict() for item in self.features],
            "score": self.score.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        """Validate and parse a reference profile from JSON."""
        schema_version = _require_int(payload, "schema_version", minimum=1)
        if schema_version != REFERENCE_SCHEMA_VERSION:
            msg = (
                f"unsupported reference profile schema_version {schema_version}; "
                f"expected {REFERENCE_SCHEMA_VERSION}"
            )
            raise ReferenceProfileError(msg)
        generated_at = _require_datetime(payload, "generated_at")
        reference_id = _require_str(payload, "reference_id")
        split = _require_mapping(payload, "split")
        features_payload = _require_list(payload, "features")
        features = tuple(
            DistributionReference.from_dict(_require_mapping_value(item, "features item"))
            for item in features_payload
        )
        if not features:
            msg = "reference features must not be empty"
            raise ReferenceProfileError(msg)
        names = [item.name for item in features]
        if len(set(names)) != len(names):
            msg = "reference feature names must be unique"
            raise ReferenceProfileError(msg)
        score = DistributionReference.from_dict(_require_mapping(payload, "score"))
        training_rows = _require_int(split, "training_rows", minimum=1)
        scored_rows = _require_int(split, "scored_rows", minimum=1)
        if scored_rows > training_rows:
            msg = "scored_rows must not exceed training_rows"
            raise ReferenceProfileError(msg)
        profile = cls(
            generated_at=generated_at,
            reference_id=reference_id,
            source=_require_str(payload, "source"),
            model_version=_require_str(payload, "model_version"),
            model_sha256=_require_str(payload, "model_sha256"),
            threshold=_require_float(payload, "threshold", minimum=0.0, maximum=1.0),
            split_strategy=_require_str(split, "strategy"),
            split_seed=_require_int(split, "seed", minimum=0),
            training_rows=training_rows,
            scored_rows=scored_rows,
            sample_method=_require_str(split, "sample_method"),
            features=features,
            score=score,
            schema_version=schema_version,
        )
        expected_id = _reference_digest(profile)
        if profile.reference_id != expected_id:
            msg = (
                "reference_id does not match the profile content; the file may have been "
                "edited or generated by an incompatible tool"
            )
            raise ReferenceProfileError(msg)
        return profile


def build_distribution_reference(
    values: ArrayLike,
    *,
    name: str,
    bin_count: int = DEFAULT_BIN_COUNT,
    quantile_sample_size: int = DEFAULT_QUANTILE_SAMPLE_SIZE,
) -> DistributionReference:
    """Build one compact distribution description.

    Missing values are counted but excluded from the bins and quantile sample. A distribution
    with no finite values cannot become a reference and raises rather than producing an empty
    profile.
    """
    array = np.asarray(values, dtype="float64").reshape(-1)
    finite = finite_values(array)
    if finite.size == 0:
        msg = f"reference distribution {name!r} has no finite values"
        raise ReferenceProfileError(msg)
    if quantile_sample_size < 2:
        msg = "quantile_sample_size must be at least 2"
        raise ReferenceProfileError(msg)

    edges = quantile_edges(finite, bin_count=bin_count)
    proportions = bin_proportions(finite, edges)
    sample = np.quantile(finite, np.linspace(0.0, 1.0, quantile_sample_size))
    missing = float(1.0 - (finite.size / array.size))
    return DistributionReference(
        name=name,
        count=int(finite.size),
        missing_rate=missing,
        mean=float(np.mean(finite)),
        bin_edges=tuple(float(value) for value in edges),
        bin_proportions=tuple(float(value) for value in proportions),
        quantile_sample=tuple(float(value) for value in sample),
    )


def build_reference_profile(
    features: pd.DataFrame,
    scores: ArrayLike,
    *,
    feature_columns: Sequence[str],
    source: str,
    model_version: str,
    model_sha256: str,
    threshold: float,
    split_strategy: str,
    split_seed: int,
    training_rows: int,
    sample_method: str,
    generated_at: datetime | None = None,
    bin_count: int = DEFAULT_BIN_COUNT,
    quantile_sample_size: int = DEFAULT_QUANTILE_SAMPLE_SIZE,
) -> ReferenceProfile:
    """Build a reference profile from training features and training scores."""
    resolved_columns = tuple(feature_columns)
    if not resolved_columns:
        msg = "at least one feature column is required"
        raise ReferenceProfileError(msg)
    if len(set(resolved_columns)) != len(resolved_columns):
        msg = "reference feature columns must be unique"
        raise ReferenceProfileError(msg)
    missing = [column for column in resolved_columns if column not in features.columns]
    if missing:
        msg = f"reference feature frame is missing columns: {', '.join(missing)}"
        raise ReferenceProfileError(msg)
    if np.asarray(scores).size == 0:
        msg = "reference scores must not be empty"
        raise ReferenceProfileError(msg)
    if training_rows < 1:
        msg = "training_rows must be at least 1"
        raise ReferenceProfileError(msg)
    if split_seed < 0:
        msg = "split_seed must be at least 0"
        raise ReferenceProfileError(msg)
    if not 0.0 <= threshold <= 1.0:
        msg = "threshold must be in [0, 1]"
        raise ReferenceProfileError(msg)
    resolved_generated_at = generated_at or datetime.now(UTC)
    if resolved_generated_at.tzinfo is None or resolved_generated_at.utcoffset() is None:
        msg = "generated_at must include a timezone"
        raise ReferenceProfileError(msg)

    distributions = tuple(
        build_distribution_reference(
            features.loc[:, column].to_numpy(),
            name=column,
            bin_count=bin_count,
            quantile_sample_size=quantile_sample_size,
        )
        for column in resolved_columns
    )
    score = build_distribution_reference(
        scores,
        name=SCORE_DISTRIBUTION_NAME,
        bin_count=bin_count,
        quantile_sample_size=quantile_sample_size,
    )
    if score.count > training_rows:
        msg = "scored_rows must not exceed training_rows"
        raise ReferenceProfileError(msg)
    profile = ReferenceProfile(
        generated_at=resolved_generated_at,
        reference_id="0" * 64,
        source=source,
        model_version=model_version,
        model_sha256=model_sha256,
        threshold=float(threshold),
        split_strategy=split_strategy,
        split_seed=int(split_seed),
        training_rows=int(training_rows),
        scored_rows=int(score.count),
        sample_method=sample_method,
        features=distributions,
        score=score,
    )
    return ReferenceProfile(
        generated_at=profile.generated_at,
        reference_id=_reference_digest(profile),
        source=profile.source,
        model_version=profile.model_version,
        model_sha256=profile.model_sha256,
        threshold=profile.threshold,
        split_strategy=profile.split_strategy,
        split_seed=profile.split_seed,
        training_rows=profile.training_rows,
        scored_rows=profile.scored_rows,
        sample_method=profile.sample_method,
        features=profile.features,
        score=profile.score,
    )


def write_reference_profile(profile: ReferenceProfile, path: Path) -> Path:
    """Write a profile as UTF-8 JSON, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(profile.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8")
    return path


def load_reference_profile(path: Path) -> ReferenceProfile:
    """Load and validate a reference profile from disk."""
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        msg = f"reference profile not found: {path}"
        raise ReferenceProfileError(msg) from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        msg = f"could not read reference profile {path}: {type(exc).__name__}"
        raise ReferenceProfileError(msg) from exc
    if not isinstance(payload, dict):
        msg = "reference profile must be a JSON object"
        raise ReferenceProfileError(msg)
    return ReferenceProfile.from_dict(payload)


def _reference_digest(profile: ReferenceProfile) -> str:
    """Hash the distribution content without the timestamp and the ID itself."""
    payload = profile.to_dict()
    payload.pop("reference_id", None)
    payload.pop("generated_at", None)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _require_mapping(payload: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    value = payload.get(field)
    return _require_mapping_value(value, field)


def _require_mapping_value(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        msg = f"{field} must be a JSON object"
        raise ReferenceProfileError(msg)
    return value


def _require_list(payload: Mapping[str, Any], field: str) -> list[Any]:
    value = payload.get(field)
    if not isinstance(value, list):
        msg = f"{field} must be a JSON array"
        raise ReferenceProfileError(msg)
    return value


def _require_str(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        msg = f"{field} must be a non-empty string"
        raise ReferenceProfileError(msg)
    return value


def _require_int(payload: Mapping[str, Any], field: str, *, minimum: int) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{field} must be an integer"
        raise ReferenceProfileError(msg)
    if value < minimum:
        msg = f"{field} must be at least {minimum}"
        raise ReferenceProfileError(msg)
    return value


def _require_float(
    payload: Mapping[str, Any],
    field: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    return _require_float_value(payload.get(field), field, minimum=minimum, maximum=maximum)


def _require_float_value(
    value: object,
    field: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f"{field} must be a number"
        raise ReferenceProfileError(msg)
    number = float(value)
    if not math.isfinite(number):
        msg = f"{field} must be finite"
        raise ReferenceProfileError(msg)
    if minimum is not None and number < minimum:
        msg = f"{field} must be at least {minimum}"
        raise ReferenceProfileError(msg)
    if maximum is not None and number > maximum:
        msg = f"{field} must be at most {maximum}"
        raise ReferenceProfileError(msg)
    return number


def _finite_float(value: object, field: str) -> float:
    return _require_float_value(value, field)


def _require_datetime(payload: Mapping[str, Any], field: str) -> datetime:
    value = _require_str(payload, field)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        msg = f"{field} must be an ISO-8601 datetime"
        raise ReferenceProfileError(msg) from exc
    if parsed.tzinfo is None:
        msg = f"{field} must include a timezone"
        raise ReferenceProfileError(msg)
    return parsed


def score_array(scores: Sequence[float]) -> NDArray[np.float64]:
    """Convert scores to a finite float array, failing on non-finite values."""
    return finite_values(np.asarray(scores, dtype="float64"))


__all__ = [
    "REFERENCE_SCHEMA_VERSION",
    "SCORE_DISTRIBUTION_NAME",
    "DistributionReference",
    "ReferenceProfile",
    "ReferenceProfileError",
    "build_distribution_reference",
    "build_reference_profile",
    "load_reference_profile",
    "score_array",
    "write_reference_profile",
]
