"""Artifact bundle layout and validated loading.

Responsibility
    Own the on-disk contract shared by the final-evaluation stage and the explainability
    layer: a self-contained mapping containing the fitted pipeline, the feature schema, the
    frozen threshold, the cost policy and model metadata. Loading validates the schema and
    returns a typed record; it never reads a dataset or a split.

Why a separate module
    ``final_evaluation`` and ``explainability`` both consume the same bundle. Keeping the
    loader here avoids one stage importing another and keeps the artifact contract in one
    place while the registry is still scaffolded.

Rules
    - A missing, corrupt or schema-incompatible bundle raises :class:`ModelBundleError`.
    - The loader does not decide whether a bundle is promoted or how it is scored.
    - The returned pipeline is the exact fitted object from the bundle; callers must not
      rebuild preprocessing.

Status
    Implemented for the local comparison/final-evaluation bundle. ``model_version`` is the
    bundle's ``run_fingerprint`` when present, otherwise the artifact SHA-256. Registry aliases
    and MLflow version resolution remain M5 work.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import joblib

from fraud_risk.models.evaluation import CostPolicy

DEFAULT_COMPARISON_MODEL_PATH = Path("artifacts") / "comparison" / "model.joblib"
DEFAULT_FROZEN_MODEL_PATH = Path("artifacts") / "final_evaluation" / "frozen_model.joblib"

BUNDLE_SCHEMA_VERSION = 2
BUNDLE_KEYS = frozenset(
    {"schema_version", "seed", "feature_columns", "threshold", "cost_policy", "model", "pipeline"}
)


class ModelBundleError(ValueError):
    """The artifact bundle is missing, corrupt or incompatible with this code."""


@dataclass(frozen=True)
class FrozenModel:
    """A validated model bundle loaded from disk."""

    path: Path
    sha256: str
    model_version: str
    pipeline: Any
    feature_columns: tuple[str, ...]
    threshold: float
    cost_policy: CostPolicy
    seed: int
    metadata: dict[str, Any]

    @property
    def family(self) -> str | None:
        """The estimator family recorded by training, when the bundle has one."""
        family = self.metadata.get("family")
        return family if isinstance(family, str) else None


def load_model_bundle(path: Path = DEFAULT_COMPARISON_MODEL_PATH) -> FrozenModel:
    """Load and validate a model bundle without touching any dataset split."""
    path = Path(path)
    if not path.is_file():
        msg = f"model bundle not found: {path}"
        raise ModelBundleError(msg)

    try:
        raw = joblib.load(path)
    except Exception as exc:
        msg = f"could not load model bundle {path}: {exc}"
        raise ModelBundleError(msg) from exc

    if not isinstance(raw, dict):
        msg = f"model bundle {path} must contain a mapping"
        raise ModelBundleError(msg)
    record = cast(dict[str, Any], raw)
    missing = BUNDLE_KEYS.difference(record)
    if missing:
        msg = f"model bundle {path} is missing keys: {sorted(missing)}"
        raise ModelBundleError(msg)
    if record["schema_version"] != BUNDLE_SCHEMA_VERSION:
        msg = (
            f"model bundle schema_version must be {BUNDLE_SCHEMA_VERSION} "
            f"(got {record['schema_version']!r})"
        )
        raise ModelBundleError(msg)

    pipeline = record["pipeline"]
    if not callable(getattr(pipeline, "predict_proba", None)):
        msg = "model bundle pipeline has no callable predict_proba"
        raise ModelBundleError(msg)

    raw_columns = record["feature_columns"]
    if not isinstance(raw_columns, (list, tuple)) or not raw_columns:
        msg = "model bundle feature_columns must be a non-empty list"
        raise ModelBundleError(msg)
    if not all(isinstance(column, str) and column for column in raw_columns):
        msg = "model bundle feature_columns must contain non-empty strings"
        raise ModelBundleError(msg)
    feature_columns = tuple(str(column) for column in raw_columns)
    if len(set(feature_columns)) != len(feature_columns):
        msg = "model bundle feature_columns must be unique"
        raise ModelBundleError(msg)

    seed = record["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        msg = f"model bundle seed must be an int in [0, 2**32) (got {seed!r})"
        raise ModelBundleError(msg)

    threshold = finite_number(record["threshold"], "threshold")
    raw_policy = record["cost_policy"]
    if not isinstance(raw_policy, dict):
        msg = "model bundle cost_policy must be a mapping"
        raise ModelBundleError(msg)
    try:
        cost_policy = CostPolicy(
            false_negative_cost=float(raw_policy["false_negative_cost"]),
            false_positive_cost=float(raw_policy["false_positive_cost"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        msg = "model bundle cost_policy is invalid"
        raise ModelBundleError(msg) from exc

    metadata = record["model"]
    if not isinstance(metadata, dict):
        msg = "model bundle model metadata must be a mapping"
        raise ModelBundleError(msg)

    digest = sha256_file(path)
    raw_version = record.get("run_fingerprint")
    if raw_version is None:
        # Bundles produced before MLflow tracking have no run fingerprint. The content hash
        # is a stable substitute: it changes whenever the artifact bytes change.
        model_version = digest
    elif isinstance(raw_version, str) and raw_version.strip():
        model_version = raw_version.strip()
    else:
        msg = "model bundle run_fingerprint must be a non-empty string when present"
        raise ModelBundleError(msg)

    return FrozenModel(
        path=path,
        sha256=digest,
        model_version=model_version,
        pipeline=pipeline,
        feature_columns=feature_columns,
        threshold=threshold,
        cost_policy=cost_policy,
        seed=seed,
        metadata=dict(metadata),
    )


def finite_number(value: object, name: str) -> float:
    """Return a finite float or raise with the offending field name."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f"{name} must be a number (got {value!r})"
        raise ModelBundleError(msg)
    number = float(value)
    if not math.isfinite(number):
        msg = f"{name} must be finite (got {value!r})"
        raise ModelBundleError(msg)
    return number


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "BUNDLE_KEYS",
    "BUNDLE_SCHEMA_VERSION",
    "DEFAULT_COMPARISON_MODEL_PATH",
    "DEFAULT_FROZEN_MODEL_PATH",
    "FrozenModel",
    "ModelBundleError",
    "finite_number",
    "load_model_bundle",
    "sha256_file",
]
