"""Reusable inference service for the selected fraud model.

Responsibility
    Load a validated artifact bundle once, check that its fitted preprocessing matches the
    bundle's declared feature schema, score exactly one transaction, apply the frozen binary
    threshold, and optionally explain that transaction with the shared SHAP layer.

What this module deliberately does not do
    - It never fits, calibrates or retrains anything. A scoring request only calls the
      already-fitted ``predict_proba`` and, when requested, the already-built explainer.
    - It never rebuilds preprocessing. The same fitted transformer that travelled inside the
      bundle is the only transform used at serving time.
    - It does not implement the API or a three-way review/block policy. The response carries
      the bundle's binary threshold; ``api.schemas`` wraps this service for FastAPI.

Failures
    A missing, corrupt or incompatible artifact raises :class:`InferenceModelError`.
    A malformed request raises :class:`InferenceInputError`. Both are subclasses of
    :class:`InferenceError`, so an API layer can map model failures to 503 and input failures
    to 422 without catching generic exceptions.

Status
    Implemented for the local validated bundle. Registry alias resolution remains M5 work.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from fraud_risk.config import Settings, get_settings
from fraud_risk.data.dataset import AMOUNT_COLUMN
from fraud_risk.models.bundle import FrozenModel, ModelBundleError, load_model_bundle
from fraud_risk.models.evaluation import binary_decisions
from fraud_risk.models.explainability import (
    DEFAULT_TOP_K,
    ExplanationError,
    ExplanationModel,
    TransactionExplanation,
    explain_transaction,
    load_explanation_model,
)

logger = logging.getLogger(__name__)


class InferenceError(ValueError):
    """An inference request cannot be served honestly."""


class InferenceInputError(InferenceError):
    """The transaction does not satisfy the model's feature contract."""


class InferenceModelError(InferenceError):
    """The model artifact is missing, corrupt or incompatible with this code."""


@dataclass(frozen=True)
class InferenceRequest:
    """One transaction, with features supplied in the model's expected order.

    ``features`` may contain every model feature at once, or every feature except
    ``Amount`` when ``amount`` is supplied separately. The second form matches the canonical
    transaction contract, where amount is a top-level field.
    """

    transaction_id: str
    features: Mapping[str, float]
    amount: float | None = None


@dataclass(frozen=True)
class PredictionResult:
    """The score, threshold and decision for one transaction, plus an optional explanation."""

    transaction_id: str
    probability: float
    threshold: float
    prediction: int
    model_version: str
    model_sha256: str
    model_family: str
    explanation: TransactionExplanation | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise without any feature snapshot or request payload."""
        return {
            "transaction_id": self.transaction_id,
            "probability": float(self.probability),
            "threshold": float(self.threshold),
            "prediction": int(self.prediction),
            "model_version": self.model_version,
            "model_sha256": self.model_sha256,
            "model_family": self.model_family,
            "explanation": (self.explanation.to_dict() if self.explanation is not None else None),
        }


class FraudInferenceService:
    """Lazy, cached inference around one validated artifact bundle.

    The artifact path is configurable directly or through
    :attr:`fraud_risk.config.Settings.model_artifact_path`. Construction performs no I/O;
    the first :meth:`predict` loads and validates the bundle, and subsequent requests reuse
    the same fitted pipeline and explainer objects.
    """

    def __init__(
        self,
        artifact_path: Path | str | None = None,
        *,
        settings: Settings | None = None,
    ) -> None:
        if artifact_path is None:
            resolved_settings = settings if settings is not None else get_settings()
            artifact_path = resolved_settings.model_artifact_path
        self._artifact_path = Path(artifact_path)
        self._model: FrozenModel | None = None
        self._explanation_model: ExplanationModel | None = None

    @property
    def artifact_path(self) -> Path:
        """The configured artifact location, whether or not it has been loaded yet."""
        return self._artifact_path

    @property
    def model(self) -> FrozenModel:
        """Return the validated bundle, loading it on first use."""
        return self._load_model()

    def clear_cache(self) -> None:
        """Drop the cached pipeline and explainer so the next request reloads the artifact."""
        self._model = None
        self._explanation_model = None

    def predict(
        self,
        request: InferenceRequest,
        *,
        explain: bool = False,
        top_k: int = DEFAULT_TOP_K,
    ) -> PredictionResult:
        """Score one transaction without fitting anything.

        ``explain=True`` builds and caches the SHAP explainer for the selected XGBoost
        bundle. Other families still score normally, but a requested explanation fails
        loudly rather than silently returning a different output.
        """
        model = self._load_model()
        features = _build_feature_frame(model, request)
        probability = _predict_probability(model, features)
        prediction = int(
            binary_decisions(
                np.asarray([probability], dtype="float64"),
                threshold=model.threshold,
            )[0]
        )

        explanation: TransactionExplanation | None = None
        if explain:
            explanation = self._explain(features, top_k=top_k)

        return PredictionResult(
            transaction_id=request.transaction_id,
            probability=probability,
            threshold=float(model.threshold),
            prediction=prediction,
            model_version=model.model_version,
            model_sha256=model.sha256,
            model_family=model.family or "unknown",
            explanation=explanation,
        )

    def _load_model(self) -> FrozenModel:
        if self._model is None:
            try:
                model = load_model_bundle(self._artifact_path)
            except ModelBundleError as exc:
                msg = f"model artifact unavailable at {self._artifact_path}: {exc}"
                raise InferenceModelError(msg) from exc
            _validate_pipeline_compatibility(model)
            self._model = model
            logger.info(
                "loaded inference model path=%s version=%s sha256=%s",
                model.path,
                model.model_version,
                model.sha256[:12],
            )
        return self._model

    def _explain(
        self,
        features: pd.DataFrame,
        *,
        top_k: int,
    ) -> TransactionExplanation:
        if self._explanation_model is None:
            # Load the bundle first so a missing artifact is reported as a model failure,
            # not as an explanation failure.
            self._load_model()
            try:
                self._explanation_model = load_explanation_model(self._artifact_path)
            except ExplanationError as exc:
                msg = f"the selected model cannot produce a requested explanation: {exc}"
                raise InferenceError(msg) from exc
        try:
            return explain_transaction(self._explanation_model, features, top_k=top_k)
        except ExplanationError as exc:
            msg = f"could not explain the transaction: {exc}"
            raise InferenceError(msg) from exc


def _validate_pipeline_compatibility(model: FrozenModel) -> None:
    """Fail unless the fitted transform matches the bundle's feature schema exactly."""
    pipeline = model.pipeline
    if not isinstance(pipeline, Pipeline):
        msg = "the bundle pipeline must be a sklearn Pipeline with 'preprocess' and 'classifier'"
        raise InferenceModelError(msg)

    steps = pipeline.named_steps
    if "preprocess" not in steps or "classifier" not in steps:
        msg = "the bundle pipeline must contain 'preprocess' and 'classifier' steps"
        raise InferenceModelError(msg)

    preprocessor = steps["preprocess"]
    try:
        transformed_names = tuple(str(name) for name in preprocessor.get_feature_names_out())
    except Exception as exc:
        msg = f"the fitted preprocessor does not expose feature names: {exc}"
        raise InferenceModelError(msg) from exc
    if transformed_names != model.feature_columns:
        msg = (
            "the fitted preprocessor output order does not match the bundle feature schema "
            f"({len(transformed_names)} vs {len(model.feature_columns)} columns). "
            "Training and inference preprocessing are incompatible."
        )
        raise InferenceModelError(msg)

    input_names = getattr(preprocessor, "feature_names_in_", None)
    if input_names is not None:
        resolved_inputs = tuple(str(name) for name in input_names)
        if resolved_inputs != model.feature_columns:
            msg = (
                "the fitted preprocessor was fitted on a different input feature order than "
                f"the bundle declares; expected {list(model.feature_columns)!r}, got "
                f"{list(resolved_inputs)!r}"
            )
            raise InferenceModelError(msg)

    classifier = steps["classifier"]
    expected_features = len(model.feature_columns)
    fitted_features = getattr(classifier, "n_features_in_", None)
    if fitted_features is not None and int(fitted_features) != expected_features:
        msg = (
            f"the fitted classifier expects {fitted_features} transformed features, but the "
            f"bundle declares {expected_features}"
        )
        raise InferenceModelError(msg)

    classes = getattr(classifier, "classes_", None)
    if classes is not None and list(np.asarray(classes).tolist()) != [0, 1]:
        msg = "the fitted classifier must expose binary classes [0, 1]"
        raise InferenceModelError(msg)


def _build_feature_frame(model: FrozenModel, request: InferenceRequest) -> pd.DataFrame:
    """Validate one request and build a one-row frame in the bundle's exact column order."""
    if not isinstance(request.transaction_id, str) or not request.transaction_id.strip():
        msg = "transaction_id must be a non-empty string"
        raise InferenceInputError(msg)

    amount = request.amount
    if amount is not None:
        if AMOUNT_COLUMN not in model.feature_columns:
            msg = f"this model does not use {AMOUNT_COLUMN!r}; omit the amount field"
            raise InferenceInputError(msg)
        _finite_number(amount, "amount")
        if amount < 0.0:
            msg = f"amount must be non-negative (got {amount!r})"
            raise InferenceInputError(msg)

    expected = tuple(
        column
        for column in model.feature_columns
        if not (column == AMOUNT_COLUMN and amount is not None)
    )
    provided = tuple(request.features.keys())
    if set(provided) != set(expected):
        missing = [column for column in expected if column not in request.features]
        unexpected = [column for column in provided if column not in expected]
        parts: list[str] = []
        if missing:
            parts.append(f"missing: {', '.join(missing)}")
        if unexpected:
            parts.append(f"unexpected: {', '.join(unexpected)}")
        msg = f"feature schema mismatch ({'; '.join(parts)})"
        raise InferenceInputError(msg)
    if provided != expected:
        msg = (
            "feature order does not match the model schema; expected "
            f"{list(expected)!r}, got {list(provided)!r}"
        )
        raise InferenceInputError(msg)

    values: dict[str, float] = {}
    for column in model.feature_columns:
        if column == AMOUNT_COLUMN and amount is not None:
            values[column] = _finite_number(amount, "amount")
        else:
            values[column] = _finite_number(
                request.features[column],
                f"feature {column!r}",
            )
    return pd.DataFrame([values], columns=list(model.feature_columns))


def _predict_probability(model: FrozenModel, features: pd.DataFrame) -> float:
    """Call the fitted pipeline once; never fit, calibrate or mutate it."""
    try:
        raw = model.pipeline.predict_proba(features)
    except Exception as exc:
        msg = f"the fitted pipeline failed while scoring the transaction: {exc}"
        raise InferenceModelError(msg) from exc
    probabilities = np.asarray(raw, dtype="float64")
    if probabilities.shape != (1, 2):
        msg = f"expected a one-row, two-column probability matrix; got {probabilities.shape}"
        raise InferenceModelError(msg)
    probability = float(probabilities[0, 1])
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        msg = f"the model returned an invalid fraud probability: {probability!r}"
        raise InferenceModelError(msg)
    return probability


def _finite_number(value: object, name: str) -> float:
    """Return a finite float, rejecting booleans, non-numbers, NaN and infinity."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        msg = f"{name} must be a number (got {value!r})"
        raise InferenceInputError(msg)
    number = float(value)
    if not math.isfinite(number):
        msg = f"{name} must be finite (got {value!r})"
        raise InferenceInputError(msg)
    return number


__all__ = [
    "FraudInferenceService",
    "InferenceError",
    "InferenceInputError",
    "InferenceModelError",
    "InferenceRequest",
    "PredictionResult",
]
