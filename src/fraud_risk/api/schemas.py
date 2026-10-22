"""API request and response schemas.

Responsibility
    Pydantic v2 models mirroring the transaction and prediction contracts in
    ``docs/ARCHITECTURE.md`` §5. These are a trust boundary: they are the only place
    untrusted input becomes typed data, and they can be reused directly by FastAPI routes.

Rules
    - Reject label fields (``is_fraud`` / the dataset's ``Class``) on the inference path.
    - Reject unknown top-level fields instead of silently dropping them.
    - Validate ranges (amount >= 0, score in [0, 1]) rather than clamping silently.
    - The feature mapping is order-sensitive: the inference service compares it with the
      artifact's exact schema order, so a reordered JSON object is rejected rather than
      silently realigned.
    - Error responses must not echo secrets or full payloads.

Status
    Request/response, batch, computed-decision, error, persisted-history and review schemas are
    implemented and used by the versioned FastAPI routes.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator

from fraud_risk.data.dataset import AMOUNT_COLUMN, TARGET_COLUMN
from fraud_risk.db.models import PredictionDecision, ReviewStatus
from fraud_risk.models.inference import InferenceRequest, PredictionResult

#: The label names that must never appear on an inference payload.
LABEL_FIELDS = frozenset({TARGET_COLUMN, "is_fraud"})


class HealthResponse(BaseModel):
    """Process liveness: the API is running and can answer requests."""

    status: Literal["ok"]
    service: str
    version: str
    environment: str


class ReadinessResponse(BaseModel):
    """Whether this process can actually serve a prediction right now.

    Deliberately exposes availability only. Model identity and the feature schema live behind
    the authenticated ``GET /model`` route, so an unauthenticated probe cannot enumerate them.
    """

    status: Literal["ready", "not_ready"]
    model_available: bool
    detail: str | None = None


class ModelMetadataResponse(BaseModel):
    """The loaded model's identity and serving contract, without exposing its path."""

    model_name: str
    model_version: str
    model_sha256: str
    model_family: str
    threshold: float = Field(allow_inf_nan=False)
    feature_count: int = Field(ge=1)
    feature_columns: list[str]


class ErrorResponse(BaseModel):
    """The stable, non-leaking error shape used by the HTTP layer."""

    detail: str


class PredictionRecordResponse(BaseModel):
    """Persisted prediction metadata returned by the authenticated history route.

    ``feature_snapshot`` is intentionally absent even when persistence is configured to store
    it: raw inputs are not needed to read a prediction back and must not leave the trust
    boundary through this API.
    """

    model_config = ConfigDict(from_attributes=True)

    prediction_id: UUID
    transaction_id: str
    created_at: datetime
    model_version: str
    model_sha256: str
    risk_score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    threshold: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    decision: PredictionDecision
    review_status: ReviewStatus


class ReviewUpdateRequest(BaseModel):
    """An analyst review transition for one persisted prediction."""

    model_config = ConfigDict(extra="forbid")

    review_status: ReviewStatus


class PredictionRequest(BaseModel):
    """One transaction submitted for scoring.

    ``features`` carries the anonymised model inputs (``V1..V28`` for the reference dataset)
    in the model's declared order. ``amount`` is a top-level field because it is the only
    model feature with a direct business meaning and its own non-negative constraint.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    transaction_id: str = Field(min_length=1, max_length=128)
    amount: float = Field(ge=0.0, allow_inf_nan=False)
    features: dict[str, float] = Field(min_length=1)
    explain: bool = False

    @field_validator("transaction_id")
    @classmethod
    def _validate_transaction_id(cls, value: str) -> str:
        """Reject blank identifiers and control characters that could poison logs or rows."""
        if not value.strip():
            msg = "transaction_id must not be blank"
            raise ValueError(msg)
        if any(ord(char) < 32 or ord(char) == 127 for char in value):
            msg = "transaction_id must not contain control characters"
            raise ValueError(msg)
        return value

    @field_validator("features")
    @classmethod
    def _validate_features(cls, value: dict[str, float]) -> dict[str, float]:
        """Reject labels, non-finite values and a duplicated amount field."""
        for name, feature_value in value.items():
            if not name:
                msg = "feature names must be non-empty strings"
                raise ValueError(msg)
            if name in LABEL_FIELDS:
                msg = f"label field {name!r} must never appear in inference features"
                raise ValueError(msg)
            if name == AMOUNT_COLUMN:
                msg = f"supply {AMOUNT_COLUMN!r} as the top-level amount field, not inside features"
                raise ValueError(msg)
            if not math.isfinite(feature_value):
                msg = f"feature {name!r} must be finite"
                raise ValueError(msg)
        return value

    def to_inference_request(self) -> InferenceRequest:
        """Convert the validated request into the reusable service input."""
        return InferenceRequest(
            transaction_id=self.transaction_id,
            features=dict(self.features),
            amount=self.amount,
        )


class BatchPredictionRequest(BaseModel):
    """A bounded list of raw transaction objects for batch scoring.

    Items stay deliberately unparsed at this level so one malformed transaction cannot reject
    the whole batch. The handler validates each object independently and returns a result or a
    safe per-item error at the same index. ``Settings.batch_max_size`` is enforced by the route
    because the maximum is deployment configuration, not a static schema constant.
    """

    model_config = ConfigDict(extra="forbid")

    items: list[dict[str, Any]] = Field(
        min_length=1,
        description="Raw transaction objects; each is validated independently by the handler.",
    )


class ExplanationFeature(BaseModel):
    """One model feature's contribution to the explained transaction output."""

    feature: str
    rank: int = Field(ge=1)
    raw_value: float
    model_value: float
    contribution: float
    abs_contribution: float = Field(ge=0.0)
    direction: str


class PredictionExplanation(BaseModel):
    """Optional top-k SHAP attribution returned alongside a score."""

    output_space: str
    explainer: str
    feature_perturbation: str
    aggregation: str
    n_folds: int = Field(ge=1)
    top_k: int = Field(ge=0)
    base_value: float
    output_value: float
    other_features_contribution: float
    additivity_residual: float
    contributions: list[ExplanationFeature]
    limitations: list[str]


class PredictionResponse(BaseModel):
    """The score, frozen threshold, binary decision and model identity."""

    transaction_id: str
    probability: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    threshold: float = Field(allow_inf_nan=False)
    prediction: int = Field(ge=0, le=1)
    model_version: str
    model_sha256: str
    model_family: str
    explanation: PredictionExplanation | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def decision(self) -> Literal["legitimate", "fraud"]:
        """Human-readable interpretation of the binary model decision."""
        return "fraud" if self.prediction == 1 else "legitimate"

    @classmethod
    def from_result(cls, result: PredictionResult) -> Self:
        """Build the response from the reusable inference-service result."""
        explanation: PredictionExplanation | None = None
        if result.explanation is not None:
            transaction = result.explanation
            explanation = PredictionExplanation(
                output_space=transaction.output_space,
                explainer=transaction.explainer,
                feature_perturbation=transaction.feature_perturbation,
                aggregation=transaction.aggregation,
                n_folds=transaction.n_folds,
                top_k=transaction.top_k,
                base_value=transaction.base_value,
                output_value=transaction.output_value,
                other_features_contribution=transaction.other_features_contribution,
                additivity_residual=transaction.additivity_residual,
                contributions=[
                    ExplanationFeature(
                        feature=item.feature,
                        rank=item.rank,
                        raw_value=item.raw_value,
                        model_value=item.model_value,
                        contribution=item.contribution,
                        abs_contribution=item.abs_contribution,
                        direction=item.direction.value,
                    )
                    for item in transaction.contributions
                ],
                limitations=list(transaction.limitations),
            )
        return cls(
            transaction_id=result.transaction_id,
            probability=result.probability,
            threshold=result.threshold,
            prediction=result.prediction,
            model_version=result.model_version,
            model_sha256=result.model_sha256,
            model_family=result.model_family,
            explanation=explanation,
        )


class BatchPredictionItemResult(BaseModel):
    """One batch slot: either a prediction or a safe per-item validation error."""

    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    transaction_id: str | None = None
    status: Literal["ok", "error"]
    prediction: PredictionResponse | None = None
    error: str | None = None

    @model_validator(mode="after")
    def _validate_payload_matches_status(self) -> Self:
        """Require exactly one of ``prediction`` and ``error`` for a given status."""
        if self.status == "ok":
            if self.prediction is None or self.error is not None:
                msg = "an ok batch result must carry a prediction and no error"
                raise ValueError(msg)
        elif self.prediction is not None or self.error is None:
            msg = "an error batch result must carry an error and no prediction"
            raise ValueError(msg)
        return self


class BatchPredictionResponse(BaseModel):
    """The loaded model identity plus one result per submitted item, in input order."""

    model_config = ConfigDict(extra="forbid")

    model_version: str
    model_sha256: str
    model_family: str
    threshold: float = Field(allow_inf_nan=False)
    total: int = Field(ge=1)
    succeeded: int = Field(ge=0)
    failed: int = Field(ge=0)
    results: list[BatchPredictionItemResult] = Field(min_length=1)


__all__ = [
    "LABEL_FIELDS",
    "BatchPredictionItemResult",
    "BatchPredictionRequest",
    "BatchPredictionResponse",
    "ErrorResponse",
    "ExplanationFeature",
    "HealthResponse",
    "ModelMetadataResponse",
    "PredictionExplanation",
    "PredictionRecordResponse",
    "PredictionRequest",
    "PredictionResponse",
    "ReadinessResponse",
    "ReviewUpdateRequest",
]
