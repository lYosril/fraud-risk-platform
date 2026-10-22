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
    Request/response schemas implemented. FastAPI route wiring remains M6 work.
"""

from __future__ import annotations

import math
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

from fraud_risk.data.dataset import AMOUNT_COLUMN, TARGET_COLUMN
from fraud_risk.models.inference import InferenceRequest, PredictionResult

#: The label names that must never appear on an inference payload.
LABEL_FIELDS = frozenset({TARGET_COLUMN, "is_fraud"})


class PredictionRequest(BaseModel):
    """One transaction submitted for scoring.

    ``features`` carries the anonymised model inputs (``V1..V28`` for the reference dataset)
    in the model's declared order. ``amount`` is a top-level field because it is the only
    model feature with a direct business meaning and its own non-negative constraint.
    """

    model_config = ConfigDict(extra="forbid")

    transaction_id: str = Field(min_length=1, max_length=128)
    amount: float = Field(ge=0.0, allow_inf_nan=False)
    features: dict[str, float] = Field(min_length=1)
    explain: bool = False

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


__all__ = [
    "LABEL_FIELDS",
    "ExplanationFeature",
    "PredictionExplanation",
    "PredictionRequest",
    "PredictionResponse",
]
