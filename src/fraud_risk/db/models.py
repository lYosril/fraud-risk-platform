"""ORM models for prediction persistence.

Responsibility
    Persist the prediction metadata the API and future dashboard need: a UUID, timestamp,
    model identity, score, applied threshold, binary decision and analyst review state.
    Raw transaction features are *not* stored unless the operator explicitly enables the
    opt-in snapshot, which is off by default.

What this module deliberately does not do
    It does not create the schema. Alembic owns migrations; the application never calls
    ``metadata.create_all()``. It also does not implement feedback labels or drift reports,
    which remain later milestones.

Status
    ``predictions`` is implemented. ``feedback_labels`` and ``drift_reports`` are still
    planned; see ``docs/ROADMAP.md``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import CheckConstraint, DateTime, Float, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from fraud_risk.db.base import Base


class PredictionDecision(StrEnum):
    """The binary model decision at the frozen threshold."""

    LEGITIMATE = "legitimate"
    FRAUD = "fraud"


class ReviewStatus(StrEnum):
    """Analyst review state for a persisted prediction."""

    PENDING = "pending"
    CONFIRMED_FRAUD = "confirmed_fraud"
    CONFIRMED_LEGITIMATE = "confirmed_legitimate"


def _utcnow() -> datetime:
    """Return a timezone-aware UTC timestamp for ORM inserts."""
    return datetime.now(UTC)


class Prediction(Base):
    """One persisted prediction, with no transaction features by default."""

    __tablename__ = "predictions"
    __table_args__ = (
        CheckConstraint(
            "risk_score >= 0 AND risk_score <= 1",
            name="risk_score_range",
        ),
        CheckConstraint(
            "threshold >= 0 AND threshold <= 1",
            name="threshold_range",
        ),
        CheckConstraint(
            "decision IN ('legitimate', 'fraud')",
            name="decision_allowed",
        ),
        CheckConstraint(
            "review_status IN ('pending', 'confirmed_fraud', 'confirmed_legitimate')",
            name="review_status_allowed",
        ),
    )

    prediction_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    transaction_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        server_default=func.now(),
        nullable=False,
        index=True,
    )
    model_version: Mapped[str] = mapped_column(String(128), nullable=False)
    model_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    decision: Mapped[str] = mapped_column(String(16), nullable=False)
    review_status: Mapped[str] = mapped_column(
        String(32),
        default=ReviewStatus.PENDING.value,
        server_default=ReviewStatus.PENDING.value,
        nullable=False,
    )
    # Opt-in only (PERSIST_FEATURE_SNAPSHOT=true). Kept nullable so the default database
    # stores no input vector at all. Never include a PAN or account reference here.
    feature_snapshot: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)


__all__ = ["Prediction", "PredictionDecision", "ReviewStatus"]
