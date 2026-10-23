"""ORM models for prediction and monitoring persistence.

Responsibility
    Persist the prediction metadata the API and dashboard need: a UUID, timestamp,
    model identity, score, applied threshold, binary decision and analyst review state.
    Raw transaction features are *not* stored unless the operator explicitly enables the
    opt-in snapshot, which is off by default.

    Monitoring adds two aggregate-only tables: ``monitoring_events`` records count-only input
    validation failures, and ``drift_reports`` stores one machine-readable drift report per
    window. Neither table stores a request payload or a transaction feature row.

What this module deliberately does not do
    It does not create the schema. Alembic owns migrations; the application never calls
    ``metadata.create_all()``. Feedback labels remain a later milestone; analyst review state
    on ``predictions`` is not a replacement for complete delayed ground truth.

Status
    ``predictions``, ``monitoring_events`` and ``drift_reports`` are implemented; see
    ``docs/ROADMAP.md`` (M8).
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import Boolean, CheckConstraint, DateTime, Float, Integer, String, func
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


class MonitoringEvent(Base):
    """Count-only input validation failures, with no payload or identifier.

    One row aggregates an event count for an endpoint and safe reason code. This keeps the
    write volume bounded when a client sends many malformed requests, and means the API never
    needs to persist a rejected body.
    """

    __tablename__ = "monitoring_events"
    __table_args__ = (
        CheckConstraint(
            "event_type IN ('input_validation_failure')",
            name="event_type_allowed",
        ),
        CheckConstraint("failure_count >= 1", name="failure_count_positive"),
    )

    event_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        server_default=func.now(),
        nullable=False,
        index=True,
    )
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    endpoint: Mapped[str] = mapped_column(String(256), nullable=False)
    reason_code: Mapped[str] = mapped_column(String(64), nullable=False)
    failure_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)


class DriftReport(Base):
    """One persisted monitoring run with its aggregate summary and full JSON report."""

    __tablename__ = "drift_reports"
    __table_args__ = (
        CheckConstraint(
            "status IN ('ok', 'warning', 'alert', 'unavailable')",
            name="status_allowed",
        ),
        CheckConstraint(
            "data_drift_status IN ('ok', 'warning', 'alert', 'unavailable')",
            name="data_drift_status_allowed",
        ),
        CheckConstraint(
            "data_quality_status IN ('ok', 'warning', 'alert', 'unavailable')",
            name="data_quality_status_allowed",
        ),
        CheckConstraint("prediction_count >= 0", name="prediction_count_non_negative"),
        CheckConstraint(
            "validation_failure_count >= 0",
            name="validation_failure_count_non_negative",
        ),
        CheckConstraint(
            "feature_snapshot_count >= 0",
            name="feature_snapshot_count_non_negative",
        ),
        CheckConstraint("feature_drift_count >= 0", name="feature_drift_count_non_negative"),
        CheckConstraint("window_end > window_start", name="window_order"),
    )

    report_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        server_default=func.now(),
        nullable=False,
        index=True,
    )
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    window_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    reference_id: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    data_drift_status: Mapped[str] = mapped_column(String(16), nullable=False)
    data_quality_status: Mapped[str] = mapped_column(String(16), nullable=False)
    prediction_count: Mapped[int] = mapped_column(Integer, nullable=False)
    validation_failure_count: Mapped[int] = mapped_column(Integer, nullable=False)
    feature_snapshot_count: Mapped[int] = mapped_column(Integer, nullable=False)
    feature_drift_count: Mapped[int] = mapped_column(Integer, nullable=False)
    # Aggregate report only: no raw transaction, identifier or feature row is stored here.
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)


class PerformanceReport(Base):
    """One persisted delayed-label performance run with its aggregate JSON report."""

    __tablename__ = "performance_reports"
    __table_args__ = (
        CheckConstraint("row_count > 0", name="row_count_positive"),
        CheckConstraint(
            "model_version_count >= 1",
            name="model_version_count_positive",
        ),
        CheckConstraint("segment_count >= 1", name="segment_count_positive"),
        CheckConstraint(
            "threshold >= 0 AND threshold <= 1",
            name="threshold_range",
        ),
        CheckConstraint(
            "period IN ('day', 'week', 'month')",
            name="period_allowed",
        ),
        CheckConstraint("window_end >= window_start", name="window_order"),
    )

    report_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        server_default=func.now(),
        nullable=False,
        index=True,
    )
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    window_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    model_version_count: Mapped[int] = mapped_column(Integer, nullable=False)
    segment_count: Mapped[int] = mapped_column(Integer, nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    period: Mapped[str] = mapped_column(String(8), nullable=False)
    labels_verified: Mapped[bool] = mapped_column(Boolean, nullable=False)
    label_source: Mapped[str] = mapped_column(String(128), nullable=False)
    # Aggregate report only: no feature row, transaction identifier or raw label is stored.
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)


__all__ = [
    "DriftReport",
    "MonitoringEvent",
    "PerformanceReport",
    "Prediction",
    "PredictionDecision",
    "ReviewStatus",
]
