"""Create monitoring event and drift report tables.

Revision ID: 0002_create_monitoring
Revises: 0001_create_predictions
Create Date: 2026-10-23
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002_create_monitoring"
down_revision: str | None = "0001_create_predictions"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create count-only validation events and persisted aggregate drift reports."""
    op.create_table(
        "monitoring_events",
        sa.Column("event_id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("endpoint", sa.String(length=256), nullable=False),
        sa.Column("reason_code", sa.String(length=64), nullable=False),
        sa.Column(
            "failure_count",
            sa.Integer(),
            server_default=sa.text("1"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "event_type IN ('input_validation_failure')",
            name="event_type_allowed",
        ),
        sa.CheckConstraint("failure_count >= 1", name="failure_count_positive"),
        sa.PrimaryKeyConstraint("event_id", name="pk_monitoring_events"),
    )
    op.create_index(
        "ix_monitoring_events_created_at",
        "monitoring_events",
        ["created_at"],
        unique=False,
    )

    op.create_table(
        "drift_reports",
        sa.Column("report_id", sa.Uuid(), nullable=False),
        sa.Column(
            "generated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("window_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("reference_id", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("data_drift_status", sa.String(length=16), nullable=False),
        sa.Column("data_quality_status", sa.String(length=16), nullable=False),
        sa.Column("prediction_count", sa.Integer(), nullable=False),
        sa.Column("validation_failure_count", sa.Integer(), nullable=False),
        sa.Column("feature_snapshot_count", sa.Integer(), nullable=False),
        sa.Column("feature_drift_count", sa.Integer(), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.CheckConstraint(
            "status IN ('ok', 'warning', 'alert', 'unavailable')",
            name="status_allowed",
        ),
        sa.CheckConstraint(
            "data_drift_status IN ('ok', 'warning', 'alert', 'unavailable')",
            name="data_drift_status_allowed",
        ),
        sa.CheckConstraint(
            "data_quality_status IN ('ok', 'warning', 'alert', 'unavailable')",
            name="data_quality_status_allowed",
        ),
        sa.CheckConstraint("prediction_count >= 0", name="prediction_count_non_negative"),
        sa.CheckConstraint(
            "validation_failure_count >= 0",
            name="validation_failure_count_non_negative",
        ),
        sa.CheckConstraint(
            "feature_snapshot_count >= 0",
            name="feature_snapshot_count_non_negative",
        ),
        sa.CheckConstraint("feature_drift_count >= 0", name="feature_drift_count_non_negative"),
        sa.CheckConstraint("window_end > window_start", name="window_order"),
        sa.PrimaryKeyConstraint("report_id", name="pk_drift_reports"),
    )
    op.create_index(
        "ix_drift_reports_generated_at",
        "drift_reports",
        ["generated_at"],
        unique=False,
    )


def downgrade() -> None:
    """Drop the monitoring tables and their indexes."""
    op.drop_index("ix_drift_reports_generated_at", table_name="drift_reports")
    op.drop_table("drift_reports")
    op.drop_index("ix_monitoring_events_created_at", table_name="monitoring_events")
    op.drop_table("monitoring_events")
