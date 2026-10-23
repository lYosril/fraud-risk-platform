"""Create the delayed-label performance report table.

Revision ID: 0003_create_performance_reports
Revises: 0002_create_monitoring
Create Date: 2026-10-23
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003_create_performance_reports"
down_revision: str | None = "0002_create_monitoring"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create aggregate-only performance reports for delayed labels."""
    op.create_table(
        "performance_reports",
        sa.Column("report_id", sa.Uuid(), nullable=False),
        sa.Column(
            "generated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("window_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("row_count", sa.Integer(), nullable=False),
        sa.Column("model_version_count", sa.Integer(), nullable=False),
        sa.Column("segment_count", sa.Integer(), nullable=False),
        sa.Column("threshold", sa.Float(), nullable=False),
        sa.Column("period", sa.String(length=8), nullable=False),
        sa.Column("labels_verified", sa.Boolean(), nullable=False),
        sa.Column("label_source", sa.String(length=128), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.CheckConstraint("row_count > 0", name="row_count_positive"),
        sa.CheckConstraint(
            "model_version_count >= 1",
            name="model_version_count_positive",
        ),
        sa.CheckConstraint("segment_count >= 1", name="segment_count_positive"),
        sa.CheckConstraint(
            "threshold >= 0 AND threshold <= 1",
            name="threshold_range",
        ),
        sa.CheckConstraint(
            "period IN ('day', 'week', 'month')",
            name="period_allowed",
        ),
        sa.CheckConstraint("window_end >= window_start", name="window_order"),
        sa.PrimaryKeyConstraint("report_id", name="pk_performance_reports"),
    )
    op.create_index(
        "ix_performance_reports_generated_at",
        "performance_reports",
        ["generated_at"],
        unique=False,
    )


def downgrade() -> None:
    """Drop the performance report table and its index."""
    op.drop_index(
        "ix_performance_reports_generated_at",
        table_name="performance_reports",
    )
    op.drop_table("performance_reports")
