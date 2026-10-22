"""Create the prediction persistence table.

Revision ID: 0001_create_predictions
Revises:
Create Date: 2026-10-22
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_create_predictions"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the predictions table and its lookup indexes."""
    op.create_table(
        "predictions",
        sa.Column("prediction_id", sa.Uuid(), nullable=False),
        sa.Column("transaction_id", sa.String(length=128), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("model_version", sa.String(length=128), nullable=False),
        sa.Column("model_sha256", sa.String(length=64), nullable=False),
        sa.Column("risk_score", sa.Float(), nullable=False),
        sa.Column("threshold", sa.Float(), nullable=False),
        sa.Column("decision", sa.String(length=16), nullable=False),
        sa.Column(
            "review_status",
            sa.String(length=32),
            server_default=sa.text("'pending'"),
            nullable=False,
        ),
        sa.Column("feature_snapshot", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        # Short names: the target metadata's naming convention prefixes them with
        # ``ck_predictions_``. Supplying the final name here would double that prefix.
        sa.CheckConstraint(
            "risk_score >= 0 AND risk_score <= 1",
            name="risk_score_range",
        ),
        sa.CheckConstraint(
            "threshold >= 0 AND threshold <= 1",
            name="threshold_range",
        ),
        sa.CheckConstraint(
            "decision IN ('legitimate', 'fraud')",
            name="decision_allowed",
        ),
        sa.CheckConstraint(
            "review_status IN ('pending', 'confirmed_fraud', 'confirmed_legitimate')",
            name="review_status_allowed",
        ),
        sa.PrimaryKeyConstraint("prediction_id", name="pk_predictions"),
    )
    op.create_index("ix_predictions_created_at", "predictions", ["created_at"], unique=False)
    op.create_index(
        "ix_predictions_transaction_id",
        "predictions",
        ["transaction_id"],
        unique=False,
    )


def downgrade() -> None:
    """Drop the predictions table and its indexes."""
    op.drop_index("ix_predictions_transaction_id", table_name="predictions")
    op.drop_index("ix_predictions_created_at", table_name="predictions")
    op.drop_table("predictions")
