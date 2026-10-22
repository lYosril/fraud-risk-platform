"""Pure transformations shared by the dashboard views and their tests.

Responsibility
    Turn typed API records into the small tables and counts Plotly/Streamlit render. Keeping
    these functions free of Streamlit and Plotly makes them deterministic and testable without
    a browser or a running API.

Rules
    A missing value stays missing. These helpers never fill an empty overview with invented
    traffic, and they never derive a model-performance metric from unlabelled predictions.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from fraud_risk.dashboard.api_client import (
    Explanation,
    OverviewSummary,
    PredictionRecord,
    RiskBin,
)

RECORD_COLUMNS = (
    "prediction_id",
    "transaction_id",
    "created_at",
    "model_version",
    "risk_score",
    "threshold",
    "decision",
    "review_status",
)
RISK_FRAME_COLUMNS = ("label", "lower", "upper", "midpoint", "width", "count")
REVIEW_TABLE_COLUMNS = (
    "prediction_id",
    "transaction_id",
    "created_at",
    "model_version",
    "risk_score",
    "threshold",
    "predicted_label",
    "review_outcome",
)
DECISION_LABELS = {
    "fraud": "Predicted fraud",
    "legitimate": "Predicted legitimate",
}
REVIEW_STATUS_LABELS = {
    "pending": "Pending review",
    "confirmed_fraud": "Confirmed fraud",
    "confirmed_legitimate": "Confirmed legitimate",
}
EXPLANATION_COLUMNS = (
    "feature",
    "rank",
    "raw_value",
    "model_value",
    "contribution",
    "abs_contribution",
    "direction",
)
VOLUME_COLUMNS = ("hour", "predictions")


def fraud_rate(summary: OverviewSummary) -> float | None:
    """Return the flag rate, or ``None`` when no predictions exist."""
    if summary.total_predictions <= 0:
        return None
    return summary.fraud_predictions / summary.total_predictions


def format_percentage(value: float | None, *, digits: int = 2) -> str:
    """Format a fraction as a percentage, preserving an honest unavailable state."""
    if value is None:
        return "n/a"
    return f"{value:.{digits}%}"


def decision_counts(summary: OverviewSummary) -> dict[str, int]:
    """Return the two decision buckets for a chart."""
    return {
        "Flagged fraud": summary.fraud_predictions,
        "Legitimate": summary.legitimate_predictions,
    }


def decision_label(value: str) -> str:
    """Name a model label without implying human confirmation."""
    return DECISION_LABELS.get(value, value)


def review_status_label(value: str) -> str:
    """Name an analyst review state, distinct from the model's predicted label."""
    return REVIEW_STATUS_LABELS.get(value, value)


def review_counts(summary: OverviewSummary) -> dict[str, int]:
    """Return the analyst review buckets for a chart."""
    return {
        "Pending": summary.pending_reviews,
        "Confirmed fraud": summary.confirmed_fraud,
        "Confirmed legitimate": summary.confirmed_legitimate,
    }


def risk_distribution_rows(summary: OverviewSummary) -> list[dict[str, Any]]:
    """Return one chart-ready row per fixed-width risk bucket."""
    rows: list[dict[str, Any]] = []
    for item in summary.risk_bins:
        rows.append(_risk_row(item))
    return rows


def _risk_row(item: RiskBin) -> dict[str, Any]:
    lower_percent = item.lower * 100
    upper_percent = item.upper * 100
    return {
        "label": f"{lower_percent:.0f}-{upper_percent:.0f}%",
        "lower": item.lower,
        "upper": item.upper,
        "midpoint": (item.lower + item.upper) / 2,
        "width": item.upper - item.lower,
        "count": item.count,
    }


def recent_predictions_frame(records: Sequence[PredictionRecord]) -> pd.DataFrame:
    """Return the non-sensitive record fields as a newest-first display table."""
    rows = [
        {
            "prediction_id": record.prediction_id,
            "transaction_id": record.transaction_id,
            "created_at": record.created_at,
            "model_version": record.model_version,
            "risk_score": record.risk_score,
            "threshold": record.threshold,
            "decision": record.decision,
            "review_status": record.review_status,
        }
        for record in records
    ]
    frame = pd.DataFrame(rows, columns=list(RECORD_COLUMNS))
    frame["created_at"] = pd.to_datetime(frame["created_at"], utc=True)
    return frame


def review_table_frame(records: Sequence[PredictionRecord]) -> pd.DataFrame:
    """Return a display table that keeps the model label and analyst outcome separate."""
    frame = recent_predictions_frame(records)
    frame["predicted_label"] = [decision_label(value) for value in frame["decision"]]
    frame["review_outcome"] = [review_status_label(value) for value in frame["review_status"]]
    display = frame.drop(columns=["decision", "review_status"])
    return display[list(REVIEW_TABLE_COLUMNS)]


def volume_frame(records: Sequence[PredictionRecord]) -> pd.DataFrame:
    """Aggregate a prediction page into hourly volume (UTC)."""
    frame = recent_predictions_frame(records)
    if frame.empty:
        return pd.DataFrame(columns=list(VOLUME_COLUMNS))
    aggregated = (
        frame.set_index("created_at")
        .resample("h")
        .size()
        .rename("predictions")
        .reset_index()
        .rename(columns={"created_at": "hour"})
    )
    return aggregated


def explanation_frame(explanation: Explanation) -> pd.DataFrame:
    """Return contributions sorted so the strongest effect is plotted at the top."""
    rows = [
        {
            "feature": item.feature,
            "rank": item.rank,
            "raw_value": item.raw_value,
            "model_value": item.model_value,
            "contribution": item.contribution,
            "abs_contribution": item.abs_contribution,
            "direction": item.direction,
        }
        for item in explanation.contributions
    ]
    frame = pd.DataFrame(rows, columns=list(EXPLANATION_COLUMNS))
    return frame.sort_values("abs_contribution", ascending=True, kind="stable").reset_index(
        drop=True
    )


__all__ = [
    "DECISION_LABELS",
    "EXPLANATION_COLUMNS",
    "RECORD_COLUMNS",
    "REVIEW_STATUS_LABELS",
    "REVIEW_TABLE_COLUMNS",
    "VOLUME_COLUMNS",
    "decision_counts",
    "decision_label",
    "explanation_frame",
    "format_percentage",
    "fraud_rate",
    "recent_predictions_frame",
    "review_counts",
    "review_status_label",
    "review_table_frame",
    "risk_distribution_rows",
    "volume_frame",
]
