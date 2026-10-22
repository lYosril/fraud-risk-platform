"""Tests for the dashboard's pure data transformations."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from fraud_risk.dashboard.api_client import (
    Explanation,
    ExplanationContribution,
    OverviewSummary,
    PredictionRecord,
    RiskBin,
)
from fraud_risk.dashboard.transformations import (
    decision_counts,
    decision_label,
    explanation_frame,
    format_percentage,
    fraud_rate,
    recent_predictions_frame,
    review_counts,
    review_status_label,
    review_table_frame,
    risk_distribution_rows,
    volume_frame,
)


def make_summary() -> OverviewSummary:
    return OverviewSummary(
        total_predictions=4,
        fraud_predictions=1,
        legitimate_predictions=3,
        pending_reviews=2,
        confirmed_fraud=1,
        confirmed_legitimate=1,
        risk_bins=(
            RiskBin(lower=0.0, upper=0.1, count=3),
            RiskBin(lower=0.1, upper=0.2, count=0),
            RiskBin(lower=0.2, upper=0.3, count=0),
            RiskBin(lower=0.3, upper=0.4, count=0),
            RiskBin(lower=0.4, upper=0.5, count=0),
            RiskBin(lower=0.5, upper=0.6, count=0),
            RiskBin(lower=0.6, upper=0.7, count=0),
            RiskBin(lower=0.7, upper=0.8, count=0),
            RiskBin(lower=0.8, upper=0.9, count=0),
            RiskBin(lower=0.9, upper=1.0, count=1),
        ),
    )


def make_record(
    transaction_id: str,
    *,
    created_at: datetime,
    risk_score: float = 0.75,
    decision: str | None = None,
    review_status: str = "pending",
) -> PredictionRecord:
    return PredictionRecord(
        prediction_id=f"prediction-{transaction_id}",
        transaction_id=transaction_id,
        created_at=created_at,
        model_version="version-1",
        model_sha256="a" * 64,
        risk_score=risk_score,
        threshold=0.5,
        decision=decision or ("fraud" if risk_score >= 0.5 else "legitimate"),
        review_status=review_status,
    )


def test_fraud_rate_and_percentage_preserve_the_empty_state() -> None:
    summary = make_summary()

    assert fraud_rate(summary) == 0.25
    assert format_percentage(fraud_rate(summary)) == "25.00%"
    assert fraud_rate(OverviewSummary(0, 0, 0, 0, 0, 0, ())) is None
    assert format_percentage(None) == "n/a"


def test_decision_and_review_counts() -> None:
    summary = make_summary()

    assert decision_counts(summary) == {"Flagged fraud": 1, "Legitimate": 3}
    assert review_counts(summary) == {
        "Pending": 2,
        "Confirmed fraud": 1,
        "Confirmed legitimate": 1,
    }


def test_risk_rows_are_chart_ready_and_keep_the_bin_width() -> None:
    rows = risk_distribution_rows(make_summary())

    assert rows[0] == {
        "label": "0-10%",
        "lower": 0.0,
        "upper": 0.1,
        "midpoint": 0.05,
        "width": 0.1,
        "count": 3,
    }
    assert rows[-1]["label"] == "90-100%"
    assert rows[-1]["count"] == 1
    assert sum(int(row["count"]) for row in rows) == 4


def test_recent_predictions_frame_has_no_feature_columns_and_utc_timestamps() -> None:
    frame = recent_predictions_frame(
        [
            make_record("txn-1", created_at=datetime(2026, 1, 1, 12, tzinfo=UTC)),
            make_record(
                "txn-2",
                created_at=datetime(2026, 1, 1, 13, tzinfo=UTC),
                risk_score=0.05,
            ),
        ]
    )

    assert list(frame.columns) == [
        "prediction_id",
        "transaction_id",
        "created_at",
        "model_version",
        "risk_score",
        "threshold",
        "decision",
        "review_status",
    ]
    assert frame["created_at"].dt.tz is not None
    assert "features" not in frame.columns
    assert frame.loc[1, "decision"] == "legitimate"


def test_empty_history_frames_keep_their_schema() -> None:
    frame = recent_predictions_frame([])
    volume = volume_frame([])

    assert frame.empty
    assert list(frame.columns) == [
        "prediction_id",
        "transaction_id",
        "created_at",
        "model_version",
        "risk_score",
        "threshold",
        "decision",
        "review_status",
    ]
    assert volume.empty
    assert list(volume.columns) == ["hour", "predictions"]


def test_review_table_separates_the_model_label_from_the_analyst_outcome() -> None:
    frame = review_table_frame(
        [
            make_record(
                "txn-1",
                created_at=datetime(2026, 1, 1, 12, tzinfo=UTC),
                decision="fraud",
                review_status="confirmed_legitimate",
            )
        ]
    )

    assert list(frame.columns) == [
        "prediction_id",
        "transaction_id",
        "created_at",
        "model_version",
        "risk_score",
        "threshold",
        "predicted_label",
        "review_outcome",
    ]
    assert frame.loc[0, "predicted_label"] == "Predicted fraud"
    assert frame.loc[0, "review_outcome"] == "Confirmed legitimate"
    assert "decision" not in frame.columns
    assert "review_status" not in frame.columns
    assert decision_label("legitimate") == "Predicted legitimate"
    assert review_status_label("pending") == "Pending review"
    assert review_status_label("unexpected") == "unexpected"


def test_empty_review_table_keeps_the_display_schema() -> None:
    frame = review_table_frame([])

    assert frame.empty
    assert list(frame.columns) == [
        "prediction_id",
        "transaction_id",
        "created_at",
        "model_version",
        "risk_score",
        "threshold",
        "predicted_label",
        "review_outcome",
    ]


def test_volume_frame_groups_records_by_hour() -> None:
    records = [
        make_record("txn-1", created_at=datetime(2026, 1, 1, 12, 5, tzinfo=UTC)),
        make_record("txn-2", created_at=datetime(2026, 1, 1, 12, 55, tzinfo=UTC)),
        make_record("txn-3", created_at=datetime(2026, 1, 1, 14, 0, tzinfo=UTC)),
    ]

    frame = volume_frame(records)

    assert list(frame.columns) == ["hour", "predictions"]
    assert frame["predictions"].tolist() == [2, 0, 1]
    assert isinstance(frame.loc[0, "hour"], pd.Timestamp)


def test_explanation_frame_sorts_by_absolute_contribution() -> None:
    explanation = Explanation(
        output_space="tree_margin",
        explainer="shap.TreeExplainer",
        feature_perturbation="tree_path_dependent",
        aggregation="mean_absolute",
        n_folds=3,
        top_k=3,
        base_value=0.1,
        output_value=0.5,
        other_features_contribution=0.0,
        additivity_residual=0.0,
        contributions=(
            ExplanationContribution("V1", 1, 1.0, 1.0, 0.4, 0.4, "increases risk"),
            ExplanationContribution("V2", 2, -1.0, -1.0, -0.8, 0.8, "decreases risk"),
            ExplanationContribution("V3", 3, 0.5, 0.5, 0.1, 0.1, "increases risk"),
        ),
        limitations=("Not causal.",),
    )

    frame = explanation_frame(explanation)

    assert frame["feature"].tolist() == ["V3", "V1", "V2"]
    assert frame["abs_contribution"].is_monotonic_increasing
    assert "features" not in frame.columns
