"""Tests for the dashboard's pure data transformations."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from fraud_risk.dashboard.api_client import (
    Explanation,
    ExplanationContribution,
    MonitoringFeatureDrift,
    MonitoringSummary,
    MonitoringVersionUsage,
    OverviewSummary,
    PerformanceMetric,
    PerformanceMetrics,
    PerformanceSegment,
    PerformanceSummary,
    PerformanceVersionSummary,
    PredictionRecord,
    RiskBin,
)
from fraud_risk.dashboard.transformations import (
    decision_counts,
    decision_label,
    explanation_frame,
    format_percentage,
    fraud_rate,
    monitoring_feature_frame,
    monitoring_status_label,
    monitoring_version_frame,
    performance_metric_text,
    performance_segment_frame,
    performance_status_label,
    performance_version_frame,
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


def make_monitoring_summary() -> MonitoringSummary:
    return MonitoringSummary(
        available=True,
        reason=None,
        report_id="report-1",
        generated_at=datetime(2026, 10, 23, 12, 0, tzinfo=UTC),
        window_start=datetime(2026, 10, 22, 12, 0, tzinfo=UTC),
        window_end=datetime(2026, 10, 23, 12, 0, tzinfo=UTC),
        status="warning",
        data_drift_status="warning",
        data_quality_status="ok",
        performance_status="unavailable",
        performance_reason="Delayed labels are unavailable.",
        prediction_count=120,
        validation_failure_count=2,
        feature_snapshot_count=100,
        feature_drift_count=1,
        score_status="ok",
        score_psi=0.02,
        score_ks_statistic=0.03,
        missing_overall_rate=0.0,
        validation_failure_rate=0.016,
        features=(
            MonitoringFeatureDrift(
                feature="V1",
                status="warning",
                psi=0.13,
                ks_statistic=0.08,
                missing_rate=0.0,
                sample_count=100,
            ),
        ),
        model_versions=(
            MonitoringVersionUsage(
                model_version="version-1",
                total=120,
                fraud=20,
                legitimate=100,
                share=1.0,
            ),
        ),
        caveats=("Data drift is not performance degradation.",),
        raw_report=None,
    )


def make_performance_metrics() -> PerformanceMetrics:
    return PerformanceMetrics(
        rows=4,
        positives=2,
        negatives=2,
        threshold=0.5,
        true_negatives=1,
        false_positives=1,
        false_negatives=1,
        true_positives=1,
        precision=PerformanceMetric(0.5, True, None),
        recall=PerformanceMetric(0.5, True, None),
        f1=PerformanceMetric(0.5, True, None),
        pr_auc=PerformanceMetric(0.833333, True, None),
        roc_auc=PerformanceMetric(0.75, True, None),
        false_positive_rate=PerformanceMetric(0.5, True, None),
    )


def make_performance_summary() -> PerformanceSummary:
    metrics = make_performance_metrics()
    return PerformanceSummary(
        available=True,
        reason=None,
        report_id="performance-1",
        generated_at=datetime(2026, 10, 23, 12, 0, tzinfo=UTC),
        window_start=datetime(2026, 10, 1, tzinfo=UTC),
        window_end=datetime(2026, 10, 23, 12, 0, tzinfo=UTC),
        threshold=0.5,
        period="month",
        label_source="verified_chargebacks",
        labels_verified=True,
        status="ok",
        row_count=4,
        model_version_count=1,
        segment_count=1,
        overall=metrics,
        model_versions=(PerformanceVersionSummary(model_version="version-1", metrics=metrics),),
        segments=(
            PerformanceSegment(
                model_version="version-1",
                period_label="2026-10",
                period_start=datetime(2026, 10, 1, tzinfo=UTC),
                period_end=datetime(2026, 11, 1, tzinfo=UTC),
                metrics=metrics,
            ),
        ),
        warnings=(),
        caveats=("Labels can arrive late.",),
        raw_report=None,
    )


def test_monitoring_tables_preserve_report_order_and_status_labels() -> None:
    summary = make_monitoring_summary()

    feature_frame = monitoring_feature_frame(summary)
    version_frame = monitoring_version_frame(summary)

    assert list(feature_frame.columns) == [
        "feature",
        "status",
        "psi",
        "ks_statistic",
        "missing_rate",
        "sample_count",
    ]
    assert feature_frame.iloc[0]["status"] == "Warning"
    assert feature_frame.iloc[0]["psi"] == pytest.approx(0.13)
    assert feature_frame.iloc[0]["missing_rate"] == "0.00%"
    assert list(version_frame.columns) == [
        "model_version",
        "total",
        "fraud",
        "legitimate",
        "share",
    ]
    assert version_frame.iloc[0]["share"] == "100.00%"
    assert monitoring_status_label("alert") == "Alert"
    assert monitoring_status_label(None) == "Unavailable"


def test_performance_tables_preserve_undefined_metrics_and_period_labels() -> None:
    summary = make_performance_summary()

    version_frame = performance_version_frame(summary)
    segment_frame = performance_segment_frame(summary)

    assert list(version_frame.columns) == [
        "model_version",
        "rows",
        "positives",
        "precision",
        "recall",
        "f1",
        "pr_auc",
        "roc_auc",
        "false_positive_rate",
    ]
    assert version_frame.iloc[0]["model_version"] == "version-1"
    assert version_frame.iloc[0]["pr_auc"] == "83.33%"
    assert list(segment_frame.columns) == [
        "model_version",
        "period",
        "rows",
        "positives",
        "precision",
        "recall",
        "f1",
        "pr_auc",
        "roc_auc",
        "false_positive_rate",
    ]
    assert segment_frame.iloc[0]["period"] == "2026-10"
    assert summary.overall is not None
    assert performance_metric_text(summary.overall.precision) == "50.00%"
    assert performance_status_label("ok") == "Complete"
    assert performance_status_label("partial") == "Partial / provisional"
    assert performance_status_label(None) == "Unavailable"


def test_performance_tables_render_undefined_metrics_as_undefined() -> None:
    summary = make_performance_summary()
    undefined = PerformanceMetric(
        None,
        False,
        "no rows were predicted positive at this threshold",
    )
    assert summary.overall is not None
    metrics = PerformanceMetrics(
        rows=summary.overall.rows,
        positives=summary.overall.positives,
        negatives=summary.overall.negatives,
        threshold=summary.overall.threshold,
        true_negatives=summary.overall.true_negatives,
        false_positives=summary.overall.false_positives,
        false_negatives=summary.overall.false_negatives,
        true_positives=summary.overall.true_positives,
        precision=undefined,
        recall=summary.overall.recall,
        f1=summary.overall.f1,
        pr_auc=summary.overall.pr_auc,
        roc_auc=summary.overall.roc_auc,
        false_positive_rate=summary.overall.false_positive_rate,
    )
    frame = performance_version_frame(
        PerformanceSummary(
            available=True,
            reason=None,
            report_id=summary.report_id,
            generated_at=summary.generated_at,
            window_start=summary.window_start,
            window_end=summary.window_end,
            threshold=summary.threshold,
            period=summary.period,
            label_source=summary.label_source,
            labels_verified=summary.labels_verified,
            status=summary.status,
            row_count=summary.row_count,
            model_version_count=summary.model_version_count,
            segment_count=summary.segment_count,
            overall=metrics,
            model_versions=(PerformanceVersionSummary("version-1", metrics),),
            segments=summary.segments,
            warnings=summary.warnings,
            caveats=summary.caveats,
            raw_report=None,
        )
    )

    assert frame.iloc[0]["precision"] == "undefined"
    assert performance_metric_text(undefined) == "undefined"


def test_empty_performance_frames_keep_their_display_schema() -> None:
    summary = PerformanceSummary(
        available=False,
        reason="No performance report is available.",
        report_id=None,
        generated_at=None,
        window_start=None,
        window_end=None,
        threshold=None,
        period=None,
        label_source=None,
        labels_verified=False,
        status=None,
        row_count=0,
        model_version_count=0,
        segment_count=0,
        overall=None,
        model_versions=(),
        segments=(),
        warnings=(),
        caveats=(),
        raw_report=None,
    )

    assert list(performance_version_frame(summary).columns) == [
        "model_version",
        "rows",
        "positives",
        "precision",
        "recall",
        "f1",
        "pr_auc",
        "roc_auc",
        "false_positive_rate",
    ]
    assert list(performance_segment_frame(summary).columns) == [
        "model_version",
        "period",
        "rows",
        "positives",
        "precision",
        "recall",
        "f1",
        "pr_auc",
        "roc_auc",
        "false_positive_rate",
    ]
    assert performance_metric_text(None) == "undefined"


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
