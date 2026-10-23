"""Unit tests for the delayed-label performance reporting workflow.

The fixtures are synthetic, deterministic and tiny. They pin the metric formulas, the explicit
undefined states, the exact input schema, version/period grouping, window filtering and the
aggregate-only JSON contract. No model artifact, PostgreSQL connection or real transaction row
is involved.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from fraud_risk.config import PerformancePeriod, Settings
from fraud_risk.monitoring.cli import main
from fraud_risk.monitoring.performance import (
    MODEL_VERSION_COLUMN,
    PERFORMANCE_FEATURE_COLUMNS,
    PERFORMANCE_REQUIRED_COLUMNS,
    SCORE_COLUMN,
    SCORED_AT_COLUMN,
    TARGET_COLUMN,
    LabeledEvaluationDataset,
    PerformanceEvaluationReport,
    PerformanceReportError,
    PerformanceSchemaError,
    build_performance_report,
    load_performance_dataset,
    period_bounds,
    period_label,
    prepare_performance_frame,
    write_performance_report,
)

SHA = "a" * 64


def make_row(
    *,
    label: int,
    score: float,
    version: str = "model-v1",
    scored_at: str = "2026-10-23T12:00:00Z",
) -> dict[str, object]:
    row: dict[str, object] = dict.fromkeys(PERFORMANCE_FEATURE_COLUMNS, 0.0)
    row.update(
        {
            TARGET_COLUMN: label,
            SCORE_COLUMN: score,
            MODEL_VERSION_COLUMN: version,
            SCORED_AT_COLUMN: scored_at,
        }
    )
    return row


def make_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(PERFORMANCE_REQUIRED_COLUMNS))


def make_dataset(
    rows: list[dict[str, object]],
    *,
    sha: str = SHA,
) -> LabeledEvaluationDataset:
    prepared = prepare_performance_frame(make_frame(rows))
    return LabeledEvaluationDataset(
        frame=prepared,
        source_sha256=sha,
        source_rows=len(prepared),
    )


def build(
    dataset: LabeledEvaluationDataset,
    *,
    threshold: float = 0.5,
    period: PerformancePeriod = "month",
    labels_verified: bool = True,
    window_start: datetime | None = None,
    window_end: datetime | None = None,
) -> PerformanceEvaluationReport:
    return build_performance_report(
        dataset=dataset,
        threshold=threshold,
        period=period,
        label_source="verified_chargebacks",
        labels_verified=labels_verified,
        window_start=window_start,
        window_end=window_end,
        generated_at=datetime(2026, 10, 23, 12, 0, tzinfo=UTC),
    )


# ---------------------------------------------------------------------------
# Metric formulas
# ---------------------------------------------------------------------------
def test_metrics_match_hand_computed_confusion_matrix() -> None:
    report = build(
        make_dataset(
            [
                make_row(label=1, score=0.9),
                make_row(label=1, score=0.4),
                make_row(label=0, score=0.8),
                make_row(label=0, score=0.1),
            ]
        )
    )

    overall = report.overall
    assert overall.rows == 4
    assert overall.positives == 2
    assert overall.negatives == 2
    assert overall.true_positives == 1
    assert overall.false_positives == 1
    assert overall.false_negatives == 1
    assert overall.true_negatives == 1
    assert overall.precision.value == pytest.approx(0.5)
    assert overall.recall.value == pytest.approx(0.5)
    assert overall.f1.value == pytest.approx(0.5)
    assert overall.pr_auc.value == pytest.approx(5.0 / 6.0)
    assert overall.roc_auc.value == pytest.approx(0.75)
    assert overall.false_positive_rate.value == pytest.approx(0.5)
    assert overall.undefined_metrics == ()
    assert report.payload["overall"]["confusion_matrix"] == {
        "true_negative": 1,
        "false_positive": 1,
        "false_negative": 1,
        "true_positive": 1,
    }


def test_threshold_changes_decisions_but_not_ranking_metrics() -> None:
    dataset = make_dataset(
        [
            make_row(label=1, score=0.9),
            make_row(label=1, score=0.4),
            make_row(label=0, score=0.8),
            make_row(label=0, score=0.1),
        ]
    )
    low = build(dataset, threshold=0.3)
    high = build(dataset, threshold=0.95)

    assert low.overall.precision.value == pytest.approx(2.0 / 3.0)
    assert high.overall.precision.defined is False
    assert high.overall.precision.reason == "no rows were predicted positive at this threshold"
    assert low.overall.pr_auc.value == high.overall.pr_auc.value
    assert low.overall.roc_auc.value == high.overall.roc_auc.value


def test_missing_class_is_explicit_not_zero() -> None:
    report = build(
        make_dataset(
            [
                make_row(label=0, score=0.1),
                make_row(label=0, score=0.2),
            ]
        )
    )

    overall = report.overall
    assert overall.positives == 0
    assert overall.precision.defined is False
    assert overall.recall.defined is False
    assert overall.f1.defined is False
    assert overall.pr_auc.defined is False
    assert overall.roc_auc.defined is False
    assert overall.false_positive_rate.value == pytest.approx(0.0)
    assert overall.pr_auc.value is None
    assert "both positive and negative" in (overall.pr_auc.reason or "")
    assert report.payload["summary"]["undefined_overall_metrics"] == [
        "precision",
        "recall",
        "f1",
        "pr_auc",
        "roc_auc",
    ]


def test_all_positive_group_has_undefined_false_positive_rate() -> None:
    report = build(make_dataset([make_row(label=1, score=0.9)]))

    assert report.overall.false_positive_rate.defined is False
    assert "no actual negatives" in (report.overall.false_positive_rate.reason or "")
    assert report.overall.precision.value == pytest.approx(1.0)
    assert report.overall.recall.value == pytest.approx(1.0)
    assert report.overall.f1.value == pytest.approx(1.0)


def test_zero_precision_and_recall_give_defined_zero_f1() -> None:
    report = build(
        make_dataset(
            [
                make_row(label=1, score=0.1),
                make_row(label=0, score=0.9),
            ]
        )
    )

    assert report.overall.precision.value == pytest.approx(0.0)
    assert report.overall.recall.value == pytest.approx(0.0)
    assert report.overall.f1.value == pytest.approx(0.0)
    assert report.overall.f1.defined is True


# ---------------------------------------------------------------------------
# Schema contract
# ---------------------------------------------------------------------------
def test_prepare_frame_normalizes_types_and_preserves_schema() -> None:
    prepared = prepare_performance_frame(make_frame([make_row(label=1, score=0.9)]))

    assert tuple(prepared.columns) == PERFORMANCE_REQUIRED_COLUMNS
    assert prepared[TARGET_COLUMN].dtype == "int8"
    assert prepared[SCORE_COLUMN].dtype == "float64"
    assert str(prepared[SCORED_AT_COLUMN].dt.tz) == "UTC"


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda frame: frame.drop(columns=["V1"]), "missing: V1"),
        (lambda frame: frame.assign(transaction_id="t-1"), "unexpected: transaction_id"),
        (
            lambda frame: frame.assign(review_status="confirmed_fraud"),
            "review_status is analyst workflow state",
        ),
        (lambda frame: frame.assign(Class=1), "raw Class column is not accepted"),
        (lambda frame: frame.assign(V1=float("nan")), "V1"),
        (lambda frame: frame.assign(V1="not-a-number"), "V1"),
        (lambda frame: frame.assign(is_fraud=2), "only the labels 0 and 1"),
        (lambda frame: frame.assign(risk_score=1.5), "must be in \\[0, 1\\]"),
        (
            lambda frame: frame.assign(scored_at="2026-10-23T12:00:00"),
            "explicit timezone offset",
        ),
        (lambda frame: frame.assign(model_version="  "), "non-empty"),
    ],
)
def test_schema_mismatches_are_rejected(
    mutate: Callable[[pd.DataFrame], pd.DataFrame],
    match: str,
) -> None:
    frame = make_frame([make_row(label=1, score=0.9)])
    with pytest.raises(PerformanceSchemaError, match=match):
        prepare_performance_frame(mutate(frame))


def test_duplicate_columns_are_rejected() -> None:
    frame = make_frame([make_row(label=1, score=0.9)])
    frame = pd.concat([frame, frame[["V1"]]], axis=1)

    with pytest.raises(PerformanceSchemaError, match="duplicate column"):
        prepare_performance_frame(frame)


def test_empty_dataset_is_rejected() -> None:
    with pytest.raises(PerformanceSchemaError, match="no data rows"):
        prepare_performance_frame(make_frame([]))


def test_load_dataset_fingerprints_the_file(tmp_path: Path) -> None:
    path = tmp_path / "evaluation.csv"
    make_frame([make_row(label=1, score=0.9)]).to_csv(path, index=False)

    dataset = load_performance_dataset(path)

    assert dataset.source_rows == 1
    assert dataset.source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Grouping and provenance
# ---------------------------------------------------------------------------
def test_report_groups_by_model_version_and_period() -> None:
    report = build(
        make_dataset(
            [
                make_row(label=1, score=0.9, version="v1", scored_at="2026-10-01T00:00:00Z"),
                make_row(label=0, score=0.1, version="v1", scored_at="2026-10-20T00:00:00Z"),
                make_row(label=1, score=0.8, version="v1", scored_at="2026-11-01T00:00:00Z"),
                make_row(label=0, score=0.2, version="v2", scored_at="2026-10-15T00:00:00Z"),
            ]
        )
    )

    assert report.model_version_count == 2
    assert report.segment_count == 3
    assert [item.model_version for item in report.model_versions] == ["v1", "v2"]
    assert [item.period_label for item in report.segments] == [
        "2026-10",
        "2026-11",
        "2026-10",
    ]
    assert report.model_versions[0].metrics.rows == 3
    assert report.model_versions[1].metrics.rows == 1
    assert report.window_start == datetime(2026, 10, 1, tzinfo=UTC)
    assert report.window_end == datetime(2026, 11, 1, tzinfo=UTC)


def test_window_filter_is_inclusive_and_records_exclusions() -> None:
    dataset = make_dataset(
        [
            make_row(label=1, score=0.9, scored_at="2026-10-01T00:00:00Z"),
            make_row(label=0, score=0.1, scored_at="2026-10-20T00:00:00Z"),
            make_row(label=1, score=0.8, scored_at="2026-11-01T00:00:00Z"),
        ]
    )
    report = build(
        dataset,
        window_start=datetime(2026, 10, 1, tzinfo=UTC),
        window_end=datetime(2026, 10, 31, 23, 59, 59, tzinfo=UTC),
    )

    assert report.row_count == 2
    assert report.payload["summary"]["rows_excluded_by_window"] == 1
    assert report.payload["window"]["requested_start"] == "2026-10-01T00:00:00+00:00"
    assert report.payload["window"]["requested_end"] == "2026-10-31T23:59:59+00:00"


def test_empty_window_is_rejected() -> None:
    dataset = make_dataset([make_row(label=1, score=0.9)])

    with pytest.raises(PerformanceReportError, match="contains no labeled rows"):
        build(
            dataset,
            window_start=datetime(2026, 12, 1, tzinfo=UTC),
            window_end=datetime(2026, 12, 2, tzinfo=UTC),
        )


def test_unverified_labels_are_explicit() -> None:
    report = build(make_dataset([make_row(label=1, score=0.9)]), labels_verified=False)

    assert report.labels_verified is False
    assert report.payload["status"] == "partial"
    assert report.payload["evaluation"]["label_provenance"]["verified"] is False
    assert report.payload["evaluation"]["label_provenance"]["analyst_review_status_used"] is False
    assert any(
        "not marked verified" in warning for warning in report.payload["summary"]["warnings"]
    )


def test_invalid_threshold_and_label_source_are_rejected() -> None:
    dataset = make_dataset([make_row(label=1, score=0.9)])

    with pytest.raises(PerformanceReportError, match="threshold"):
        build(dataset, threshold=1.5)
    with pytest.raises(PerformanceReportError, match="label_source"):
        build_performance_report(
            dataset=dataset,
            threshold=0.5,
            period="month",
            label_source="  ",
            labels_verified=True,
        )


def test_period_bounds_and_labels_are_calendar_based() -> None:
    timestamp = datetime(2026, 10, 23, 12, 30, tzinfo=UTC)

    assert period_bounds(timestamp, "day") == (
        datetime(2026, 10, 23, tzinfo=UTC),
        datetime(2026, 10, 24, tzinfo=UTC),
    )
    assert period_bounds(timestamp, "week") == (
        datetime(2026, 10, 19, tzinfo=UTC),
        datetime(2026, 10, 26, tzinfo=UTC),
    )
    assert period_bounds(timestamp, "month") == (
        datetime(2026, 10, 1, tzinfo=UTC),
        datetime(2026, 11, 1, tzinfo=UTC),
    )
    assert period_label(datetime(2026, 10, 23, tzinfo=UTC), "day") == "2026-10-23"
    assert period_label(datetime(2026, 10, 23, tzinfo=UTC), "week") == "2026-W43"
    assert period_label(datetime(2026, 10, 23, tzinfo=UTC), "month") == "2026-10"


# ---------------------------------------------------------------------------
# JSON and CLI
# ---------------------------------------------------------------------------
def test_report_json_is_aggregate_only_and_written_to_latest(tmp_path: Path) -> None:
    report = build(make_dataset([make_row(label=1, score=0.9), make_row(label=0, score=0.1)]))

    path = write_performance_report(report, tmp_path)
    latest = tmp_path / "latest.json"
    text = path.read_text(encoding="utf-8")
    payload = json.loads(text)

    assert path.name.startswith("performance_")
    assert latest.read_text(encoding="utf-8") == text
    assert payload["schema_version"] == 1
    assert "overall" in payload
    assert "V1" not in payload["overall"]
    assert "is_fraud" not in payload["overall"]
    assert payload["dataset"]["target_column"] == TARGET_COLUMN


def test_cli_writes_a_report_without_persisting(tmp_path: Path) -> None:
    csv_path = tmp_path / "evaluation.csv"
    make_frame(
        [
            make_row(label=1, score=0.9),
            make_row(label=0, score=0.1),
        ]
    ).to_csv(csv_path, index=False)
    settings = Settings(app_env="test", performance_report_dir=tmp_path)

    exit_code = main(
        [
            "performance",
            "--path",
            str(csv_path),
            "--threshold",
            "0.5",
            "--label-source",
            "verified_chargebacks",
            "--labels-verified",
            "--no-persist",
            "--report-dir",
            str(tmp_path),
        ],
        settings=settings,
    )

    assert exit_code == 0
    payload = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    assert payload["summary"]["row_count"] == 2
    assert payload["evaluation"]["labels_verified"] is True
    assert payload["overall"]["pr_auc"]["defined"] is True


def test_cli_rejects_a_schema_mismatch(tmp_path: Path) -> None:
    csv_path = tmp_path / "evaluation.csv"
    frame = make_frame([make_row(label=1, score=0.9)]).drop(columns=["V1"])
    frame.to_csv(csv_path, index=False)
    settings = Settings(app_env="test", performance_report_dir=tmp_path)

    exit_code = main(
        [
            "performance",
            "--path",
            str(csv_path),
            "--threshold",
            "0.5",
            "--label-source",
            "verified_chargebacks",
            "--labels-verified",
            "--no-persist",
        ],
        settings=settings,
    )

    assert exit_code == 1
    assert not (tmp_path / "latest.json").exists()
