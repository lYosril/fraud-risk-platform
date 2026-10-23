"""Unit tests for drift statistics, reference profiles and monitoring reports.

All fixtures are synthetic and deterministic. No production transaction row, trained model or
PostgreSQL connection is involved. The tests pin the PSI definition against distributions with
an analytically known value, the KS statistic against a hand-constructed shift, missing-value
accounting, the empty-window state, and the explicit unavailable performance section.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from fraud_risk.monitoring.drift import (
    DriftStatus,
    bin_proportions,
    combine_statuses,
    ks_p_value,
    ks_statistic,
    missing_rate,
    population_stability_index,
    psi_from_proportions,
    quantile_edges,
)
from fraud_risk.monitoring.reference import (
    ReferenceProfile,
    ReferenceProfileError,
    build_distribution_reference,
    build_reference_profile,
    load_reference_profile,
    write_reference_profile,
)
from fraud_risk.monitoring.report import (
    AggregateCounts,
    AnalysisSample,
    ModelVersionUsage,
    MonitoringReport,
    MonitoringReportError,
    MonitoringThresholds,
    build_monitoring_report,
    feature_rows_from_snapshots,
    write_report,
)

FEATURE_COLUMNS = ("V1", "V2", "Amount")
NOW = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)


def make_thresholds() -> MonitoringThresholds:
    return MonitoringThresholds(
        feature_psi_warning=0.10,
        feature_psi_alert=0.25,
        score_psi_warning=0.10,
        score_psi_alert=0.25,
        ks_warning=0.10,
        ks_alert=0.20,
        missing_rate_warning=0.01,
        missing_rate_alert=0.05,
        validation_failure_rate_warning=0.02,
        validation_failure_rate_alert=0.05,
        sample_limit=50_000,
    )


def make_reference() -> tuple[ReferenceProfile, pd.DataFrame, NDArray[np.float64]]:
    """A deterministic synthetic reference with three features and a score sample."""
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(
        {
            "V1": rng.normal(0.0, 1.0, 200),
            "V2": rng.normal(5.0, 2.0, 200),
            "Amount": rng.uniform(0.0, 100.0, 200),
        }
    )
    scores = rng.beta(2.0, 8.0, 200)
    reference = build_reference_profile(
        frame,
        scores,
        feature_columns=FEATURE_COLUMNS,
        source="synthetic test training split",
        model_version="test-model",
        model_sha256="a" * 64,
        threshold=0.5,
        split_strategy="temporal",
        split_seed=42,
        training_rows=400,
        sample_method="all test rows",
        generated_at=NOW,
    )
    return reference, frame, scores


def make_aggregate(
    *,
    prediction_count: int,
    fraud_count: int = 0,
    validation_failure_count: int = 0,
    review_status_counts: dict[str, int] | None = None,
) -> AggregateCounts:
    legitimate = prediction_count - fraud_count
    return AggregateCounts(
        prediction_count=prediction_count,
        fraud_count=fraud_count,
        legitimate_count=legitimate,
        model_versions=(
            ModelVersionUsage(
                model_version="test-model",
                total=prediction_count,
                fraud=fraud_count,
                legitimate=legitimate,
            ),
        )
        if prediction_count
        else (),
        review_status_counts=review_status_counts or {},
        validation_failure_count=validation_failure_count,
    )


def sample_from_frame(
    frame: pd.DataFrame,
    scores: NDArray[np.float64],
) -> AnalysisSample:
    rows = tuple(
        {column: float(frame.iloc[index][column]) for column in FEATURE_COLUMNS}
        for index in range(len(frame))
    )
    return AnalysisSample(
        scores=tuple(float(value) for value in scores),
        feature_rows=rows,
        snapshot_count=len(rows),
    )


def build_report(
    *,
    aggregate: AggregateCounts,
    sample: AnalysisSample | None,
    reference: ReferenceProfile | None = None,
) -> MonitoringReport:
    resolved_reference = reference or make_reference()[0]
    return build_monitoring_report(
        reference=resolved_reference,
        aggregate=aggregate,
        sample=sample,
        thresholds=make_thresholds(),
        window_start=NOW - timedelta(hours=24),
        window_end=NOW,
        generated_at=NOW,
    )


# ---------------------------------------------------------------------------
# Drift statistics
# ---------------------------------------------------------------------------
def test_psi_is_zero_for_identical_proportions() -> None:
    assert psi_from_proportions([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)


def test_psi_matches_a_hand_constructed_value() -> None:
    """Two-bin example: (0.75-0.50)*ln(0.75/0.50) + (0.25-0.50)*ln(0.25/0.50)."""
    expected = (0.75 - 0.50) * math.log(0.75 / 0.50) + (0.25 - 0.50) * math.log(0.25 / 0.50)

    result = psi_from_proportions([0.5, 0.5], [0.75, 0.25])

    assert result == pytest.approx(expected)


def test_psi_uses_reference_quantile_bins_and_detects_a_shift() -> None:
    reference = np.linspace(0.0, 1.0, 500)
    unshifted = np.linspace(0.0, 1.0, 500)
    shifted = np.linspace(2.0, 3.0, 500)

    assert population_stability_index(reference, unshifted) == pytest.approx(0.0, abs=1e-12)
    assert population_stability_index(reference, shifted) > 0.25


def test_quantile_edges_are_strictly_increasing_and_handle_constants() -> None:
    edges = quantile_edges(np.linspace(0.0, 1.0, 100), bin_count=10)

    assert len(edges) == 9
    assert np.all(np.diff(edges) > 0.0)
    constant_edges = quantile_edges(np.ones(20), bin_count=10)
    assert constant_edges.tolist() == [1.0]
    assert quantile_edges(np.ones(20), bin_count=1).size == 0


def test_bin_proportions_cover_every_reference_bin() -> None:
    edges = np.array([0.25, 0.5, 0.75])
    proportions = bin_proportions(np.linspace(0.0, 1.0, 100), edges)

    assert len(proportions) == 4
    assert proportions.sum() == pytest.approx(1.0)


def test_ks_statistic_detects_a_known_half_distribution_shift() -> None:
    reference = np.arange(100, dtype="float64")
    shifted = reference + 50.0

    statistic = ks_statistic(reference, shifted)

    assert statistic == pytest.approx(0.5)
    assert ks_p_value(statistic, 100, 100) < 0.001


def test_missing_rate_counts_none_and_non_finite_values() -> None:
    assert missing_rate([1.0, None, float("nan"), 2.0]) == pytest.approx(0.5)
    assert missing_rate([1.0, 2.0]) == 0.0
    with pytest.raises(ValueError):
        missing_rate([])


def test_status_combination_keeps_the_most_severe_available_state() -> None:
    assert combine_statuses([DriftStatus.UNAVAILABLE, DriftStatus.OK]) is DriftStatus.OK
    assert (
        combine_statuses([DriftStatus.OK, DriftStatus.WARNING, DriftStatus.ALERT])
        is DriftStatus.ALERT
    )
    assert combine_statuses([DriftStatus.UNAVAILABLE]) is DriftStatus.UNAVAILABLE


# ---------------------------------------------------------------------------
# Reference profiles
# ---------------------------------------------------------------------------
def test_reference_profile_round_trips_and_detects_tampering(tmp_path: Path) -> None:
    reference, _frame, _scores = make_reference()
    path = write_reference_profile(reference, tmp_path / "reference.json")

    loaded = load_reference_profile(path)

    assert loaded.reference_id == reference.reference_id
    assert [item.name for item in loaded.features] == list(FEATURE_COLUMNS)
    assert loaded.score.name == "risk_score"
    assert loaded.scored_rows == 200

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["features"][0]["mean"] = payload["features"][0]["mean"] + 1.0
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ReferenceProfileError, match="reference_id"):
        load_reference_profile(tampered)


def test_reference_distribution_rejects_an_all_missing_column() -> None:
    with pytest.raises(ReferenceProfileError, match="no finite values"):
        build_distribution_reference(np.array([np.nan, np.nan]), name="V1")


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
def test_unshifted_window_is_ok_and_performance_stays_unavailable() -> None:
    reference, frame, scores = make_reference()
    report = build_report(
        reference=reference,
        aggregate=make_aggregate(prediction_count=len(frame)),
        sample=sample_from_frame(frame, scores),
    )

    assert report.status is DriftStatus.OK
    assert report.data_drift_status is DriftStatus.OK
    assert report.payload["feature_drift"]["status"] == "ok"
    assert report.payload["score_drift"]["status"] == "ok"
    assert report.payload["performance"]["available"] is False
    assert report.payload["performance"]["status"] == "unavailable"
    assert "Delayed ground-truth labels are unavailable" in report.payload["performance"]["reason"]
    assert report.payload["traffic"]["predictions"] == len(frame)
    assert report.payload["traffic"]["sampled_predictions"] == len(frame)
    assert report.payload["traffic"]["sample_truncated"] is False


def test_shifted_window_produces_a_visible_breach() -> None:
    reference, frame, _scores = make_reference()
    shifted = frame.copy()
    shifted["V1"] = shifted["V1"] + 10.0
    shifted["V2"] = shifted["V2"] + 10.0
    shifted_scores = np.full(len(shifted), 0.95)
    report = build_report(
        reference=reference,
        aggregate=make_aggregate(prediction_count=len(shifted)),
        sample=sample_from_frame(shifted, shifted_scores),
    )

    assert report.status is DriftStatus.ALERT
    assert report.data_drift_status is DriftStatus.ALERT
    assert report.feature_drift_count >= 1
    assert report.payload["score_drift"]["psi"] > 0.25
    assert report.payload["feature_drift"]["breached_feature_count"] >= 1


def test_empty_window_is_a_valid_unavailable_report() -> None:
    report = build_report(
        aggregate=make_aggregate(prediction_count=0),
        sample=None,
    )

    assert report.status is DriftStatus.UNAVAILABLE
    assert report.data_drift_status is DriftStatus.UNAVAILABLE
    assert report.data_quality_status is DriftStatus.UNAVAILABLE
    assert report.payload["feature_drift"]["available"] is False
    assert report.payload["score_drift"]["available"] is False
    assert report.payload["validation_failures"]["available"] is False
    assert "empty" in report.payload["summary"]["reason"]
    assert report.payload["performance"]["available"] is False


def test_feature_snapshots_absent_leaves_feature_and_missing_sections_unavailable() -> None:
    reference, _frame, scores = make_reference()
    sample = AnalysisSample(
        scores=tuple(float(value) for value in scores),
        feature_rows=(),
        snapshot_count=0,
    )
    report = build_report(
        reference=reference,
        aggregate=make_aggregate(prediction_count=200),
        sample=sample,
    )

    assert report.payload["feature_drift"]["available"] is False
    assert report.payload["missing_values"]["available"] is False
    assert report.payload["score_drift"]["available"] is True
    assert any("Feature distribution" in caveat for caveat in report.payload["caveats"])


def test_missing_values_are_reported_when_snapshots_exist() -> None:
    reference, frame, scores = make_reference()
    rows: list[dict[str, float | None]] = [
        {column: float(frame.iloc[index][column]) for column in FEATURE_COLUMNS}
        for index in range(len(frame))
    ]
    rows[0]["V1"] = None
    rows[1]["V1"] = None
    sample = AnalysisSample(
        scores=tuple(float(value) for value in scores),
        feature_rows=tuple(rows),
        snapshot_count=len(rows),
    )
    report = build_report(
        reference=reference,
        aggregate=make_aggregate(prediction_count=200),
        sample=sample,
    )

    missing = report.payload["missing_values"]
    assert missing["available"] is True
    assert missing["overall_missing_rate"] == pytest.approx(2 / (200 * 3))
    assert missing["features"][0]["missing_rate"] == pytest.approx(0.01)
    assert missing["features"][0]["status"] in {"warning", "alert"}


def test_validation_failure_rate_is_classified_against_configured_thresholds() -> None:
    reference, frame, scores = make_reference()
    report = build_report(
        reference=reference,
        aggregate=make_aggregate(
            prediction_count=95,
            validation_failure_count=5,
        ),
        sample=sample_from_frame(frame, scores),
    )

    validation = report.payload["validation_failures"]
    assert validation["count"] == 5
    assert validation["rate"] == pytest.approx(0.05)
    assert validation["status"] == "alert"
    assert report.data_quality_status is DriftStatus.ALERT


def test_reviewed_subset_is_reported_but_never_used_as_overall_performance() -> None:
    reference, frame, scores = make_reference()
    report = build_report(
        reference=reference,
        aggregate=make_aggregate(
            prediction_count=200,
            review_status_counts={
                "pending": 190,
                "confirmed_fraud": 4,
                "confirmed_legitimate": 6,
            },
        ),
        sample=sample_from_frame(frame, scores),
    )

    performance = report.payload["performance"]
    assert performance["available"] is False
    assert performance["reviewed_predictions"] == 10
    assert performance["confirmed_fraud"] == 4
    assert performance["confirmed_legitimate"] == 6
    assert performance["coverage"] == pytest.approx(0.05)
    assert performance["metrics"] is None
    assert "non-random" in performance["reason"]


def test_truncated_sample_is_explicit_and_counts_remain_exact() -> None:
    reference, frame, scores = make_reference()
    sample = sample_from_frame(frame.iloc[:20], scores[:20])
    report = build_report(
        reference=reference,
        aggregate=make_aggregate(prediction_count=1_000, fraud_count=10),
        sample=sample,
    )

    traffic = report.payload["traffic"]
    assert traffic["predictions"] == 1_000
    assert traffic["sampled_predictions"] == 20
    assert traffic["sample_truncated"] is True
    assert traffic["decision_mix"]["fraud"] == 10
    assert any("newest" in caveat for caveat in report.payload["caveats"])


def test_feature_rows_from_snapshots_never_crashes_on_a_malformed_row() -> None:
    rows, count = feature_rows_from_snapshots(
        [
            {"amount": 10.0, "features": {"V1": 0.5, "V2": None}},
            {"unexpected": "shape"},
        ],
        FEATURE_COLUMNS,
    )

    assert count == 2
    assert rows[0]["Amount"] == 10.0
    assert rows[0]["V1"] == 0.5
    assert rows[0]["V2"] is None
    assert rows[1]["V1"] is None


def test_report_writer_emits_timestamped_and_latest_json(tmp_path: Path) -> None:
    reference, frame, scores = make_reference()
    report = build_report(
        reference=reference,
        aggregate=make_aggregate(prediction_count=200),
        sample=sample_from_frame(frame, scores),
    )

    path = write_report(report, tmp_path)

    assert path.is_file()
    assert (tmp_path / "latest.json").is_file()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["report_id"] == str(report.report_id)
    assert payload["schema_version"] == 1
    assert "NaN" not in path.read_text(encoding="utf-8")


def test_window_must_be_timezone_aware_and_ordered() -> None:
    reference, _frame, _scores = make_reference()
    with pytest.raises(MonitoringReportError, match="timezone"):
        build_monitoring_report(
            reference=reference,
            aggregate=make_aggregate(prediction_count=0),
            sample=None,
            thresholds=make_thresholds(),
            window_start=datetime(2026, 10, 22, 12, 0, tzinfo=UTC).replace(tzinfo=None),
            window_end=NOW,
        )
    with pytest.raises(MonitoringReportError, match="after"):
        build_monitoring_report(
            reference=reference,
            aggregate=make_aggregate(prediction_count=0),
            sample=None,
            thresholds=make_thresholds(),
            window_start=NOW,
            window_end=NOW,
        )
