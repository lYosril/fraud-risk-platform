"""Unit tests for the monitoring persistence and failure-handling boundaries.

The session is a mock: these tests pin validation, commit/rollback policy, safe failure
swallowing and the aggregate shape without requiring PostgreSQL. Real round-trips remain in
``tests/test_db_integration.py`` and are marked ``integration``.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import cast
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from fraud_risk.db.models import DriftReport, MonitoringEvent
from fraud_risk.db.repository import PersistenceError
from fraud_risk.db.session import SessionFactory
from fraud_risk.monitoring.drift import DriftStatus
from fraud_risk.monitoring.events import MonitoringEventRecorder
from fraud_risk.monitoring.report import MonitoringReport
from fraud_risk.monitoring.repository import MonitoringRepository

NOW = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)
START = NOW - timedelta(hours=1)


def make_session() -> MagicMock:
    return MagicMock(spec=Session)


def make_report() -> MonitoringReport:
    return MonitoringReport(
        report_id=uuid4(),
        generated_at=NOW,
        window_start=START,
        window_end=NOW,
        reference_id="a" * 64,
        status=DriftStatus.OK,
        data_drift_status=DriftStatus.OK,
        data_quality_status=DriftStatus.OK,
        prediction_count=7,
        validation_failure_count=2,
        feature_snapshot_count=0,
        feature_drift_count=0,
        payload={"schema_version": 1, "summary": {"status": "ok"}},
    )


# ---------------------------------------------------------------------------
# Validation failures
# ---------------------------------------------------------------------------
def test_validation_failure_write_trims_and_bounds_text() -> None:
    session = make_session()
    repository = MonitoringRepository(session)

    repository.record_validation_failures(
        endpoint="  /api/v1/predict  ",
        reason_code="  request_validation  ",
        count=3,
    )

    record = session.add.call_args.args[0]
    assert isinstance(record, MonitoringEvent)
    assert record.endpoint == "/api/v1/predict"
    assert record.reason_code == "request_validation"
    assert record.failure_count == 3
    assert record.event_type == "input_validation_failure"
    session.commit.assert_called_once()


def test_validation_failure_write_rejects_invalid_arguments_without_touching_the_session() -> None:
    session = make_session()
    repository = MonitoringRepository(session)

    with pytest.raises(ValueError, match="at least 1"):
        repository.record_validation_failures(endpoint="/predict", reason_code="bad", count=0)
    with pytest.raises(ValueError, match="endpoint"):
        repository.record_validation_failures(endpoint="   ", reason_code="bad")
    with pytest.raises(ValueError, match="reason_code"):
        repository.record_validation_failures(endpoint="/predict", reason_code="   ")

    session.add.assert_not_called()
    session.commit.assert_not_called()


def test_validation_failure_write_rolls_back_and_raises_a_persistence_error() -> None:
    session = make_session()
    session.commit.side_effect = SQLAlchemyError("database went away")
    repository = MonitoringRepository(session)

    with pytest.raises(PersistenceError, match="monitoring event"):
        repository.record_validation_failures(endpoint="/predict", reason_code="bad")

    session.rollback.assert_called_once()


def test_event_recorder_skips_disabled_and_non_positive_counts() -> None:
    factory = MagicMock()
    recorder = MonitoringEventRecorder(cast(SessionFactory, factory), enabled=False)

    recorder.record_validation_failures(endpoint="/predict", reason_code="bad")
    MonitoringEventRecorder(cast(SessionFactory, factory), enabled=True).record_validation_failures(
        endpoint="/predict",
        reason_code="bad",
        count=0,
    )

    factory.assert_not_called()


def test_event_recorder_uses_safe_defaults_and_swallows_database_failures() -> None:
    session = make_session()
    session.__enter__.return_value = session
    session.__exit__.return_value = None
    session.commit.side_effect = SQLAlchemyError("database went away")
    factory = MagicMock(return_value=session)
    recorder = MonitoringEventRecorder(cast(SessionFactory, factory), enabled=True)

    recorder.record_validation_failures(endpoint="   ", reason_code="   ")

    record = session.add.call_args.args[0]
    assert record.endpoint == "unknown"
    assert record.reason_code == "validation_error"
    session.rollback.assert_called_once()


# ---------------------------------------------------------------------------
# Window aggregates and sample
# ---------------------------------------------------------------------------
def test_window_aggregate_returns_sorted_versions_and_exact_counts() -> None:
    session = make_session()
    usage = MagicMock()
    usage.all.return_value = [
        ("model-b", "fraud", 2),
        ("model-b", "legitimate", 3),
        ("model-a", "legitimate", 4),
    ]
    reviews = MagicMock()
    reviews.all.return_value = [("pending", 5), ("confirmed_fraud", 2)]
    session.execute.side_effect = [usage, reviews]
    session.scalar.return_value = 9

    aggregate = MonitoringRepository(session).window_aggregate(start=START, end=NOW)

    assert aggregate.prediction_count == 9
    assert aggregate.fraud_count == 2
    assert aggregate.legitimate_count == 7
    assert [item.model_version for item in aggregate.model_versions] == ["model-a", "model-b"]
    assert aggregate.model_versions[0].total == 4
    assert aggregate.model_versions[1].fraud == 2
    assert aggregate.review_status_counts == {"pending": 5, "confirmed_fraud": 2}
    assert aggregate.validation_failure_count == 9


def test_window_aggregate_rejects_invalid_windows_before_querying() -> None:
    session = make_session()
    repository = MonitoringRepository(session)

    with pytest.raises(ValueError, match="timezone"):
        repository.window_aggregate(start=START.replace(tzinfo=None), end=NOW)
    with pytest.raises(ValueError, match="before"):
        repository.window_aggregate(start=NOW, end=START)

    session.execute.assert_not_called()


def test_window_aggregate_rolls_back_and_raises_a_persistence_error() -> None:
    session = make_session()
    session.execute.side_effect = SQLAlchemyError("read failed")

    with pytest.raises(PersistenceError, match="aggregates"):
        MonitoringRepository(session).window_aggregate(start=START, end=NOW)

    session.rollback.assert_called_once()


def test_analysis_sample_converts_scores_and_skips_absent_snapshots() -> None:
    session = make_session()
    result = MagicMock()
    result.all.return_value = [
        (0.2, {"amount": 10.0, "features": {"V1": 0.5}}),
        (0.8, None),
        (0.9, ["not", "a", "mapping"]),
    ]
    session.execute.return_value = result

    sample = MonitoringRepository(session).analysis_sample(
        start=START,
        end=NOW,
        limit=10,
        feature_columns=("V1", "Amount"),
    )

    assert sample.scores == (0.2, 0.8, 0.9)
    assert sample.snapshot_count == 1
    assert sample.feature_rows[0]["V1"] == 0.5
    assert sample.feature_rows[0]["Amount"] == 10.0
    assert len(sample.feature_rows) == 1


def test_analysis_sample_rejects_invalid_limit_and_window() -> None:
    session = make_session()
    repository = MonitoringRepository(session)

    with pytest.raises(ValueError, match="limit"):
        repository.analysis_sample(start=START, end=NOW, limit=0, feature_columns=("V1",))
    with pytest.raises(ValueError, match="timezone"):
        repository.analysis_sample(
            start=START.replace(tzinfo=None),
            end=NOW,
            limit=1,
            feature_columns=("V1",),
        )

    session.execute.assert_not_called()


def test_analysis_sample_rolls_back_and_raises_a_persistence_error() -> None:
    session = make_session()
    session.execute.side_effect = SQLAlchemyError("read failed")

    with pytest.raises(PersistenceError, match="sample"):
        MonitoringRepository(session).analysis_sample(
            start=START,
            end=NOW,
            limit=10,
            feature_columns=("V1",),
        )

    session.rollback.assert_called_once()


# ---------------------------------------------------------------------------
# Drift-report persistence
# ---------------------------------------------------------------------------
def test_save_report_maps_the_typed_report_to_a_row() -> None:
    session = make_session()
    report = make_report()

    record = MonitoringRepository(session).save_report(report)

    assert isinstance(record, DriftReport)
    assert record.report_id == report.report_id
    assert record.status == "ok"
    assert record.data_drift_status == "ok"
    assert record.data_quality_status == "ok"
    assert record.prediction_count == 7
    assert record.payload == report.payload
    session.add.assert_called_once_with(record)
    session.commit.assert_called_once()


def test_save_report_rolls_back_and_raises_a_persistence_error() -> None:
    session = make_session()
    session.commit.side_effect = SQLAlchemyError("write failed")

    with pytest.raises(PersistenceError, match="drift report"):
        MonitoringRepository(session).save_report(make_report())

    session.rollback.assert_called_once()


def test_latest_report_returns_the_first_row_or_none() -> None:
    session = make_session()
    record = DriftReport(
        report_id=uuid4(),
        generated_at=NOW,
        window_start=START,
        window_end=NOW,
        reference_id="a" * 64,
        status="ok",
        data_drift_status="ok",
        data_quality_status="ok",
        prediction_count=0,
        validation_failure_count=0,
        feature_snapshot_count=0,
        feature_drift_count=0,
        payload={},
    )
    session.scalars.return_value.first.return_value = record

    assert MonitoringRepository(session).latest_report() is record

    session.scalars.return_value.first.return_value = None
    assert MonitoringRepository(session).latest_report() is None


def test_latest_report_rolls_back_and_raises_a_persistence_error() -> None:
    session = make_session()
    session.scalars.side_effect = SQLAlchemyError("read failed")

    with pytest.raises(PersistenceError, match="drift report"):
        MonitoringRepository(session).latest_report()

    session.rollback.assert_called_once()


def test_monitoring_repository_does_not_leak_feature_values_into_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    session = make_session()
    session.commit.side_effect = SQLAlchemyError("database went away")
    repository = MonitoringRepository(session)

    with (
        caplog.at_level(logging.WARNING, logger="fraud_risk.monitoring.repository"),
        pytest.raises(PersistenceError),
    ):
        repository.record_validation_failures(
            endpoint="/predict",
            reason_code="bad",
            count=1,
        )

    assert "monitoring event write failed" in caplog.text
    assert "database went away" not in caplog.text
