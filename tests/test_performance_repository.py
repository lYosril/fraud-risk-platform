"""Unit tests for delayed-label performance report persistence.

These tests use a mocked session to pin the commit/rollback policy and the aggregate-only row
mapping. The real PostgreSQL round-trip lives in ``tests/test_db_integration.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from fraud_risk.db.models import PerformanceReport
from fraud_risk.db.repository import PersistenceError
from fraud_risk.monitoring.performance import (
    MetricValue,
    PerformanceEvaluationReport,
    PerformanceMetrics,
)
from fraud_risk.monitoring.performance_repository import PerformanceRepository

NOW = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)


def make_metrics() -> PerformanceMetrics:
    metric = MetricValue(0.5, True, None)
    return PerformanceMetrics(
        rows=4,
        positives=2,
        negatives=2,
        threshold=0.5,
        true_negatives=1,
        false_positives=1,
        false_negatives=1,
        true_positives=1,
        precision=metric,
        recall=metric,
        f1=metric,
        pr_auc=metric,
        roc_auc=metric,
        false_positive_rate=metric,
    )


def make_report() -> PerformanceEvaluationReport:
    metrics = make_metrics()
    return PerformanceEvaluationReport(
        report_id=uuid4(),
        generated_at=NOW,
        window_start=datetime(2026, 10, 1, tzinfo=UTC),
        window_end=NOW,
        row_count=4,
        model_version_count=1,
        segment_count=1,
        threshold=0.5,
        period="month",
        labels_verified=True,
        label_source="verified_chargebacks",
        overall=metrics,
        model_versions=(),
        segments=(),
        payload={"schema_version": 1, "summary": {"row_count": 4}},
    )


def test_save_report_maps_the_typed_report_to_a_row() -> None:
    session = MagicMock(spec=Session)
    report = make_report()

    record = PerformanceRepository(session).save_report(report)

    assert isinstance(record, PerformanceReport)
    assert record.report_id == report.report_id
    assert record.row_count == 4
    assert record.model_version_count == 1
    assert record.segment_count == 1
    assert record.period == "month"
    assert record.labels_verified is True
    assert record.label_source == "verified_chargebacks"
    assert record.payload == report.payload
    session.add.assert_called_once_with(record)
    session.commit.assert_called_once()


def test_save_report_rolls_back_and_raises_a_persistence_error() -> None:
    session = MagicMock(spec=Session)
    session.commit.side_effect = SQLAlchemyError("write failed")

    with pytest.raises(PersistenceError, match="performance report"):
        PerformanceRepository(session).save_report(make_report())

    session.rollback.assert_called_once()


def test_latest_report_returns_the_first_row_or_none() -> None:
    session = MagicMock(spec=Session)
    record = MagicMock()
    session.scalars.return_value.first.return_value = record

    assert PerformanceRepository(session).latest_report() is record

    session.scalars.return_value.first.return_value = None
    assert PerformanceRepository(session).latest_report() is None


def test_latest_report_rolls_back_and_raises_a_persistence_error() -> None:
    session = MagicMock(spec=Session)
    session.scalars.side_effect = SQLAlchemyError("read failed")

    with pytest.raises(PersistenceError, match="performance report"):
        PerformanceRepository(session).latest_report()

    session.rollback.assert_called_once()
