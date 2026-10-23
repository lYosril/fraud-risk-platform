"""Unit tests for the database contract and repository transaction policy.

These tests never connect to PostgreSQL. They pin the model shape (including the privacy
defaults), repository commit/rollback behaviour, the request-scoped session lifecycle and the
offline Alembic rendering. Real round-trip and schema/migration checks live in
``tests/test_db_integration.py`` and require a PostgreSQL service.
"""

from __future__ import annotations

import io
import logging
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import CheckConstraint, Table
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from starlette.requests import Request

from fraud_risk.api.dependencies import get_db_session
from fraud_risk.config import get_settings
from fraud_risk.db.models import (
    DriftReport,
    MonitoringEvent,
    PerformanceReport,
    Prediction,
    PredictionDecision,
    ReviewStatus,
)
from fraud_risk.db.repository import (
    InvalidReviewTransitionError,
    PersistenceError,
    PredictionCreate,
    PredictionRepository,
)
from fraud_risk.db.session import DB_SESSION_FACTORY_STATE_KEY

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_COLUMNS = {
    "prediction_id",
    "transaction_id",
    "created_at",
    "model_version",
    "model_sha256",
    "risk_score",
    "threshold",
    "decision",
    "review_status",
    "feature_snapshot",
}


def make_create(
    transaction_id: str = "txn-db-unit",
    *,
    feature_snapshot: dict[str, Any] | None = None,
) -> PredictionCreate:
    return PredictionCreate(
        transaction_id=transaction_id,
        model_version="model-v1",
        model_sha256="a" * 64,
        risk_score=0.75,
        threshold=0.5,
        decision=PredictionDecision.FRAUD,
        feature_snapshot=feature_snapshot,
    )


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
def test_prediction_table_contract_and_privacy_defaults() -> None:
    table = Prediction.__table__
    columns = table.columns

    assert set(columns.keys()) == EXPECTED_COLUMNS
    assert columns["prediction_id"].primary_key is True
    assert columns["transaction_id"].nullable is False
    assert columns["created_at"].type.timezone is True  # type: ignore[attr-defined]
    assert columns["review_status"].default.arg == ReviewStatus.PENDING.value
    assert columns["feature_snapshot"].nullable is True
    assert "features" not in columns
    assert "amount" not in columns
    assert "is_fraud" not in columns


def test_prediction_table_has_named_check_constraints() -> None:
    table = cast(Table, Prediction.__table__)
    check_constraints = {
        constraint.name
        for constraint in table.constraints
        if isinstance(constraint, CheckConstraint)
    }

    assert {
        "ck_predictions_risk_score_range",
        "ck_predictions_threshold_range",
        "ck_predictions_decision_allowed",
        "ck_predictions_review_status_allowed",
    } <= check_constraints


def test_monitoring_tables_store_aggregates_not_raw_payloads() -> None:
    event_columns = set(MonitoringEvent.__table__.columns.keys())
    report_columns = set(DriftReport.__table__.columns.keys())

    assert event_columns == {
        "event_id",
        "created_at",
        "event_type",
        "endpoint",
        "reason_code",
        "failure_count",
    }
    assert report_columns == {
        "report_id",
        "generated_at",
        "window_start",
        "window_end",
        "reference_id",
        "status",
        "data_drift_status",
        "data_quality_status",
        "prediction_count",
        "validation_failure_count",
        "feature_snapshot_count",
        "feature_drift_count",
        "payload",
    }
    assert "transaction_id" not in event_columns
    assert "transaction_id" not in report_columns
    assert "features" not in event_columns
    assert "features" not in report_columns


def test_monitoring_tables_have_named_check_constraints() -> None:
    event_constraints = {
        constraint.name
        for constraint in cast(Table, MonitoringEvent.__table__).constraints
        if isinstance(constraint, CheckConstraint)
    }
    report_constraints = {
        constraint.name
        for constraint in cast(Table, DriftReport.__table__).constraints
        if isinstance(constraint, CheckConstraint)
    }

    assert {
        "ck_monitoring_events_event_type_allowed",
        "ck_monitoring_events_failure_count_positive",
    } <= event_constraints
    assert {
        "ck_drift_reports_status_allowed",
        "ck_drift_reports_data_drift_status_allowed",
        "ck_drift_reports_data_quality_status_allowed",
        "ck_drift_reports_window_order",
    } <= report_constraints


def test_performance_reports_store_aggregates_not_raw_labels_or_rows() -> None:
    table = cast(Table, PerformanceReport.__table__)
    columns = set(table.columns.keys())

    assert columns == {
        "report_id",
        "generated_at",
        "window_start",
        "window_end",
        "row_count",
        "model_version_count",
        "segment_count",
        "threshold",
        "period",
        "labels_verified",
        "label_source",
        "payload",
    }
    assert "transaction_id" not in columns
    assert "features" not in columns
    assert "risk_score" not in columns
    assert "is_fraud" not in columns
    assert "review_status" not in columns


def test_performance_reports_have_named_check_constraints() -> None:
    table = cast(Table, PerformanceReport.__table__)
    check_constraints = {
        constraint.name
        for constraint in table.constraints
        if isinstance(constraint, CheckConstraint)
    }

    assert {
        "ck_performance_reports_row_count_positive",
        "ck_performance_reports_model_version_count_positive",
        "ck_performance_reports_segment_count_positive",
        "ck_performance_reports_threshold_range",
        "ck_performance_reports_period_allowed",
        "ck_performance_reports_window_order",
    } <= check_constraints


def test_enums_match_the_persisted_vocabulary() -> None:
    assert {member.value for member in PredictionDecision} == {"legitimate", "fraud"}
    assert {member.value for member in ReviewStatus} == {
        "pending",
        "confirmed_fraud",
        "confirmed_legitimate",
    }


# ---------------------------------------------------------------------------
# Repository transaction behaviour
# ---------------------------------------------------------------------------
def test_repository_creates_rows_and_commits_once() -> None:
    session = MagicMock(spec=Session)
    repository = PredictionRepository(session)

    records = repository.create_many([make_create()])

    session.add_all.assert_called_once()
    session.commit.assert_called_once()
    session.rollback.assert_not_called()
    assert len(records) == 1
    assert isinstance(records[0].prediction_id, type(uuid4()))
    assert records[0].created_at.tzinfo is not None
    assert records[0].decision == "fraud"
    assert records[0].review_status == "pending"
    assert records[0].feature_snapshot is None


def test_repository_rolls_back_when_commit_fails() -> None:
    session = MagicMock(spec=Session)
    session.commit.side_effect = SQLAlchemyError("database went away")
    repository = PredictionRepository(session)

    with pytest.raises(PersistenceError, match="could not persist"):
        repository.create_many([make_create()])

    session.rollback.assert_called_once()


def test_repository_does_not_touch_the_session_for_an_empty_batch() -> None:
    session = MagicMock(spec=Session)
    repository = PredictionRepository(session)

    assert repository.create_many([]) == []
    session.add_all.assert_not_called()
    session.commit.assert_not_called()
    session.rollback.assert_not_called()


def test_repository_get_and_review_status_update() -> None:
    session = MagicMock(spec=Session)
    record = Prediction(
        prediction_id=uuid4(),
        transaction_id="txn-review",
        created_at=datetime.now(UTC),
        model_version="model-v1",
        model_sha256="a" * 64,
        risk_score=0.75,
        threshold=0.5,
        decision="fraud",
        review_status="pending",
    )
    session.get.return_value = record
    repository = PredictionRepository(session)

    assert repository.get(record.prediction_id) is record
    session.get.assert_called_once_with(Prediction, record.prediction_id)

    updated = repository.set_review_status(
        record.prediction_id,
        ReviewStatus.CONFIRMED_FRAUD,
    )

    assert updated is record
    assert record.review_status == "confirmed_fraud"
    session.commit.assert_called_once()


def test_repository_review_update_returns_none_when_missing() -> None:
    session = MagicMock(spec=Session)
    session.get.return_value = None
    repository = PredictionRepository(session)

    assert repository.set_review_status(uuid4(), ReviewStatus.CONFIRMED_FRAUD) is None
    session.commit.assert_not_called()


def test_repository_list_recent_returns_a_bounded_newest_first_page() -> None:
    session = MagicMock(spec=Session)
    record = Prediction(
        prediction_id=uuid4(),
        transaction_id="txn-list",
        created_at=datetime.now(UTC),
        model_version="model-v1",
        model_sha256="a" * 64,
        risk_score=0.75,
        threshold=0.5,
        decision="fraud",
        review_status="pending",
    )
    session.scalars.return_value.all.return_value = [record]
    repository = PredictionRepository(session)

    assert repository.list_recent(limit=5, offset=10) == [record]

    statement = session.scalars.call_args.args[0]
    rendered = str(statement)
    assert "ORDER BY predictions.created_at DESC" in rendered
    assert "LIMIT" in rendered
    assert "OFFSET" in rendered


def test_repository_list_recent_applies_all_history_filters() -> None:
    session = MagicMock(spec=Session)
    session.scalars.return_value.all.return_value = []
    repository = PredictionRepository(session)
    created_from = datetime(2026, 1, 1, tzinfo=UTC)
    created_to = datetime(2026, 1, 31, 23, 59, 59, tzinfo=UTC)

    repository.list_recent(
        limit=10,
        created_from=created_from,
        created_to=created_to,
        min_risk_score=0.25,
        max_risk_score=0.75,
        decision=PredictionDecision.FRAUD,
        review_status=ReviewStatus.PENDING,
    )

    statement = session.scalars.call_args.args[0]
    rendered = str(statement)
    assert "predictions.created_at >=" in rendered
    assert "predictions.created_at <=" in rendered
    assert "predictions.risk_score >=" in rendered
    assert "predictions.risk_score <=" in rendered
    assert "predictions.decision =" in rendered
    assert "predictions.review_status =" in rendered


def test_repository_review_update_rejects_a_terminal_transition() -> None:
    session = MagicMock(spec=Session)
    record = Prediction(
        prediction_id=uuid4(),
        transaction_id="txn-terminal",
        created_at=datetime.now(UTC),
        model_version="model-v1",
        model_sha256="a" * 64,
        risk_score=0.75,
        threshold=0.5,
        decision="fraud",
        review_status="confirmed_fraud",
    )
    session.get.return_value = record
    repository = PredictionRepository(session)

    with pytest.raises(InvalidReviewTransitionError, match="not allowed"):
        repository.set_review_status(record.prediction_id, ReviewStatus.CONFIRMED_LEGITIMATE)

    assert record.review_status == "confirmed_fraud"
    session.commit.assert_not_called()


def test_repository_review_update_rejects_a_transition_back_to_pending() -> None:
    session = MagicMock(spec=Session)
    record = Prediction(
        prediction_id=uuid4(),
        transaction_id="txn-pending",
        created_at=datetime.now(UTC),
        model_version="model-v1",
        model_sha256="a" * 64,
        risk_score=0.75,
        threshold=0.5,
        decision="fraud",
        review_status="pending",
    )
    session.get.return_value = record
    repository = PredictionRepository(session)

    with pytest.raises(InvalidReviewTransitionError, match="not allowed"):
        repository.set_review_status(record.prediction_id, ReviewStatus.PENDING)

    assert record.review_status == "pending"
    session.commit.assert_not_called()


def test_repository_summarize_counts_and_risk_bins() -> None:
    session = MagicMock(spec=Session)
    row = {
        "total_predictions": 4,
        "fraud_predictions": 2,
        "legitimate_predictions": 2,
        "pending_reviews": 1,
        "confirmed_fraud": 2,
        "confirmed_legitimate": 1,
    }
    for index in range(10):
        row[f"risk_bin_{index}"] = 0
    row["risk_bin_0"] = 1
    row["risk_bin_7"] = 2
    row["risk_bin_9"] = 1
    session.execute.return_value.mappings.return_value.one.return_value = row
    repository = PredictionRepository(session)

    summary = repository.summarize()

    assert summary.total_predictions == 4
    assert summary.fraud_predictions == 2
    assert summary.legitimate_predictions == 2
    assert summary.pending_reviews == 1
    assert summary.confirmed_fraud == 2
    assert summary.confirmed_legitimate == 1
    assert [(item.lower, item.upper, item.count) for item in summary.risk_bins] == [
        (0.0, 0.1, 1),
        (0.1, 0.2, 0),
        (0.2, 0.3, 0),
        (0.3, 0.4, 0),
        (0.4, 0.5, 0),
        (0.5, 0.6, 0),
        (0.6, 0.7, 0),
        (0.7, 0.8, 2),
        (0.8, 0.9, 0),
        (0.9, 1.0, 1),
    ]
    session.execute.assert_called_once()
    session.rollback.assert_not_called()


def test_repository_rejects_an_invalid_history_page() -> None:
    session = MagicMock(spec=Session)
    repository = PredictionRepository(session)

    with pytest.raises(ValueError, match="limit"):
        repository.list_recent(limit=0)
    with pytest.raises(ValueError, match="offset"):
        repository.list_recent(offset=-1)

    session.scalars.assert_not_called()


def test_repository_history_list_failure_rolls_back() -> None:
    session = MagicMock(spec=Session)
    session.scalars.side_effect = SQLAlchemyError("connection lost")
    repository = PredictionRepository(session)

    with pytest.raises(PersistenceError, match="could not read prediction history"):
        repository.list_recent()

    session.rollback.assert_called_once()


def test_repository_summary_failure_rolls_back() -> None:
    session = MagicMock(spec=Session)
    session.execute.side_effect = SQLAlchemyError("connection lost")
    repository = PredictionRepository(session)

    with pytest.raises(PersistenceError, match="could not summarize prediction history"):
        repository.summarize()

    session.rollback.assert_called_once()


def test_repository_read_failure_rolls_back() -> None:
    session = MagicMock(spec=Session)
    session.get.side_effect = SQLAlchemyError("connection lost")
    repository = PredictionRepository(session)

    with pytest.raises(PersistenceError, match="could not read"):
        repository.get(uuid4())

    session.rollback.assert_called_once()


def test_repository_review_failure_rolls_back() -> None:
    session = MagicMock(spec=Session)
    record = Prediction(
        prediction_id=uuid4(),
        transaction_id="txn-review",
        created_at=datetime.now(UTC),
        model_version="model-v1",
        model_sha256="a" * 64,
        risk_score=0.75,
        threshold=0.5,
        decision="fraud",
        review_status="pending",
    )
    session.get.return_value = record
    session.commit.side_effect = SQLAlchemyError("database went away")
    repository = PredictionRepository(session)

    with pytest.raises(PersistenceError, match="could not update"):
        repository.set_review_status(record.prediction_id, ReviewStatus.CONFIRMED_FRAUD)

    session.rollback.assert_called_once()


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------
class _FakeSession:
    def __init__(self) -> None:
        self.rolled_back = False
        self.closed = False

    def rollback(self) -> None:
        self.rolled_back = True

    def close(self) -> None:
        self.closed = True


def _request_for(session: _FakeSession) -> Request:
    app = MagicMock()
    app.state = MagicMock()
    setattr(app.state, DB_SESSION_FACTORY_STATE_KEY, lambda: session)
    scope: dict[str, Any] = {
        "type": "http",
        "app": app,
        "method": "GET",
        "path": "/",
        "headers": [],
    }
    return Request(scope)


def test_session_dependency_closes_after_success() -> None:
    session = _FakeSession()
    generator = cast(
        Generator[Session, None, None],
        get_db_session(_request_for(session)),
    )
    yielded: Any = next(generator)

    assert yielded is session
    generator.close()

    assert session.closed is True
    assert session.rolled_back is False


def test_session_dependency_rolls_back_and_closes_on_failure() -> None:
    session = _FakeSession()
    generator = cast(
        Generator[Session, None, None],
        get_db_session(_request_for(session)),
    )
    yielded: Any = next(generator)

    assert yielded is session
    with pytest.raises(RuntimeError, match="route failed"):
        generator.throw(RuntimeError("route failed"))

    assert session.closed is True
    assert session.rolled_back is True


# ---------------------------------------------------------------------------
# Migration consistency (offline, no database connection)
# ---------------------------------------------------------------------------
def _alembic_config() -> Config:
    return Config(str(REPO_ROOT / "alembic.ini"))


def test_alembic_has_exactly_one_head() -> None:
    script = ScriptDirectory.from_config(_alembic_config())

    assert script.get_heads() == ["0003_create_performance_reports"]


def test_offline_migration_renders_the_prediction_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+psycopg://fraud:change-me@localhost:5432/fraud",
    )
    get_settings.cache_clear()
    config = _alembic_config()
    output = io.StringIO()
    config.output_buffer = output

    try:
        command.upgrade(config, "head", sql=True)
    finally:
        get_settings.cache_clear()

    sql = output.getvalue()
    assert "CREATE TABLE predictions" in sql
    for column in EXPECTED_COLUMNS:
        assert column in sql
    assert "CREATE TABLE monitoring_events" in sql
    assert "CREATE TABLE drift_reports" in sql
    assert "CREATE INDEX ix_predictions_created_at" in sql
    assert "CREATE INDEX ix_predictions_transaction_id" in sql
    assert "CREATE INDEX ix_monitoring_events_created_at" in sql
    assert "CREATE INDEX ix_drift_reports_generated_at" in sql
    assert "ck_predictions_risk_score_range" in sql
    assert "ck_predictions_review_status_allowed" in sql
    assert "ck_drift_reports_window_order" in sql
    assert "ck_predictions_ck_predictions" not in sql
    assert logging.getLogger("fraud_risk.monitoring.repository").disabled is False
