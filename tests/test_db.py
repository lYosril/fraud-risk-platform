"""Unit tests for the database contract and repository transaction policy.

These tests never connect to PostgreSQL. They pin the model shape (including the privacy
defaults), repository commit/rollback behaviour, the request-scoped session lifecycle and the
offline Alembic rendering. Real round-trip and schema/migration checks live in
``tests/test_db_integration.py`` and require a PostgreSQL service.
"""

from __future__ import annotations

import io
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
from fraud_risk.db.models import Prediction, PredictionDecision, ReviewStatus
from fraud_risk.db.repository import PersistenceError, PredictionCreate, PredictionRepository
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

    assert script.get_heads() == ["0001_create_predictions"]


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
    assert "CREATE INDEX ix_predictions_created_at" in sql
    assert "CREATE INDEX ix_predictions_transaction_id" in sql
    assert "ck_predictions_risk_score_range" in sql
    assert "ck_predictions_review_status_allowed" in sql
    assert "ck_predictions_ck_predictions" not in sql
