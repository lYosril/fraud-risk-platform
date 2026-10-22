"""PostgreSQL integration checks for migrations and repository round-trips.

These tests are skipped unless ``TEST_DATABASE_URL`` points at a reachable PostgreSQL database.
They never use SQLite. Run them with the Compose service:

    docker compose up -d postgres
    $env:TEST_DATABASE_URL = "postgresql+psycopg://fraud:change-me@localhost:5432/fraud"
    pytest -q -m integration tests/test_db_integration.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlalchemy import Engine, create_engine, delete, inspect, text
from sqlalchemy.exc import OperationalError

from fraud_risk.db.models import Prediction, PredictionDecision, ReviewStatus
from fraud_risk.db.repository import (
    InvalidReviewTransitionError,
    PredictionCreate,
    PredictionRepository,
)
from fraud_risk.db.session import build_session_factory

pytestmark = pytest.mark.integration

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


@pytest.fixture(scope="session")
def postgres_url() -> str:
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("set TEST_DATABASE_URL to run PostgreSQL integration tests")
    if not url.startswith("postgresql"):
        pytest.fail("TEST_DATABASE_URL must be a PostgreSQL SQLAlchemy URL")
    return url


@pytest.fixture(scope="session")
def postgres_engine(postgres_url: str) -> Iterator[Engine]:
    engine = create_engine(postgres_url, pool_pre_ping=True)
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except OperationalError as exc:
        engine.dispose()
        pytest.fail(f"PostgreSQL is not reachable at TEST_DATABASE_URL: {exc}")
    yield engine
    engine.dispose()


@pytest.fixture(scope="session")
def migrated_engine(postgres_engine: Engine, postgres_url: str) -> Engine:
    environment = {**os.environ, "DATABASE_URL": postgres_url}
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(f"alembic upgrade head failed:\n{result.stdout}\n{result.stderr}")
    return postgres_engine


def test_migrated_schema_matches_the_orm_models(migrated_engine: Engine) -> None:
    inspector = inspect(migrated_engine)

    assert "predictions" in inspector.get_table_names()
    database_columns = {column["name"] for column in inspector.get_columns("predictions")}
    assert database_columns == EXPECTED_COLUMNS
    assert database_columns == set(Prediction.__table__.columns.keys())

    check_constraints = {
        constraint["name"] for constraint in inspector.get_check_constraints("predictions")
    }
    assert {
        "ck_predictions_risk_score_range",
        "ck_predictions_threshold_range",
        "ck_predictions_decision_allowed",
        "ck_predictions_review_status_allowed",
    } <= check_constraints

    index_names = {index["name"] for index in inspector.get_indexes("predictions")}
    assert {"ix_predictions_created_at", "ix_predictions_transaction_id"} <= index_names


def test_prediction_round_trip_and_review_status_update(migrated_engine: Engine) -> None:
    transaction_id = f"integration-{uuid4()}"
    session_factory = build_session_factory(migrated_engine)

    try:
        with session_factory() as session:
            repository = PredictionRepository(session)
            created = repository.create_many(
                [
                    PredictionCreate(
                        transaction_id=transaction_id,
                        model_version="integration-model-v1",
                        model_sha256="b" * 64,
                        risk_score=0.81,
                        threshold=0.5,
                        decision=PredictionDecision.FRAUD,
                        feature_snapshot={
                            "amount": 42.5,
                            "features": {"V1": 0.1, "V2": -0.2},
                        },
                    )
                ]
            )
            prediction_id: UUID = created[0].prediction_id

            fetched = repository.get(prediction_id)
            assert fetched is not None
            assert fetched.transaction_id == transaction_id
            assert fetched.risk_score == pytest.approx(0.81)
            assert fetched.threshold == pytest.approx(0.5)
            assert fetched.decision == "fraud"
            assert fetched.review_status == "pending"
            assert fetched.feature_snapshot == {
                "amount": 42.5,
                "features": {"V1": 0.1, "V2": -0.2},
            }
            assert fetched.created_at.tzinfo is not None

            updated = repository.set_review_status(
                prediction_id,
                ReviewStatus.CONFIRMED_FRAUD,
            )
            assert updated is not None
            assert updated.review_status == "confirmed_fraud"

            listed = repository.list_recent(limit=100)
            assert any(record.prediction_id == prediction_id for record in listed)

            filtered = repository.list_recent(
                limit=100,
                created_from=fetched.created_at,
                created_to=fetched.created_at,
                min_risk_score=0.8,
                max_risk_score=0.82,
                decision=PredictionDecision.FRAUD,
                review_status=ReviewStatus.CONFIRMED_FRAUD,
            )
            assert any(record.prediction_id == prediction_id for record in filtered)
            assert all(record.decision == "fraud" for record in filtered)
            assert all(record.review_status == "confirmed_fraud" for record in filtered)

            with pytest.raises(InvalidReviewTransitionError, match="not allowed"):
                repository.set_review_status(
                    prediction_id,
                    ReviewStatus.CONFIRMED_LEGITIMATE,
                )
            unchanged = repository.get(prediction_id)
            assert unchanged is not None
            assert unchanged.review_status == "confirmed_fraud"

            summary = repository.summarize()
            assert summary.total_predictions >= 1
            assert summary.fraud_predictions >= 1
            assert sum(item.count for item in summary.risk_bins) == summary.total_predictions
    finally:
        with session_factory() as cleanup:
            cleanup.execute(delete(Prediction).where(Prediction.transaction_id == transaction_id))
            cleanup.commit()
