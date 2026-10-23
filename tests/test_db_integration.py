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
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import Engine, create_engine, delete, inspect, text
from sqlalchemy.exc import OperationalError

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
    PredictionCreate,
    PredictionRepository,
)
from fraud_risk.db.session import build_session_factory
from fraud_risk.monitoring.drift import DriftStatus
from fraud_risk.monitoring.performance import (
    MODEL_VERSION_COLUMN,
    PERFORMANCE_FEATURE_COLUMNS,
    PERFORMANCE_REQUIRED_COLUMNS,
    SCORE_COLUMN,
    SCORED_AT_COLUMN,
    TARGET_COLUMN,
    LabeledEvaluationDataset,
    build_performance_report,
    prepare_performance_frame,
)
from fraud_risk.monitoring.performance_repository import PerformanceRepository
from fraud_risk.monitoring.reference import build_reference_profile
from fraud_risk.monitoring.report import (
    AggregateCounts,
    AnalysisSample,
    ModelVersionUsage,
    MonitoringReport,
    MonitoringThresholds,
    build_monitoring_report,
)
from fraud_risk.monitoring.repository import MonitoringRepository

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

    assert "monitoring_events" in inspector.get_table_names()
    assert "drift_reports" in inspector.get_table_names()
    assert {column["name"] for column in inspector.get_columns("monitoring_events")} == set(
        MonitoringEvent.__table__.columns.keys()
    )
    assert {column["name"] for column in inspector.get_columns("drift_reports")} == set(
        DriftReport.__table__.columns.keys()
    )

    assert "performance_reports" in inspector.get_table_names()
    assert {column["name"] for column in inspector.get_columns("performance_reports")} == set(
        PerformanceReport.__table__.columns.keys()
    )
    performance_constraints = {
        constraint["name"] for constraint in inspector.get_check_constraints("performance_reports")
    }
    assert {
        "ck_performance_reports_row_count_positive",
        "ck_performance_reports_model_version_count_positive",
        "ck_performance_reports_segment_count_positive",
        "ck_performance_reports_threshold_range",
        "ck_performance_reports_period_allowed",
        "ck_performance_reports_window_order",
    } <= performance_constraints
    performance_indexes = {index["name"] for index in inspector.get_indexes("performance_reports")}
    assert "ix_performance_reports_generated_at" in performance_indexes


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


def test_shifted_window_persists_an_alert_report(migrated_engine: Engine) -> None:
    session_factory = build_session_factory(migrated_engine)
    now = datetime.now(UTC)
    report_time = now + timedelta(days=2)
    rng = np.random.default_rng(3)
    reference_frame = pd.DataFrame(
        {
            "V1": rng.normal(0.0, 1.0, 100),
            "V2": rng.normal(5.0, 2.0, 100),
            "Amount": rng.uniform(0.0, 100.0, 100),
        }
    )
    reference_scores = rng.beta(2.0, 8.0, 100)
    reference = build_reference_profile(
        reference_frame,
        reference_scores,
        feature_columns=("V1", "V2", "Amount"),
        source="synthetic integration fixture",
        model_version="test-model",
        model_sha256="d" * 64,
        threshold=0.5,
        split_strategy="temporal",
        split_seed=42,
        training_rows=200,
        sample_method="all test rows",
        generated_at=now,
    )
    shifted = reference_frame + 10.0
    rows = tuple(
        {column: float(shifted.iloc[index][column]) for column in ("V1", "V2", "Amount")}
        for index in range(len(shifted))
    )
    sample = AnalysisSample(
        scores=tuple(float(value) for value in np.full(100, 0.99)),
        feature_rows=rows,
        snapshot_count=len(rows),
    )
    aggregate = AggregateCounts(
        prediction_count=100,
        fraud_count=0,
        legitimate_count=100,
        model_versions=(
            ModelVersionUsage(
                model_version="test-model",
                total=100,
                fraud=0,
                legitimate=100,
            ),
        ),
        review_status_counts={},
        validation_failure_count=0,
    )
    report = build_monitoring_report(
        reference=reference,
        aggregate=aggregate,
        sample=sample,
        thresholds=MonitoringThresholds(
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
        ),
        window_start=report_time - timedelta(hours=1),
        window_end=report_time,
        generated_at=report_time,
    )

    try:
        assert report.status is DriftStatus.ALERT
        with session_factory() as session:
            repository = MonitoringRepository(session)
            repository.save_report(report)
            latest = repository.latest_report()
            assert latest is not None
            assert latest.report_id == report.report_id
            assert latest.status == "alert"
            assert latest.payload["feature_drift"]["breached_feature_count"] >= 1
    finally:
        with session_factory() as cleanup:
            cleanup.execute(delete(DriftReport).where(DriftReport.report_id == report.report_id))
            cleanup.commit()


def test_monitoring_event_and_report_round_trip(migrated_engine: Engine) -> None:
    session_factory = build_session_factory(migrated_engine)
    now = datetime.now(UTC)
    window_start = now - timedelta(hours=1)
    report_time = now + timedelta(days=1)
    payload = {
        "schema_version": 1,
        "summary": {"status": "ok", "prediction_count": 0},
    }
    report = MonitoringReport(
        report_id=uuid4(),
        generated_at=report_time,
        window_start=report_time - timedelta(hours=1),
        window_end=report_time,
        reference_id="c" * 64,
        status=DriftStatus.OK,
        data_drift_status=DriftStatus.OK,
        data_quality_status=DriftStatus.OK,
        prediction_count=0,
        validation_failure_count=3,
        feature_snapshot_count=0,
        feature_drift_count=0,
        payload=payload,
    )

    try:
        with session_factory() as session:
            repository = MonitoringRepository(session)
            repository.record_validation_failures(
                endpoint="/api/v1/predict",
                reason_code="request_validation",
                count=3,
            )
            aggregate = repository.window_aggregate(
                start=window_start,
                end=datetime.now(UTC) + timedelta(seconds=1),
            )
            assert aggregate.validation_failure_count >= 3

            repository.save_report(report)
            latest = repository.latest_report()
            assert latest is not None
            assert latest.report_id == report.report_id
            assert latest.status == "ok"
            assert latest.payload == payload
    finally:
        with session_factory() as cleanup:
            cleanup.execute(delete(DriftReport).where(DriftReport.report_id == report.report_id))
            cleanup.execute(
                delete(MonitoringEvent).where(MonitoringEvent.created_at >= window_start)
            )
            cleanup.commit()


def test_performance_report_round_trip(migrated_engine: Engine) -> None:
    session_factory = build_session_factory(migrated_engine)
    generated_at = datetime.now(UTC) + timedelta(days=3)
    scored_at = generated_at - timedelta(hours=1)
    rows: list[dict[str, object]] = []
    for label, score in ((1, 0.9), (0, 0.1)):
        row: dict[str, object] = dict.fromkeys(PERFORMANCE_FEATURE_COLUMNS, 0.0)
        row.update(
            {
                TARGET_COLUMN: label,
                SCORE_COLUMN: score,
                MODEL_VERSION_COLUMN: "integration-performance-v1",
                SCORED_AT_COLUMN: scored_at.isoformat(),
            }
        )
        rows.append(row)
    frame = pd.DataFrame(rows, columns=list(PERFORMANCE_REQUIRED_COLUMNS))
    dataset = LabeledEvaluationDataset(
        frame=prepare_performance_frame(frame),
        source_sha256="d" * 64,
        source_rows=len(frame),
    )
    report = build_performance_report(
        dataset=dataset,
        threshold=0.5,
        period="month",
        label_source="verified_chargebacks",
        labels_verified=True,
        generated_at=generated_at,
    )

    try:
        with session_factory() as session:
            repository = PerformanceRepository(session)
            repository.save_report(report)
            latest = repository.latest_report()
            assert latest is not None
            assert latest.report_id == report.report_id
            assert latest.row_count == 2
            assert latest.model_version_count == 1
            assert latest.segment_count == 1
            assert latest.labels_verified is True
            assert latest.label_source == "verified_chargebacks"
            assert latest.payload == report.payload
    finally:
        with session_factory() as cleanup:
            cleanup.execute(
                delete(PerformanceReport).where(PerformanceReport.report_id == report.report_id)
            )
            cleanup.commit()
