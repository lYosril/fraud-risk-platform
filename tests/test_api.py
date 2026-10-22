"""HTTP tests for the versioned FastAPI backend.

The tests use HTTPX's ASGI transport directly, with dependency overrides for the inference
service. No real artifact or PostgreSQL instance is needed: the model path is covered by the
inference suite, and these tests pin the HTTP contract, error mapping, readiness truthfulness,
CORS policy, factory laziness and production debug-tool removal.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI

from fraud_risk import __version__
from fraud_risk.api.dependencies import get_inference_service, get_prediction_repository
from fraud_risk.api.main import create_app
from fraud_risk.config import Settings
from fraud_risk.db.models import Prediction, PredictionDecision, ReviewStatus
from fraud_risk.db.repository import PersistenceError, PredictionCreate
from fraud_risk.models.bundle import FrozenModel
from fraud_risk.models.evaluation import CostPolicy
from fraud_risk.models.inference import (
    InferenceInputError,
    InferenceModelError,
    InferenceRequest,
    PredictionResult,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_SHA = "a" * 64
MODEL_VERSION = "api-test-version"
FEATURE_COLUMNS = ("V1", "V2", "Amount")
ADMIN_TOKEN = "test-admin-token"  # noqa: S105 - synthetic test credential
ADMIN_HEADERS = {"Authorization": f"Bearer {ADMIN_TOKEN}"}


def make_model() -> FrozenModel:
    """A typed bundle record sufficient for metadata and readiness routes."""
    return FrozenModel(
        path=Path("api-test-model.joblib"),
        sha256=MODEL_SHA,
        model_version=MODEL_VERSION,
        pipeline=object(),
        feature_columns=FEATURE_COLUMNS,
        threshold=0.5,
        cost_policy=CostPolicy(100.0, 1.0),
        seed=42,
        metadata={"family": "xgboost"},
    )


class StubInferenceService:
    """A minimal stand-in for the real service; scoring itself is tested elsewhere."""

    def __init__(
        self,
        *,
        model: FrozenModel | None = None,
        error: Exception | None = None,
    ) -> None:
        self._model = model
        self._error = error
        self.predict_calls = 0
        self.last_request: InferenceRequest | None = None

    @property
    def model(self) -> FrozenModel:
        if self._error is not None:
            raise self._error
        if self._model is None:
            raise AssertionError("stub has neither a model nor an error")
        return self._model

    def predict(
        self,
        request: InferenceRequest,
        *,
        explain: bool = False,
        top_k: int = 10,
    ) -> PredictionResult:
        self.predict_calls += 1
        self.last_request = request
        if self._error is not None:
            raise self._error
        model = self.model
        return PredictionResult(
            transaction_id=request.transaction_id,
            probability=0.75,
            threshold=model.threshold,
            prediction=1,
            model_version=model.model_version,
            model_sha256=model.sha256,
            model_family=model.family or "unknown",
            explanation=None,
        )


class FeatureCheckingStub(StubInferenceService):
    """Reject an unknown feature the way the real service's schema check does."""

    def predict(
        self,
        request: InferenceRequest,
        *,
        explain: bool = False,
        top_k: int = 10,
    ) -> PredictionResult:
        if "V999" in request.features:
            raise InferenceInputError("feature schema mismatch (unexpected: V999)")
        return super().predict(request, explain=explain, top_k=top_k)


class FakePredictionRepository:
    """Record persistence calls without opening a PostgreSQL connection."""

    def __init__(self, *, error: Exception | None = None) -> None:
        self.rows: list[PredictionCreate] = []
        self.records: dict[UUID, Prediction] = {}
        self.error = error

    def create_many(self, rows: Sequence[PredictionCreate]) -> list[object]:
        if self.error is not None:
            raise self.error
        self.rows.extend(rows)
        return []

    def get(self, prediction_id: UUID) -> Prediction | None:
        if self.error is not None:
            raise self.error
        return self.records.get(prediction_id)

    def set_review_status(
        self,
        prediction_id: UUID,
        review_status: ReviewStatus,
    ) -> Prediction | None:
        if self.error is not None:
            raise self.error
        record = self.records.get(prediction_id)
        if record is None:
            return None
        record.review_status = ReviewStatus(review_status).value
        return record


def make_stored_prediction(
    *,
    prediction_id: UUID | None = None,
    review_status: ReviewStatus = ReviewStatus.PENDING,
) -> Prediction:
    """Build an ORM-shaped record for the read/review routes without a database."""
    return Prediction(
        prediction_id=prediction_id or uuid4(),
        transaction_id="txn-stored",
        created_at=datetime.now(UTC),
        model_version=MODEL_VERSION,
        model_sha256=MODEL_SHA,
        risk_score=0.75,
        threshold=0.5,
        decision=PredictionDecision.FRAUD.value,
        review_status=review_status.value,
        feature_snapshot={"amount": 42.5, "features": {"V1": 0.1, "V2": -0.2}},
    )


def make_app(
    *,
    service: StubInferenceService | None = None,
    repository: FakePredictionRepository | None = None,
    **settings_overrides: Any,
) -> FastAPI:
    settings = Settings(  # type: ignore[call-arg]  # `_env_file` is runtime-only
        _env_file=None,
        **settings_overrides,
    )
    app = create_app(settings)
    if service is not None:
        app.dependency_overrides[get_inference_service] = lambda: service
    resolved_repository = repository if repository is not None else FakePredictionRepository()
    app.dependency_overrides[get_prediction_repository] = lambda: resolved_repository
    return app


def prediction_payload(transaction_id: str) -> dict[str, Any]:
    """A valid synthetic transaction body; feature values are illustrative only."""
    return {
        "transaction_id": transaction_id,
        "amount": 42.5,
        "features": {"V1": 0.1, "V2": -0.2},
    }


def expected_prediction(transaction_id: str) -> dict[str, Any]:
    """The response the stub produces for any valid transaction in these tests."""
    return {
        "transaction_id": transaction_id,
        "probability": 0.75,
        "threshold": 0.5,
        "prediction": 1,
        "decision": "fraud",
        "model_version": MODEL_VERSION,
        "model_sha256": MODEL_SHA,
        "model_family": "xgboost",
        "explanation": None,
    }


async def _request(
    app: FastAPI,
    method: str,
    path: str,
    *,
    raise_app_exceptions: bool = True,
    **kwargs: Any,
) -> httpx.Response:
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=raise_app_exceptions)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.request(method, path, **kwargs)


def request(
    app: FastAPI,
    method: str,
    path: str,
    *,
    raise_app_exceptions: bool = True,
    **kwargs: Any,
) -> httpx.Response:
    """Run one request through HTTPX's ASGI transport without pytest-asyncio."""
    return asyncio.run(
        _request(
            app,
            method,
            path,
            raise_app_exceptions=raise_app_exceptions,
            **kwargs,
        )
    )


# ---------------------------------------------------------------------------
# Versioned health, readiness and model metadata
# ---------------------------------------------------------------------------
def test_health_is_versioned_and_distinct_from_readiness() -> None:
    app = make_app(service=StubInferenceService(model=make_model()))

    response = request(app, "GET", "/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "fraud-risk-api",
        "version": __version__,
        "environment": "local",
    }
    assert request(app, "GET", "/health").status_code == 404


def test_readiness_reports_a_loadable_model_as_ready() -> None:
    app = make_app(service=StubInferenceService(model=make_model()))

    response = request(app, "GET", "/api/v1/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "model_available": True,
        "detail": None,
    }


def test_readiness_reports_a_missing_model_as_not_ready() -> None:
    app = make_app(
        service=StubInferenceService(
            error=InferenceModelError("missing artifact at C:/private/model.joblib")
        )
    )

    response = request(app, "GET", "/api/v1/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "model_available": False,
        "detail": "model unavailable",
    }
    assert "private" not in response.text


def test_model_metadata_uses_the_loaded_bundle_and_configured_name() -> None:
    app = make_app(
        service=StubInferenceService(model=make_model()),
        model_name="fraud-risk-test",
        api_admin_token=ADMIN_TOKEN,
    )

    response = request(app, "GET", "/api/v1/model", headers=ADMIN_HEADERS)

    assert response.status_code == 200
    assert response.json() == {
        "model_name": "fraud-risk-test",
        "model_version": MODEL_VERSION,
        "model_sha256": MODEL_SHA,
        "model_family": "xgboost",
        "threshold": 0.5,
        "feature_count": 3,
        "feature_columns": ["V1", "V2", "Amount"],
    }


def test_model_metadata_maps_a_missing_artifact_to_a_safe_503(tmp_path: Path) -> None:
    app = make_app(
        model_artifact_path=tmp_path / "missing.joblib",
        api_admin_token=ADMIN_TOKEN,
    )

    response = request(app, "GET", "/api/v1/model", headers=ADMIN_HEADERS)

    assert response.status_code == 503
    assert response.json() == {"detail": "model unavailable"}
    assert str(tmp_path) not in response.text


def test_admin_routes_fail_closed_when_the_token_is_not_configured() -> None:
    app = make_app(service=StubInferenceService(model=make_model()))
    prediction_id = uuid4()

    for method, path in (
        ("GET", "/api/v1/model"),
        ("GET", f"/api/v1/predictions/{prediction_id}"),
        ("PATCH", f"/api/v1/predictions/{prediction_id}/review"),
    ):
        kwargs: dict[str, Any] = (
            {"json": {"review_status": "confirmed_fraud"}} if method == "PATCH" else {}
        )
        response = request(app, method, path, **kwargs)

        assert response.status_code == 503
        assert response.json() == {"detail": "administrative API is not configured"}


def test_model_metadata_requires_the_configured_bearer_token() -> None:
    app = make_app(
        service=StubInferenceService(model=make_model()),
        api_admin_token=ADMIN_TOKEN,
    )

    missing = request(app, "GET", "/api/v1/model")
    wrong = request(
        app,
        "GET",
        "/api/v1/model",
        headers={"Authorization": "Bearer not-the-admin-token"},
    )
    valid = request(app, "GET", "/api/v1/model", headers=ADMIN_HEADERS)

    assert missing.status_code == 401
    assert missing.json() == {"detail": "invalid or missing credentials"}
    assert missing.headers["www-authenticate"] == "Bearer"
    assert wrong.status_code == 401
    assert wrong.json() == {"detail": "invalid or missing credentials"}
    assert valid.status_code == 200


def test_prediction_history_returns_no_feature_snapshot() -> None:
    repository = FakePredictionRepository()
    record = make_stored_prediction()
    repository.records[record.prediction_id] = record
    app = make_app(
        service=StubInferenceService(model=make_model()),
        repository=repository,
        api_admin_token=ADMIN_TOKEN,
    )

    response = request(
        app,
        "GET",
        f"/api/v1/predictions/{record.prediction_id}",
        headers=ADMIN_HEADERS,
    )

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "prediction_id": str(record.prediction_id),
        "transaction_id": "txn-stored",
        "created_at": body["created_at"],
        "model_version": MODEL_VERSION,
        "model_sha256": MODEL_SHA,
        "risk_score": 0.75,
        "threshold": 0.5,
        "decision": "fraud",
        "review_status": "pending",
    }
    assert "feature_snapshot" not in body
    assert "V1" not in response.text


def test_prediction_history_requires_auth_and_reports_an_unknown_id() -> None:
    repository = FakePredictionRepository()
    app = make_app(
        service=StubInferenceService(model=make_model()),
        repository=repository,
        api_admin_token=ADMIN_TOKEN,
    )
    missing_id = uuid4()

    unauthenticated = request(app, "GET", f"/api/v1/predictions/{missing_id}")
    unknown = request(
        app,
        "GET",
        f"/api/v1/predictions/{missing_id}",
        headers=ADMIN_HEADERS,
    )

    assert unauthenticated.status_code == 401
    assert unknown.status_code == 404
    assert unknown.json() == {"detail": "prediction not found"}


def test_prediction_history_maps_a_storage_failure_to_a_safe_503() -> None:
    repository = FakePredictionRepository(error=PersistenceError("database is down"))
    app = make_app(
        service=StubInferenceService(model=make_model()),
        repository=repository,
        api_admin_token=ADMIN_TOKEN,
    )

    response = request(
        app,
        "GET",
        f"/api/v1/predictions/{uuid4()}",
        headers=ADMIN_HEADERS,
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "prediction storage unavailable"}
    assert "database is down" not in response.text


def test_review_update_is_authenticated_and_persists_the_transition() -> None:
    repository = FakePredictionRepository()
    record = make_stored_prediction()
    repository.records[record.prediction_id] = record
    app = make_app(
        service=StubInferenceService(model=make_model()),
        repository=repository,
        api_admin_token=ADMIN_TOKEN,
    )
    path = f"/api/v1/predictions/{record.prediction_id}/review"

    unauthenticated = request(app, "PATCH", path, json={"review_status": "confirmed_fraud"})
    updated = request(
        app,
        "PATCH",
        path,
        headers=ADMIN_HEADERS,
        json={"review_status": "confirmed_fraud"},
    )

    assert unauthenticated.status_code == 401
    assert updated.status_code == 200
    assert updated.json()["review_status"] == "confirmed_fraud"
    assert record.review_status == "confirmed_fraud"


def test_review_update_rejects_an_invalid_status_and_unknown_record() -> None:
    repository = FakePredictionRepository()
    record = make_stored_prediction()
    repository.records[record.prediction_id] = record
    app = make_app(
        service=StubInferenceService(model=make_model()),
        repository=repository,
        api_admin_token=ADMIN_TOKEN,
    )

    invalid = request(
        app,
        "PATCH",
        f"/api/v1/predictions/{record.prediction_id}/review",
        headers=ADMIN_HEADERS,
        json={"review_status": "maybe"},
    )
    unknown = request(
        app,
        "PATCH",
        f"/api/v1/predictions/{uuid4()}/review",
        headers=ADMIN_HEADERS,
        json={"review_status": "confirmed_fraud"},
    )

    assert invalid.status_code == 422
    assert record.review_status == "pending"
    assert unknown.status_code == 404


def test_history_path_rejects_an_invalid_uuid_before_the_repository() -> None:
    repository = FakePredictionRepository()
    app = make_app(
        service=StubInferenceService(model=make_model()),
        repository=repository,
        api_admin_token=ADMIN_TOKEN,
    )

    response = request(app, "GET", "/api/v1/predictions/not-a-uuid", headers=ADMIN_HEADERS)

    assert response.status_code == 422
    assert "not-a-uuid" not in response.text


def test_admin_token_is_not_echoed_in_the_response_or_access_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    app = make_app(
        service=StubInferenceService(model=make_model()),
        api_admin_token=ADMIN_TOKEN,
    )

    response = request(
        app,
        "GET",
        "/api/v1/model",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}-wrong"},
    )

    assert response.status_code == 401
    assert ADMIN_TOKEN not in response.text
    assert ADMIN_TOKEN not in caplog.text


# ---------------------------------------------------------------------------
# Prediction contract and HTTP error mapping
# ---------------------------------------------------------------------------
def test_prediction_uses_the_inference_service_dependency() -> None:
    service = StubInferenceService(model=make_model())
    repository = FakePredictionRepository()
    app = make_app(service=service, repository=repository)

    response = request(
        app,
        "POST",
        "/api/v1/predict",
        json={
            "transaction_id": "txn-1",
            "amount": 42.5,
            "features": {"V1": 0.1, "V2": -0.2},
            "explain": False,
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "transaction_id": "txn-1",
        "probability": 0.75,
        "threshold": 0.5,
        "prediction": 1,
        "decision": "fraud",
        "model_version": MODEL_VERSION,
        "model_sha256": MODEL_SHA,
        "model_family": "xgboost",
        "explanation": None,
    }
    assert service.predict_calls == 1
    assert service.last_request is not None
    assert tuple(service.last_request.features) == ("V1", "V2")
    assert service.last_request.amount == 42.5
    assert len(repository.rows) == 1
    persisted = repository.rows[0]
    assert persisted.transaction_id == "txn-1"
    assert persisted.model_version == MODEL_VERSION
    assert persisted.model_sha256 == MODEL_SHA
    assert persisted.risk_score == 0.75
    assert persisted.threshold == 0.5
    assert persisted.decision is PredictionDecision.FRAUD
    assert persisted.review_status is ReviewStatus.PENDING
    assert persisted.feature_snapshot is None


def test_prediction_persists_a_feature_snapshot_only_when_opted_in() -> None:
    repository = FakePredictionRepository()
    app = make_app(
        service=StubInferenceService(model=make_model()),
        repository=repository,
        persist_feature_snapshot=True,
    )

    response = request(
        app,
        "POST",
        "/api/v1/predict",
        json=prediction_payload("txn-snapshot"),
    )

    assert response.status_code == 200
    assert repository.rows[0].feature_snapshot == {
        "amount": 42.5,
        "features": {"V1": 0.1, "V2": -0.2},
    }


def test_persistence_failure_maps_to_a_safe_503() -> None:
    repository = FakePredictionRepository(error=PersistenceError("database is down"))
    app = make_app(
        service=StubInferenceService(model=make_model()),
        repository=repository,
    )

    response = request(
        app,
        "POST",
        "/api/v1/predict",
        json=prediction_payload("txn-storage-failure"),
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "prediction storage unavailable"}
    assert "database is down" not in response.text


def test_unexpected_failure_returns_a_safe_500() -> None:
    repository = FakePredictionRepository(error=RuntimeError("private path C:/secret/model.joblib"))
    app = make_app(
        service=StubInferenceService(model=make_model()),
        repository=repository,
    )

    response = request(
        app,
        "POST",
        "/api/v1/predict",
        json=prediction_payload("txn-unexpected-failure"),
        raise_app_exceptions=False,
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "internal server error"}
    assert "private" not in response.text
    assert "RuntimeError" not in response.text


def test_prediction_validation_failure_does_not_reach_the_service() -> None:
    service = StubInferenceService(model=make_model())
    app = make_app(service=service)

    response = request(
        app,
        "POST",
        "/api/v1/predict",
        json={
            "transaction_id": "txn-1",
            "amount": 42.5,
            "features": {"V1": 0.1, "V2": -0.2},
            "is_fraud": 0,
        },
    )

    assert response.status_code == 422
    assert service.predict_calls == 0


def test_request_validation_error_does_not_echo_the_payload() -> None:
    service = StubInferenceService(model=make_model())
    app = make_app(service=service)

    response = request(
        app,
        "POST",
        "/api/v1/predict",
        json={
            "transaction_id": "txn-1",
            "amount": -5.0,
            "features": {"V1": 987654.321, "V2": -0.2},
        },
    )

    assert response.status_code == 422
    assert set(response.json()) == {"detail"}
    assert isinstance(response.json()["detail"], str)
    assert "987654.321" not in response.text
    assert service.predict_calls == 0


def test_inference_input_error_maps_to_422() -> None:
    app = make_app(
        service=StubInferenceService(error=InferenceInputError("feature order does not match"))
    )

    response = request(
        app,
        "POST",
        "/api/v1/predict",
        json={
            "transaction_id": "txn-1",
            "amount": 42.5,
            "features": {"V1": 0.1, "V2": -0.2},
        },
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "feature order does not match"}


def test_inference_model_error_maps_to_a_safe_503() -> None:
    app = make_app(
        service=StubInferenceService(
            error=InferenceModelError("corrupt artifact at C:/private/model.joblib")
        )
    )

    response = request(
        app,
        "POST",
        "/api/v1/predict",
        json={
            "transaction_id": "txn-1",
            "amount": 42.5,
            "features": {"V1": 0.1, "V2": -0.2},
        },
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "model unavailable"}
    assert "private" not in response.text


# ---------------------------------------------------------------------------
# Batch prediction contract
# ---------------------------------------------------------------------------
def test_batch_prediction_scores_every_item_and_returns_model_identity() -> None:
    service = StubInferenceService(model=make_model())
    repository = FakePredictionRepository()
    app = make_app(service=service, repository=repository)

    response = request(
        app,
        "POST",
        "/api/v1/predict/batch",
        json={"items": [prediction_payload("txn-1"), prediction_payload("txn-2")]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == MODEL_VERSION
    assert body["model_sha256"] == MODEL_SHA
    assert body["model_family"] == "xgboost"
    assert body["threshold"] == 0.5
    assert body["total"] == 2
    assert body["succeeded"] == 2
    assert body["failed"] == 0
    assert body["results"] == [
        {
            "index": 0,
            "transaction_id": "txn-1",
            "status": "ok",
            "prediction": expected_prediction("txn-1"),
            "error": None,
        },
        {
            "index": 1,
            "transaction_id": "txn-2",
            "status": "ok",
            "prediction": expected_prediction("txn-2"),
            "error": None,
        },
    ]
    assert service.predict_calls == 2
    assert [row.transaction_id for row in repository.rows] == ["txn-1", "txn-2"]
    assert all(row.decision is PredictionDecision.FRAUD for row in repository.rows)


def test_batch_returns_per_item_errors_without_aborting_the_batch() -> None:
    service = FeatureCheckingStub(model=make_model())
    repository = FakePredictionRepository()
    app = make_app(service=service, repository=repository)
    invalid_label = {**prediction_payload("txn-label"), "is_fraud": 0}
    invalid_numeric = {
        "transaction_id": "txn-numeric",
        "amount": 42.5,
        "features": {"V1": "0.1", "V2": -0.2},
    }
    unknown_feature = {
        "transaction_id": "txn-feature",
        "amount": 42.5,
        "features": {"V1": 0.1, "V2": -0.2, "V999": 0.0},
    }

    response = request(
        app,
        "POST",
        "/api/v1/predict/batch",
        json={
            "items": [
                prediction_payload("txn-1"),
                invalid_label,
                invalid_numeric,
                unknown_feature,
                prediction_payload("txn-2"),
            ]
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 5
    assert body["succeeded"] == 2
    assert body["failed"] == 3
    assert [item["status"] for item in body["results"]] == [
        "ok",
        "error",
        "error",
        "error",
        "ok",
    ]
    assert body["results"][1]["transaction_id"] == "txn-label"
    assert body["results"][1]["error"].startswith("is_fraud:")
    assert body["results"][2]["transaction_id"] == "txn-numeric"
    assert body["results"][2]["error"].startswith("features.V1:")
    assert "0.1" not in body["results"][2]["error"]
    assert body["results"][3]["error"] == "feature schema mismatch (unexpected: V999)"
    assert body["results"][4]["prediction"] == expected_prediction("txn-2")
    assert service.predict_calls == 2
    assert [row.transaction_id for row in repository.rows] == ["txn-1", "txn-2"]


def test_batch_rejects_an_empty_items_list() -> None:
    app = make_app(service=StubInferenceService(model=make_model()))

    response = request(app, "POST", "/api/v1/predict/batch", json={"items": []})

    assert response.status_code == 422
    assert response.json()["detail"].startswith("items:")


def test_batch_rejects_an_oversized_batch() -> None:
    service = StubInferenceService(model=make_model())
    repository = FakePredictionRepository()
    app = make_app(service=service, repository=repository, batch_max_size=2)

    response = request(
        app,
        "POST",
        "/api/v1/predict/batch",
        json={
            "items": [
                prediction_payload("txn-1"),
                prediction_payload("txn-2"),
                prediction_payload("txn-3"),
            ]
        },
    )

    assert response.status_code == 413
    assert "maximum is 2" in response.json()["detail"]
    assert service.predict_calls == 0
    assert repository.rows == []


def test_batch_maps_a_missing_model_to_a_safe_503() -> None:
    repository = FakePredictionRepository()
    app = make_app(
        service=StubInferenceService(
            error=InferenceModelError("missing artifact at C:/private/model.joblib")
        ),
        repository=repository,
    )

    response = request(
        app,
        "POST",
        "/api/v1/predict/batch",
        json={"items": [prediction_payload("txn-1")]},
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "model unavailable"}
    assert "private" not in response.text
    assert repository.rows == []


def test_batch_persistence_failure_maps_to_a_safe_503() -> None:
    repository = FakePredictionRepository(error=PersistenceError("database is down"))
    app = make_app(
        service=StubInferenceService(model=make_model()),
        repository=repository,
    )

    response = request(
        app,
        "POST",
        "/api/v1/predict/batch",
        json={"items": [prediction_payload("txn-1"), prediction_payload("txn-2")]},
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "prediction storage unavailable"}
    assert "database is down" not in response.text


def test_request_size_limit_rejects_an_oversized_body_before_scoring() -> None:
    service = StubInferenceService(model=make_model())
    app = make_app(service=service, api_max_request_bytes=1024)

    response = request(
        app,
        "POST",
        "/api/v1/predict/batch",
        json={"items": [prediction_payload("x" * 2000)]},
    )

    assert response.status_code == 413
    assert response.json() == {"detail": "request body too large"}
    assert service.predict_calls == 0


def test_batch_rejects_a_non_object_item_as_an_invalid_envelope() -> None:
    app = make_app(service=StubInferenceService(model=make_model()))

    response = request(
        app,
        "POST",
        "/api/v1/predict/batch",
        json={"items": [42]},
    )

    assert response.status_code == 422
    assert set(response.json()) == {"detail"}


# ---------------------------------------------------------------------------
# Factory, CORS and production hardening
# ---------------------------------------------------------------------------
def test_factory_is_lazy_and_readiness_detects_the_missing_artifact(tmp_path: Path) -> None:
    app = make_app(model_artifact_path=tmp_path / "not-created" / "model.joblib")

    response = request(app, "GET", "/api/v1/ready")

    assert response.status_code == 503
    assert response.json()["model_available"] is False


def test_cors_is_configured_from_settings_and_rejects_unlisted_origins() -> None:
    app = make_app(
        service=StubInferenceService(model=make_model()),
        cors_allowed_origins="https://dashboard.example",
    )

    allowed = request(
        app,
        "GET",
        "/api/v1/health",
        headers={"Origin": "https://dashboard.example"},
    )
    denied = request(
        app,
        "GET",
        "/api/v1/health",
        headers={"Origin": "https://evil.example"},
    )

    assert allowed.headers["access-control-allow-origin"] == "https://dashboard.example"
    assert "access-control-allow-origin" not in denied.headers


def test_cors_preflight_allows_admin_bearer_headers_and_patch() -> None:
    app = make_app(
        service=StubInferenceService(model=make_model()),
        cors_allowed_origins="https://dashboard.example",
    )

    response = request(
        app,
        "OPTIONS",
        "/api/v1/predictions/00000000-0000-0000-0000-000000000000/review",
        headers={
            "Origin": "https://dashboard.example",
            "Access-Control-Request-Method": "PATCH",
            "Access-Control-Request-Headers": "Authorization, Content-Type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://dashboard.example"
    allow_methods = response.headers["access-control-allow-methods"]
    allow_headers = response.headers["access-control-allow-headers"].lower()
    assert "PATCH" in allow_methods
    assert "authorization" in allow_headers


def test_production_disables_interactive_docs_and_openapi() -> None:
    app = make_app(
        app_env="production",
        database_url="postgresql+psycopg://user:real@db.internal:5432/fraud",
        cors_allowed_origins="https://dashboard.example",
    )

    assert app.docs_url is None
    assert app.redoc_url is None
    assert app.openapi_url is None
    assert app.debug is False
    assert request(app, "GET", "/docs").status_code == 404
    assert request(app, "GET", "/openapi.json").status_code == 404


def test_local_openapi_covers_every_versioned_endpoint() -> None:
    app = make_app(service=StubInferenceService(model=make_model()))

    schema = app.openapi()

    assert app.docs_url == "/docs"
    assert app.openapi_url == "/openapi.json"
    assert set(schema["paths"]) == {
        "/api/v1/health",
        "/api/v1/ready",
        "/api/v1/model",
        "/api/v1/predict",
        "/api/v1/predict/batch",
        "/api/v1/predictions/{prediction_id}",
        "/api/v1/predictions/{prediction_id}/review",
    }
    components = schema["components"]["schemas"]
    assert "decision" in components["PredictionResponse"]["properties"]
    assert "items" in components["BatchPredictionRequest"]["properties"]
    assert "results" in components["BatchPredictionResponse"]["properties"]
    assert "feature_snapshot" not in components["PredictionRecordResponse"]["properties"]
    assert "review_status" in components["ReviewUpdateRequest"]["properties"]
    assert {"index", "transaction_id", "status", "prediction", "error"} <= set(
        components["BatchPredictionItemResult"]["properties"]
    )


@pytest.mark.slow
def test_importing_the_api_module_opens_no_network_connection(tmp_path: Path) -> None:
    """Importing the app factory must neither dial out nor eagerly load a model."""
    script = (
        "import socket\n"
        "def refuse(*args, **kwargs):\n"
        "    raise AssertionError('network connection during module import')\n"
        "socket.socket.connect = refuse\n"
        "socket.socket.connect_ex = refuse\n"
        "import fraud_risk.api.main as api\n"
        "assert api.app.state.settings.app_env == 'local'\n"
        "print('ok')\n"
    )
    config_names = {name.upper() for name in Settings.model_fields}
    env = {key: value for key, value in os.environ.items() if key.upper() not in config_names}
    env["PYTHONPATH"] = str(REPO_ROOT / "src")

    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        cwd=tmp_path,
        env=env,
    )

    assert result.stdout.strip() == "ok"
