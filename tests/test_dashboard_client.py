"""Unit tests for the dashboard HTTP client.

No network or Streamlit server is involved: the module-level ``urlopen`` is replaced with a
small fake so the tests pin URL construction, bearer handling, response validation, timeouts
and every documented failure mode.
"""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime
from email.message import Message
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlsplit
from urllib.request import Request

import pytest

from fraud_risk.dashboard import api_client
from fraud_risk.dashboard.api_client import (
    ApiAuthenticationError,
    ApiConflictError,
    ApiHTTPError,
    ApiResponseError,
    ApiUnavailableError,
    DashboardApiClient,
)

BASE_URL = "http://api.example:8000"
ADMIN_TOKEN = "test-dashboard-admin-token"  # noqa: S105 - synthetic test credential
FEATURE_SHA = "a" * 64


class FakeResponse:
    """Minimal context-manager response returned by the fake opener."""

    def __init__(self, payload: Any = None, *, raw: bytes | None = None, status: int = 200):
        self._raw = raw if raw is not None else json.dumps(payload).encode("utf-8")
        self.status = status

    def read(self) -> bytes:
        return self._raw

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        return None


def patch_opener(
    monkeypatch: pytest.MonkeyPatch,
    *,
    response: FakeResponse | None = None,
    error: Exception | None = None,
) -> list[tuple[Request, float]]:
    calls: list[tuple[Request, float]] = []

    def fake_urlopen(request: Request, timeout: float) -> FakeResponse:
        calls.append((request, timeout))
        if error is not None:
            raise error
        if response is None:
            raise AssertionError("fake opener has neither a response nor an error")
        return response

    monkeypatch.setattr(api_client, "urlopen", fake_urlopen)
    return calls


def make_client(*, admin_token: str | None = None) -> DashboardApiClient:
    return DashboardApiClient(BASE_URL, admin_token=admin_token, timeout=2.5)


def model_payload() -> dict[str, Any]:
    return {
        "model_name": "fraud-risk-test",
        "model_version": "version-1",
        "model_sha256": FEATURE_SHA,
        "model_family": "xgboost",
        "threshold": 0.5,
        "feature_count": 3,
        "feature_columns": ["V1", "V2", "Amount"],
    }


def prediction_payload() -> dict[str, Any]:
    return {
        "transaction_id": "txn-1",
        "probability": 0.92,
        "threshold": 0.5,
        "prediction": 1,
        "decision": "fraud",
        "model_version": "version-1",
        "model_sha256": FEATURE_SHA,
        "model_family": "xgboost",
        "explanation": {
            "output_space": "tree_margin",
            "explainer": "shap.TreeExplainer",
            "feature_perturbation": "tree_path_dependent",
            "aggregation": "mean_absolute",
            "n_folds": 3,
            "top_k": 1,
            "base_value": 0.1,
            "output_value": 0.5,
            "other_features_contribution": 0.0,
            "additivity_residual": 0.0,
            "contributions": [
                {
                    "feature": "V1",
                    "rank": 1,
                    "raw_value": -1.2,
                    "model_value": -1.2,
                    "contribution": 0.4,
                    "abs_contribution": 0.4,
                    "direction": "increases fraud risk",
                }
            ],
            "limitations": ["Attribution is not causal evidence."],
        },
    }


def record_payload() -> dict[str, Any]:
    return {
        "prediction_id": "00000000-0000-0000-0000-000000000001",
        "transaction_id": "txn-stored",
        "created_at": "2026-01-02T12:00:00Z",
        "model_version": "version-1",
        "model_sha256": FEATURE_SHA,
        "risk_score": 0.75,
        "threshold": 0.5,
        "decision": "fraud",
        "review_status": "pending",
    }


def summary_payload() -> dict[str, Any]:
    return {
        "total_predictions": 3,
        "fraud_predictions": 2,
        "legitimate_predictions": 1,
        "pending_reviews": 1,
        "confirmed_fraud": 1,
        "confirmed_legitimate": 1,
        "risk_bins": [
            {"lower": index / 10, "upper": (index + 1) / 10, "count": 0} for index in range(10)
        ],
    }


# ---------------------------------------------------------------------------
# URLs, auth and transport failures
# ---------------------------------------------------------------------------
def test_health_uses_the_versioned_url_and_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "status": "ok",
        "service": "fraud-risk-api",
        "version": "0.1.0",
        "environment": "local",
    }
    calls = patch_opener(monkeypatch, response=FakeResponse(payload))

    health = make_client().health()

    assert health.service == "fraud-risk-api"
    assert calls[0][0].full_url == f"{BASE_URL}/api/v1/health"
    assert calls[0][0].get_method() == "GET"
    assert calls[0][1] == 2.5


def test_readiness_accepts_the_documented_503_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_opener(
        monkeypatch,
        response=FakeResponse(
            {"status": "not_ready", "model_available": False, "detail": "model unavailable"},
            status=503,
        ),
    )

    readiness = make_client().readiness()

    assert readiness.status == "not_ready"
    assert readiness.model_available is False
    assert readiness.detail == "model unavailable"


def test_admin_request_sends_the_bearer_token_without_leaking_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = patch_opener(monkeypatch, response=FakeResponse(model_payload()))
    client = make_client(admin_token=ADMIN_TOKEN)

    metadata = client.model_metadata()

    request = calls[0][0]
    assert request.get_header("Authorization") == f"Bearer {ADMIN_TOKEN}"
    assert metadata.feature_columns == ("V1", "V2", "Amount")
    assert ADMIN_TOKEN not in repr(client)


def test_admin_request_without_a_token_fails_before_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = patch_opener(monkeypatch, response=FakeResponse(model_payload()))

    with pytest.raises(ApiAuthenticationError, match="API_ADMIN_TOKEN"):
        make_client().model_metadata()

    assert calls == []


def test_unreachable_api_raises_a_typed_unavailable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_opener(monkeypatch, error=URLError("connection refused"))

    with pytest.raises(ApiUnavailableError, match="could not reach"):
        make_client().health()


def test_authentication_http_error_preserves_the_safe_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = HTTPError(
        f"{BASE_URL}/api/v1/model",
        401,
        "Unauthorized",
        hdrs=Message(),
        fp=io.BytesIO(b'{"detail": "invalid or missing credentials"}'),
    )
    patch_opener(monkeypatch, error=error)

    with pytest.raises(ApiAuthenticationError) as captured:
        make_client(admin_token=ADMIN_TOKEN).model_metadata()

    assert captured.value.status_code == 401
    assert captured.value.detail == "invalid or missing credentials"


def test_non_json_error_response_becomes_a_response_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = HTTPError(
        f"{BASE_URL}/api/v1/health",
        500,
        "Server Error",
        hdrs=Message(),
        fp=io.BytesIO(b"<html>private path C:/model</html>"),
    )
    patch_opener(monkeypatch, error=error)

    with pytest.raises(ApiHTTPError) as captured:
        make_client().health()

    assert captured.value.status_code == 500
    assert "private path" not in str(captured.value)


def test_malformed_success_body_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_opener(monkeypatch, response=FakeResponse(raw=b"<html>not json</html>"))

    with pytest.raises(ApiResponseError, match="non-JSON"):
        make_client().health()


# ---------------------------------------------------------------------------
# Payload validation
# ---------------------------------------------------------------------------
def test_prediction_payload_and_explanation_are_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_opener(monkeypatch, response=FakeResponse(prediction_payload()))

    result = make_client().predict(
        transaction_id="txn-1",
        amount=42.5,
        features={"V1": -1.2, "V2": 0.4},
        explain=True,
    )

    assert result.probability == 0.92
    assert result.threshold == 0.5
    assert result.decision == "fraud"
    assert result.explanation is not None
    assert result.explanation.contributions[0].feature == "V1"


def test_prediction_request_preserves_feature_order(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = patch_opener(monkeypatch, response=FakeResponse(prediction_payload()))

    make_client().predict(
        transaction_id="txn-1",
        amount=42.5,
        features={"V2": 0.4, "V1": -1.2},
        explain=False,
    )

    request = calls[0][0]
    assert isinstance(request.data, bytes)
    body = json.loads(request.data)
    assert list(body["features"]) == ["V2", "V1"]
    assert body["amount"] == 42.5
    assert body["explain"] is False


def test_prediction_rejects_a_probability_outside_the_unit_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = prediction_payload()
    payload["probability"] = 1.5
    patch_opener(monkeypatch, response=FakeResponse(payload))

    with pytest.raises(ApiResponseError, match="probability"):
        make_client().predict(transaction_id="txn-1", amount=1.0, features={"V1": 0.0})


def test_prediction_rejects_an_inconsistent_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = prediction_payload()
    payload["decision"] = "legitimate"
    patch_opener(monkeypatch, response=FakeResponse(payload))

    with pytest.raises(ApiResponseError, match="decision"):
        make_client().predict(transaction_id="txn-1", amount=1.0, features={"V1": 0.0})


def test_invalid_local_input_is_rejected_before_network(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = patch_opener(monkeypatch, response=FakeResponse(prediction_payload()))
    client = make_client()

    with pytest.raises(ValueError, match="non-negative"):
        client.predict(transaction_id="txn-1", amount=-1.0, features={"V1": 0.0})
    with pytest.raises(ValueError, match="finite"):
        client.predict(transaction_id="txn-1", amount=1.0, features={"V1": float("nan")})
    with pytest.raises(ValueError, match="transaction_id"):
        client.predict(transaction_id=" ", amount=1.0, features={"V1": 0.0})

    assert calls == []


def test_summary_and_history_are_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    client = make_client(admin_token=ADMIN_TOKEN)
    patch_opener(monkeypatch, response=FakeResponse(summary_payload()))
    summary = client.prediction_summary()
    assert summary.total_predictions == 3
    assert len(summary.risk_bins) == 10

    patch_opener(
        monkeypatch,
        response=FakeResponse(
            {"items": [record_payload()], "limit": 10, "offset": 5},
        ),
    )
    history = client.recent_predictions(limit=10, offset=5)
    assert len(history) == 1
    assert history[0].review_status == "pending"
    assert history[0].created_at.tzinfo is not None


def test_history_rejects_an_invalid_page_size(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = patch_opener(monkeypatch, response=FakeResponse({"items": [], "limit": 1}))

    with pytest.raises(ValueError, match="limit"):
        make_client(admin_token=ADMIN_TOKEN).recent_predictions(limit=0)

    assert calls == []


def test_history_filter_query_is_serialized(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = patch_opener(
        monkeypatch,
        response=FakeResponse({"items": [], "limit": 10, "offset": 0}),
    )

    make_client(admin_token=ADMIN_TOKEN).recent_predictions(
        limit=10,
        created_from=datetime(2026, 1, 1, tzinfo=UTC),
        created_to=datetime(2026, 1, 31, 23, 59, 59, tzinfo=UTC),
        min_risk_score=0.25,
        max_risk_score=0.75,
        decision="fraud",
        review_status="pending",
    )

    query = parse_qs(urlsplit(calls[0][0].full_url).query)
    assert query == {
        "limit": ["10"],
        "offset": ["0"],
        "created_from": ["2026-01-01T00:00:00+00:00"],
        "created_to": ["2026-01-31T23:59:59+00:00"],
        "min_risk_score": ["0.25"],
        "max_risk_score": ["0.75"],
        "decision": ["fraud"],
        "review_status": ["pending"],
    }
    assert calls[0][0].get_header("Authorization") == f"Bearer {ADMIN_TOKEN}"


def test_review_update_sends_a_patch_and_parses_the_terminal_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = record_payload()
    payload["review_status"] = "confirmed_fraud"
    calls = patch_opener(monkeypatch, response=FakeResponse(payload))
    prediction_id = "00000000-0000-0000-0000-000000000001"

    updated = make_client(admin_token=ADMIN_TOKEN).review_prediction(
        prediction_id,
        review_status="confirmed_fraud",
    )

    request = calls[0][0]
    assert updated.review_status == "confirmed_fraud"
    assert request.get_method() == "PATCH"
    assert request.full_url == (f"{BASE_URL}/api/v1/predictions/{prediction_id}/review")
    assert request.get_header("Authorization") == f"Bearer {ADMIN_TOKEN}"
    assert isinstance(request.data, bytes)
    assert json.loads(request.data) == {"review_status": "confirmed_fraud"}


def test_review_conflict_maps_to_a_typed_conflict_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = HTTPError(
        f"{BASE_URL}/api/v1/predictions/00000000-0000-0000-0000-000000000001/review",
        409,
        "Conflict",
        hdrs=Message(),
        fp=io.BytesIO(
            b'{"detail": "review transition from confirmed_fraud to '
            b'confirmed_legitimate is not allowed"}'
        ),
    )
    patch_opener(monkeypatch, error=error)

    with pytest.raises(ApiConflictError) as captured:
        make_client(admin_token=ADMIN_TOKEN).review_prediction(
            "00000000-0000-0000-0000-000000000001",
            review_status="confirmed_legitimate",
        )

    assert captured.value.status_code == 409
    assert "not allowed" in captured.value.detail


def test_invalid_history_filters_are_rejected_before_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = patch_opener(
        monkeypatch,
        response=FakeResponse({"items": [], "limit": 10, "offset": 0}),
    )
    client = make_client(admin_token=ADMIN_TOKEN)

    with pytest.raises(ValueError, match="timezone"):
        client.recent_predictions(
            created_from=datetime(2026, 1, 1, tzinfo=UTC).replace(tzinfo=None)
        )
    with pytest.raises(ValueError, match="greater"):
        client.recent_predictions(min_risk_score=0.8, max_risk_score=0.2)
    with pytest.raises(ValueError, match="confirmed_fraud"):
        client.review_prediction("prediction-1", review_status="pending")

    assert calls == []


def test_summary_rejects_a_missing_field(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = summary_payload()
    del payload["fraud_predictions"]
    patch_opener(monkeypatch, response=FakeResponse(payload))

    with pytest.raises(ApiResponseError, match="fraud_predictions"):
        make_client(admin_token=ADMIN_TOKEN).prediction_summary()
