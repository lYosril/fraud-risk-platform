"""Typed HTTP client used by the Streamlit dashboard.

Responsibility
    Translate the versioned REST API into small immutable records the dashboard can render.
    The client performs no scoring, plotting or Streamlit work, and it never imports model
    internals or database code.

Failure handling
    The dashboard must not turn an unreachable server or a malformed payload into plausible
    numbers. Every failure is raised as a :class:`DashboardApiError` subclass so the UI can
    show an explicit unavailable/error state instead of inventing data.

Secrets
    The administrative token is supplied at construction time, normally from
    ``Settings.api_admin_token``. It is only placed in the ``Authorization`` header and is
    never included in exceptions, response objects or ``repr`` output.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import Request, urlopen

from fraud_risk.config import Settings

#: The API caps a history page at this size; mirror it in the client.
MAX_HISTORY_LIMIT = 200
DEFAULT_TIMEOUT_SECONDS = 5.0
#: The only values an analyst can submit. ``pending`` is a state, never a manual update.
REVIEW_OUTCOMES = frozenset({"confirmed_fraud", "confirmed_legitimate"})
REVIEW_STATUSES = frozenset({"pending", *REVIEW_OUTCOMES})
_ALLOWED_DECISIONS = frozenset({"fraud", "legitimate"})


class DashboardApiError(RuntimeError):
    """The dashboard could not obtain a valid response from the API."""


class ApiUnavailableError(DashboardApiError):
    """The API could not be reached within the configured timeout."""


class ApiHTTPError(DashboardApiError):
    """The API returned an unsuccessful HTTP status with a safe detail message."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"API returned HTTP {status_code}: {detail}")


class ApiAuthenticationError(ApiHTTPError):
    """The administrative token is missing, rejected or the admin surface is disabled."""


class ApiValidationError(ApiHTTPError):
    """The API rejected the submitted transaction, query or review transition."""


class ApiConflictError(ApiHTTPError):
    """The requested analyst review transition conflicts with the stored workflow state."""


class ApiResponseError(DashboardApiError):
    """The API returned data that does not satisfy the documented contract."""


@dataclass(frozen=True, slots=True)
class HealthStatus:
    """Liveness metadata returned by ``GET /health``."""

    status: str
    service: str
    version: str
    environment: str


@dataclass(frozen=True, slots=True)
class ReadinessStatus:
    """Whether the API can currently load and serve its model."""

    status: str
    model_available: bool
    detail: str | None


@dataclass(frozen=True, slots=True)
class RiskBin:
    """One fixed-width bucket in the persisted risk-score distribution."""

    lower: float
    upper: float
    count: int


@dataclass(frozen=True, slots=True)
class OverviewSummary:
    """Aggregate counts over the persisted prediction history."""

    total_predictions: int
    fraud_predictions: int
    legitimate_predictions: int
    pending_reviews: int
    confirmed_fraud: int
    confirmed_legitimate: int
    risk_bins: tuple[RiskBin, ...]


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    """One non-sensitive persisted prediction row."""

    prediction_id: str
    transaction_id: str
    created_at: datetime
    model_version: str
    model_sha256: str
    risk_score: float
    threshold: float
    decision: str
    review_status: str


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    """The loaded model's serving contract, exposed by the admin route."""

    model_name: str
    model_version: str
    model_sha256: str
    model_family: str
    threshold: float
    feature_count: int
    feature_columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExplanationContribution:
    """One ranked SHAP contribution."""

    feature: str
    rank: int
    raw_value: float
    model_value: float
    contribution: float
    abs_contribution: float
    direction: str


@dataclass(frozen=True, slots=True)
class Explanation:
    """An optional top-k explanation returned with a prediction."""

    output_space: str
    explainer: str
    feature_perturbation: str
    aggregation: str
    n_folds: int
    top_k: int
    base_value: float
    output_value: float
    other_features_contribution: float
    additivity_residual: float
    contributions: tuple[ExplanationContribution, ...]
    limitations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Prediction:
    """A single transaction score and its optional explanation."""

    transaction_id: str
    probability: float
    threshold: float
    prediction: int
    decision: str
    model_version: str
    model_sha256: str
    model_family: str
    explanation: Explanation | None


class DashboardApiClient:
    """Small synchronous client for the dashboard's API calls.

    Args:
        base_url: API origin, for example ``http://localhost:8000``. The versioned prefix is
            deliberately included by each method so the client works with a non-default
            ``API_PREFIX`` only when that value is reflected in the configured URL.
        admin_token: Value of ``API_ADMIN_TOKEN`` for the protected routes. ``None`` means the
            overview, history and model metadata are unavailable.
        timeout: Socket timeout in seconds.
    """

    def __init__(
        self,
        base_url: str,
        admin_token: str | None = None,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        api_prefix: str = "/api/v1",
    ) -> None:
        resolved_url = base_url.strip().rstrip("/")
        parsed = urlsplit(resolved_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            msg = "base_url must be an absolute http(s) URL"
            raise ValueError(msg)
        if timeout <= 0:
            msg = "timeout must be positive"
            raise ValueError(msg)
        self._base_url = resolved_url
        self._admin_token = admin_token.strip() if admin_token and admin_token.strip() else None
        self._timeout = timeout
        self._api_prefix = api_prefix.rstrip("/")

    @classmethod
    def from_settings(cls, settings: Settings) -> DashboardApiClient:
        """Build a client from the single configuration entry point."""
        token = (
            settings.api_admin_token.get_secret_value()
            if settings.api_admin_token is not None
            else None
        )
        return cls(
            settings.dashboard_api_url,
            admin_token=token,
            api_prefix=settings.api_prefix,
        )

    @property
    def base_url(self) -> str:
        """The configured API origin, safe to display."""
        return self._base_url

    @property
    def admin_available(self) -> bool:
        """Whether the client can call the protected history/model routes."""
        return self._admin_token is not None

    def __repr__(self) -> str:
        return (
            f"DashboardApiClient(base_url={self._base_url!r}, "
            f"admin_available={self.admin_available})"
        )

    def health(self) -> HealthStatus:
        """Return process liveness."""
        _, body = self._request("GET", "/health")
        return _parse_health(body)

    def readiness(self) -> ReadinessStatus:
        """Return model readiness, treating the documented 503 as a valid status response."""
        _, body = self._request("GET", "/ready", accepted_statuses=frozenset({503}))
        return _parse_readiness(body)

    def model_metadata(self) -> ModelMetadata:
        """Return the loaded model's identity and feature contract (admin)."""
        _, body = self._request("GET", "/model", admin=True)
        return _parse_model_metadata(body)

    def prediction_summary(self) -> OverviewSummary:
        """Return aggregate counts over all persisted predictions (admin)."""
        _, body = self._request("GET", "/predictions/summary", admin=True)
        return _parse_summary(body)

    def recent_predictions(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
        min_risk_score: float | None = None,
        max_risk_score: float | None = None,
        decision: str | None = None,
        review_status: str | None = None,
    ) -> list[PredictionRecord]:
        """Return a filtered, bounded newest-first page of persisted history (admin)."""
        if not 1 <= limit <= MAX_HISTORY_LIMIT:
            msg = f"limit must be between 1 and {MAX_HISTORY_LIMIT}"
            raise ValueError(msg)
        if offset < 0:
            msg = "offset must not be negative"
            raise ValueError(msg)
        for name, value in (("created_from", created_from), ("created_to", created_to)):
            if value is not None and (value.tzinfo is None or value.utcoffset() is None):
                msg = f"{name} must include a timezone"
                raise ValueError(msg)
        if created_from is not None and created_to is not None and created_from > created_to:
            msg = "created_from must not be after created_to"
            raise ValueError(msg)

        resolved_min = _optional_range_score(min_risk_score, "min_risk_score")
        resolved_max = _optional_range_score(max_risk_score, "max_risk_score")
        if resolved_min is not None and resolved_max is not None and resolved_min > resolved_max:
            msg = "min_risk_score must not be greater than max_risk_score"
            raise ValueError(msg)
        if decision is not None and decision not in _ALLOWED_DECISIONS:
            msg = "decision must be 'fraud' or 'legitimate'"
            raise ValueError(msg)
        if review_status is not None and review_status not in REVIEW_STATUSES:
            msg = "review_status is not a recognised review state"
            raise ValueError(msg)

        params: dict[str, str | int | float] = {"limit": limit, "offset": offset}
        if created_from is not None:
            params["created_from"] = created_from.isoformat()
        if created_to is not None:
            params["created_to"] = created_to.isoformat()
        if resolved_min is not None:
            params["min_risk_score"] = resolved_min
        if resolved_max is not None:
            params["max_risk_score"] = resolved_max
        if decision is not None:
            params["decision"] = decision
        if review_status is not None:
            params["review_status"] = review_status

        _, body = self._request(
            "GET",
            f"/predictions?{urlencode(params)}",
            admin=True,
        )
        return _parse_history(body)

    def get_prediction(self, prediction_id: str) -> PredictionRecord:
        """Return one persisted prediction (admin)."""
        if not prediction_id.strip():
            msg = "prediction_id must not be blank"
            raise ValueError(msg)
        _, body = self._request(
            "GET",
            f"/predictions/{quote(prediction_id, safe='')}",
            admin=True,
        )
        return _parse_record(body)

    def review_prediction(
        self,
        prediction_id: str,
        *,
        review_status: str,
    ) -> PredictionRecord:
        """Persist one analyst review outcome (admin).

        Only the two terminal outcomes are accepted here. ``pending`` is not a manual review
        action; the API additionally enforces the transition against the stored row.
        """
        if not prediction_id.strip():
            msg = "prediction_id must not be blank"
            raise ValueError(msg)
        if review_status not in REVIEW_OUTCOMES:
            msg = "review_status must be 'confirmed_fraud' or 'confirmed_legitimate'"
            raise ValueError(msg)
        _, body = self._request(
            "PATCH",
            f"/predictions/{quote(prediction_id, safe='')}/review",
            payload={"review_status": review_status},
            admin=True,
        )
        return _parse_record(body)

    def predict(
        self,
        *,
        transaction_id: str,
        amount: float,
        features: Mapping[str, float],
        explain: bool = False,
    ) -> Prediction:
        """Score one transaction through the public prediction route."""
        if not transaction_id.strip():
            msg = "transaction_id must not be blank"
            raise ValueError(msg)
        if not features:
            msg = "at least one feature is required"
            raise ValueError(msg)

        resolved_amount = _input_number(amount, "amount")
        if resolved_amount < 0:
            msg = "amount must be non-negative"
            raise ValueError(msg)
        resolved_features: dict[str, float] = {}
        for name, value in features.items():
            if not isinstance(name, str) or not name:
                msg = "feature names must be non-empty strings"
                raise ValueError(msg)
            resolved_features[name] = _input_number(value, f"feature {name!r}")

        payload: dict[str, Any] = {
            "transaction_id": transaction_id,
            "amount": resolved_amount,
            "features": resolved_features,
            "explain": bool(explain),
        }
        _, body = self._request("POST", "/predict", payload=payload)
        return _parse_prediction(body)

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: Mapping[str, Any] | None = None,
        admin: bool = False,
        accepted_statuses: frozenset[int] = frozenset(),
    ) -> tuple[int, Mapping[str, Any]]:
        """Perform one JSON request and return ``(status, decoded_object)``."""
        headers = {"Accept": "application/json"}
        data: bytes | None = None
        if payload is not None:
            data = json.dumps(payload, allow_nan=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if admin:
            token = self._admin_token
            if token is None:
                msg = "API_ADMIN_TOKEN is not configured"
                raise ApiAuthenticationError(503, msg)
            headers["Authorization"] = f"Bearer {token}"

        request = Request(  # noqa: S310 - base_url is validated as http(s) above
            f"{self._base_url}{self._api_prefix}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=self._timeout) as response:  # noqa: S310
                status = int(response.status)
                raw = response.read()
        except HTTPError as exc:
            raw = exc.read()
            status = int(exc.code)
            if status not in accepted_statuses:
                raise _http_error(status, raw) from exc
        except (URLError, TimeoutError, OSError) as exc:
            msg = f"could not reach the API at {self._base_url}"
            raise ApiUnavailableError(msg) from exc

        body = _decode_object(raw)
        if status >= 400 and status not in accepted_statuses:
            raise _http_error(status, body)
        return status, body


def _http_error(status: int, raw: bytes | Mapping[str, Any]) -> ApiHTTPError:
    """Map an unsuccessful response to a safe, typed exception."""
    if isinstance(raw, bytes):
        try:
            body = _decode_object(raw)
        except ApiResponseError:
            body = {}
    else:
        body = raw
    detail_value = body.get("detail")
    detail = (
        detail_value if isinstance(detail_value, str) and detail_value.strip() else f"HTTP {status}"
    )
    if status in {401, 403}:
        return ApiAuthenticationError(status, detail)
    if status == 409:
        return ApiConflictError(status, detail)
    if status == 422:
        return ApiValidationError(status, detail)
    return ApiHTTPError(status, detail)


def _decode_object(raw: bytes) -> Mapping[str, Any]:
    """Decode a JSON object, rejecting HTML error pages and JSON arrays."""
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        msg = "the API returned a non-JSON response"
        raise ApiResponseError(msg) from exc
    if not isinstance(decoded, dict):
        msg = "the API response must be a JSON object"
        raise ApiResponseError(msg)
    return cast("Mapping[str, Any]", decoded)


def _parse_health(body: Mapping[str, Any]) -> HealthStatus:
    status = _require_str(body, "status")
    if status != "ok":
        msg = "health status must be 'ok'"
        raise ApiResponseError(msg)
    return HealthStatus(
        status=status,
        service=_require_str(body, "service"),
        version=_require_str(body, "version"),
        environment=_require_str(body, "environment"),
    )


def _parse_readiness(body: Mapping[str, Any]) -> ReadinessStatus:
    status = _require_str(body, "status")
    if status not in {"ready", "not_ready"}:
        msg = "readiness status must be 'ready' or 'not_ready'"
        raise ApiResponseError(msg)
    return ReadinessStatus(
        status=status,
        model_available=_require_bool(body, "model_available"),
        detail=_optional_str(body, "detail"),
    )


def _parse_model_metadata(body: Mapping[str, Any]) -> ModelMetadata:
    feature_columns = tuple(_require_str_list(body, "feature_columns"))
    feature_count = _require_int(body, "feature_count", minimum=1)
    if feature_count != len(feature_columns):
        msg = "feature_count does not match feature_columns"
        raise ApiResponseError(msg)
    return ModelMetadata(
        model_name=_require_str(body, "model_name"),
        model_version=_require_str(body, "model_version"),
        model_sha256=_require_str(body, "model_sha256"),
        model_family=_require_str(body, "model_family"),
        threshold=_require_float(body, "threshold", minimum=0.0, maximum=1.0),
        feature_count=feature_count,
        feature_columns=feature_columns,
    )


def _parse_summary(body: Mapping[str, Any]) -> OverviewSummary:
    return OverviewSummary(
        total_predictions=_require_int(body, "total_predictions", minimum=0),
        fraud_predictions=_require_int(body, "fraud_predictions", minimum=0),
        legitimate_predictions=_require_int(body, "legitimate_predictions", minimum=0),
        pending_reviews=_require_int(body, "pending_reviews", minimum=0),
        confirmed_fraud=_require_int(body, "confirmed_fraud", minimum=0),
        confirmed_legitimate=_require_int(body, "confirmed_legitimate", minimum=0),
        risk_bins=tuple(_parse_risk_bin(item) for item in _require_list(body, "risk_bins")),
    )


def _parse_risk_bin(value: Any) -> RiskBin:
    item = _require_mapping(value, "risk_bins item")
    lower = _require_float(item, "lower", minimum=0.0, maximum=1.0)
    upper = _require_float(item, "upper", minimum=0.0, maximum=1.0)
    if lower >= upper:
        msg = "a risk bin's lower bound must be below its upper bound"
        raise ApiResponseError(msg)
    return RiskBin(
        lower=lower,
        upper=upper,
        count=_require_int(item, "count", minimum=0),
    )


def _parse_history(body: Mapping[str, Any]) -> list[PredictionRecord]:
    items = _require_list(body, "items")
    return [_parse_record(_require_mapping(item, "history item")) for item in items]


def _parse_record(body: Mapping[str, Any]) -> PredictionRecord:
    decision = _require_str(body, "decision")
    if decision not in _ALLOWED_DECISIONS:
        msg = f"unexpected decision {decision!r}"
        raise ApiResponseError(msg)
    review_status = _require_str(body, "review_status")
    if review_status not in REVIEW_STATUSES:
        msg = f"unexpected review_status {review_status!r}"
        raise ApiResponseError(msg)
    return PredictionRecord(
        prediction_id=_require_str(body, "prediction_id"),
        transaction_id=_require_str(body, "transaction_id"),
        created_at=_require_datetime(body, "created_at"),
        model_version=_require_str(body, "model_version"),
        model_sha256=_require_str(body, "model_sha256"),
        risk_score=_require_float(body, "risk_score", minimum=0.0, maximum=1.0),
        threshold=_require_float(body, "threshold", minimum=0.0, maximum=1.0),
        decision=decision,
        review_status=review_status,
    )


def _parse_prediction(body: Mapping[str, Any]) -> Prediction:
    probability = _require_float(body, "probability", minimum=0.0, maximum=1.0)
    prediction = _require_int(body, "prediction", minimum=0, maximum=1)
    decision = _require_str(body, "decision")
    expected_decision = "fraud" if prediction == 1 else "legitimate"
    if decision != expected_decision:
        msg = "the response's decision does not match its binary prediction"
        raise ApiResponseError(msg)
    explanation_value = body.get("explanation")
    explanation = (
        None
        if explanation_value is None
        else _parse_explanation(_require_mapping(explanation_value, "explanation"))
    )
    return Prediction(
        transaction_id=_require_str(body, "transaction_id"),
        probability=probability,
        threshold=_require_float(body, "threshold", minimum=0.0, maximum=1.0),
        prediction=prediction,
        decision=decision,
        model_version=_require_str(body, "model_version"),
        model_sha256=_require_str(body, "model_sha256"),
        model_family=_require_str(body, "model_family"),
        explanation=explanation,
    )


def _parse_explanation(body: Mapping[str, Any]) -> Explanation:
    contributions = _require_list(body, "contributions")
    return Explanation(
        output_space=_require_str(body, "output_space"),
        explainer=_require_str(body, "explainer"),
        feature_perturbation=_require_str(body, "feature_perturbation"),
        aggregation=_require_str(body, "aggregation"),
        n_folds=_require_int(body, "n_folds", minimum=1),
        top_k=_require_int(body, "top_k", minimum=0),
        base_value=_require_float(body, "base_value"),
        output_value=_require_float(body, "output_value"),
        other_features_contribution=_require_float(body, "other_features_contribution"),
        additivity_residual=_require_float(body, "additivity_residual"),
        contributions=tuple(
            _parse_contribution(_require_mapping(item, "contribution")) for item in contributions
        ),
        limitations=tuple(_require_str_list(body, "limitations")),
    )


def _parse_contribution(body: Mapping[str, Any]) -> ExplanationContribution:
    return ExplanationContribution(
        feature=_require_str(body, "feature"),
        rank=_require_int(body, "rank", minimum=1),
        raw_value=_require_float(body, "raw_value"),
        model_value=_require_float(body, "model_value"),
        contribution=_require_float(body, "contribution"),
        abs_contribution=_require_float(body, "abs_contribution", minimum=0.0),
        direction=_require_str(body, "direction"),
    )


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        msg = f"{field} must be a JSON object"
        raise ApiResponseError(msg)
    return cast("Mapping[str, Any]", value)


def _require_list(body: Mapping[str, Any], field: str) -> list[Any]:
    value = body.get(field)
    if not isinstance(value, list):
        msg = f"{field} must be a JSON array"
        raise ApiResponseError(msg)
    return value


def _require_str(body: Mapping[str, Any], field: str) -> str:
    value = body.get(field)
    if not isinstance(value, str) or not value:
        msg = f"{field} must be a non-empty string"
        raise ApiResponseError(msg)
    return value


def _optional_str(body: Mapping[str, Any], field: str) -> str | None:
    value = body.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        msg = f"{field} must be a string or null"
        raise ApiResponseError(msg)
    return value


def _require_bool(body: Mapping[str, Any], field: str) -> bool:
    value = body.get(field)
    if not isinstance(value, bool):
        msg = f"{field} must be a boolean"
        raise ApiResponseError(msg)
    return value


def _require_int(
    body: Mapping[str, Any],
    field: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    value = body.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{field} must be an integer"
        raise ApiResponseError(msg)
    if minimum is not None and value < minimum:
        msg = f"{field} must be at least {minimum}"
        raise ApiResponseError(msg)
    if maximum is not None and value > maximum:
        msg = f"{field} must be at most {maximum}"
        raise ApiResponseError(msg)
    return value


def _require_float(
    body: Mapping[str, Any],
    field: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    return _finite_number(body.get(field), field, minimum=minimum, maximum=maximum)


def _finite_number(
    value: Any,
    field: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f"{field} must be a number"
        raise ApiResponseError(msg)
    number = float(value)
    if not math.isfinite(number):
        msg = f"{field} must be finite"
        raise ApiResponseError(msg)
    if minimum is not None and number < minimum:
        msg = f"{field} must be at least {minimum}"
        raise ApiResponseError(msg)
    if maximum is not None and number > maximum:
        msg = f"{field} must be at most {maximum}"
        raise ApiResponseError(msg)
    return number


def _input_number(value: Any, field: str) -> float:
    """Validate a locally supplied number, raising ``ValueError`` for bad form input."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f"{field} must be a number"
        raise ValueError(msg)
    number = float(value)
    if not math.isfinite(number):
        msg = f"{field} must be finite"
        raise ValueError(msg)
    return number


def _optional_range_score(value: float | None, field: str) -> float | None:
    """Validate an optional probability bound before it reaches the query string."""
    if value is None:
        return None
    number = _input_number(value, field)
    if not 0.0 <= number <= 1.0:
        msg = f"{field} must be between 0 and 1"
        raise ValueError(msg)
    return number


def _require_str_list(body: Mapping[str, Any], field: str) -> list[str]:
    values = _require_list(body, field)
    if any(not isinstance(value, str) or not value for value in values):
        msg = f"{field} must contain non-empty strings"
        raise ApiResponseError(msg)
    return cast("list[str]", values)


def _require_datetime(body: Mapping[str, Any], field: str) -> datetime:
    value = _require_str(body, field)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        msg = f"{field} must be an ISO-8601 datetime"
        raise ApiResponseError(msg) from exc
    if parsed.tzinfo is None:
        msg = f"{field} must include a timezone"
        raise ApiResponseError(msg)
    return parsed


__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "MAX_HISTORY_LIMIT",
    "REVIEW_OUTCOMES",
    "REVIEW_STATUSES",
    "ApiAuthenticationError",
    "ApiConflictError",
    "ApiHTTPError",
    "ApiResponseError",
    "ApiUnavailableError",
    "ApiValidationError",
    "DashboardApiClient",
    "DashboardApiError",
    "Explanation",
    "ExplanationContribution",
    "HealthStatus",
    "ModelMetadata",
    "OverviewSummary",
    "Prediction",
    "PredictionRecord",
    "ReadinessStatus",
    "RiskBin",
]
