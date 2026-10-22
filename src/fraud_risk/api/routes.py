"""HTTP route handlers for the versioned API.

Responsibility
    Translate validated HTTP requests into calls on the shared inference service and return the
    Pydantic response contracts. Handlers stay thin: feature validation, model loading and
    scoring belong to ``models.inference``; configuration belongs to ``config.Settings``.

Rules
    - Health checks liveness only. Readiness genuinely attempts to load/validate the model and
      returns 503 when it is unavailable.
    - Model loading and scoring run in Starlette's thread pool so the async event loop is never
      blocked by filesystem I/O, preprocessing or ``predict_proba``.
    - A batch item is validated independently; one bad transaction cannot discard the rest.
    - Successful predictions are persisted through the injected repository; a failed write
      rolls back and becomes a safe 503 rather than a false success.
    - Model metadata, prediction history and analyst review require the administrative bearer
      token. They fail closed with 503 when ``API_ADMIN_TOKEN`` is not configured.
    - No handler logs a request payload or feature snapshot.
    - Model and persistence failures are mapped to stable, non-leaking 503s by ``api.main``.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from fraud_risk import __version__
from fraud_risk.api.dependencies import (
    get_app_settings,
    get_inference_service,
    get_prediction_repository,
    require_admin_api_token,
)
from fraud_risk.api.schemas import (
    BatchPredictionItemResult,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse,
    HealthResponse,
    ModelMetadataResponse,
    PredictionHistoryResponse,
    PredictionRecordResponse,
    PredictionRequest,
    PredictionResponse,
    PredictionSummaryResponse,
    ReadinessResponse,
    ReviewUpdateRequest,
    RiskBinResponse,
)
from fraud_risk.api.validation import safe_validation_detail
from fraud_risk.config import Settings
from fraud_risk.db.models import PredictionDecision, ReviewStatus
from fraud_risk.db.repository import (
    InvalidReviewTransitionError,
    PredictionCreate,
    PredictionRepository,
)
from fraud_risk.models.inference import (
    FraudInferenceService,
    InferenceError,
    InferenceInputError,
    InferenceModelError,
)

logger = logging.getLogger(__name__)

router = APIRouter()

SettingsDep = Annotated[Settings, Depends(get_app_settings)]
InferenceServiceDep = Annotated[FraudInferenceService, Depends(get_inference_service)]
PredictionRepositoryDep = Annotated[PredictionRepository, Depends(get_prediction_repository)]


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Process liveness",
)
def health(settings: SettingsDep) -> HealthResponse:
    """Report that this process is running. This does not claim the model is loadable."""
    return HealthResponse(
        status="ok",
        service="fraud-risk-api",
        version=__version__,
        environment=settings.app_env,
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ReadinessResponse,
            "description": "The configured model artifact is missing, corrupt or incompatible",
        }
    },
    summary="Model readiness",
)
async def readiness(response: Response, service: InferenceServiceDep) -> ReadinessResponse:
    """Load and validate the model, reporting 503 if this process cannot score."""
    try:
        await run_in_threadpool(lambda: service.model)
    except InferenceModelError as exc:
        logger.warning("readiness check failed: %s", exc)
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadinessResponse(
            status="not_ready",
            model_available=False,
            detail="model unavailable",
        )
    return ReadinessResponse(
        status="ready",
        model_available=True,
    )


@router.get(
    "/model",
    response_model=ModelMetadataResponse,
    dependencies=[Depends(require_admin_api_token)],
    responses={
        status.HTTP_401_UNAUTHORIZED: {
            "model": ErrorResponse,
            "description": "Missing or invalid administrative bearer token",
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ErrorResponse,
            "description": (
                "The configured model artifact is unavailable or the admin API is disabled"
            ),
        },
    },
    summary="Selected model metadata (admin)",
)
async def model_metadata(
    settings: SettingsDep,
    service: InferenceServiceDep,
) -> ModelMetadataResponse:
    """Return the loaded model's identity and serving contract."""
    model = await run_in_threadpool(lambda: service.model)
    return ModelMetadataResponse(
        model_name=settings.model_name,
        model_version=model.model_version,
        model_sha256=model.sha256,
        model_family=model.family or "unknown",
        threshold=model.threshold,
        feature_count=len(model.feature_columns),
        feature_columns=list(model.feature_columns),
    )


@router.get(
    "/predictions/summary",
    response_model=PredictionSummaryResponse,
    dependencies=[Depends(require_admin_api_token)],
    responses={
        status.HTTP_401_UNAUTHORIZED: {
            "model": ErrorResponse,
            "description": "Missing or invalid administrative bearer token",
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ErrorResponse,
            "description": "Prediction storage is unavailable or the admin API is disabled",
        },
    },
    summary="Aggregate prediction history (admin)",
)
async def prediction_summary(
    repository: PredictionRepositoryDep,
) -> PredictionSummaryResponse:
    """Return bounded aggregate counts for the dashboard overview."""
    summary = await run_in_threadpool(repository.summarize)
    return PredictionSummaryResponse(
        total_predictions=summary.total_predictions,
        fraud_predictions=summary.fraud_predictions,
        legitimate_predictions=summary.legitimate_predictions,
        pending_reviews=summary.pending_reviews,
        confirmed_fraud=summary.confirmed_fraud,
        confirmed_legitimate=summary.confirmed_legitimate,
        risk_bins=[
            RiskBinResponse(lower=item.lower, upper=item.upper, count=item.count)
            for item in summary.risk_bins
        ],
    )


@router.get(
    "/predictions",
    response_model=PredictionHistoryResponse,
    dependencies=[Depends(require_admin_api_token)],
    responses={
        status.HTTP_401_UNAUTHORIZED: {
            "model": ErrorResponse,
            "description": "Missing or invalid administrative bearer token",
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ErrorResponse,
            "description": "Prediction storage is unavailable or the admin API is disabled",
        },
    },
    summary="List persisted prediction history (admin)",
)
async def list_predictions(
    repository: PredictionRepositoryDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
    created_from: Annotated[datetime | None, Query(description="Inclusive UTC start")] = None,
    created_to: Annotated[datetime | None, Query(description="Inclusive UTC end")] = None,
    min_risk_score: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
    max_risk_score: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
    decision: Annotated[PredictionDecision | None, Query()] = None,
    review_status: Annotated[ReviewStatus | None, Query()] = None,
) -> PredictionHistoryResponse:
    """Return filtered, bounded newest-first history; feature snapshots are never exposed."""
    _validate_history_filters(
        created_from=created_from,
        created_to=created_to,
        min_risk_score=min_risk_score,
        max_risk_score=max_risk_score,
    )
    records = await run_in_threadpool(
        repository.list_recent,
        limit=limit,
        offset=offset,
        created_from=created_from,
        created_to=created_to,
        min_risk_score=min_risk_score,
        max_risk_score=max_risk_score,
        decision=decision,
        review_status=review_status,
    )
    return PredictionHistoryResponse(
        items=[PredictionRecordResponse.model_validate(record) for record in records],
        limit=limit,
        offset=offset,
    )


@router.get(
    "/predictions/{prediction_id}",
    response_model=PredictionRecordResponse,
    dependencies=[Depends(require_admin_api_token)],
    responses={
        status.HTTP_401_UNAUTHORIZED: {
            "model": ErrorResponse,
            "description": "Missing or invalid administrative bearer token",
        },
        status.HTTP_404_NOT_FOUND: {
            "model": ErrorResponse,
            "description": "No prediction exists with this identifier",
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ErrorResponse,
            "description": "Prediction storage is unavailable or the admin API is disabled",
        },
    },
    summary="Read one persisted prediction (admin)",
)
async def get_prediction(
    prediction_id: UUID,
    repository: PredictionRepositoryDep,
) -> PredictionRecordResponse:
    """Return persisted metadata for one prediction; the feature snapshot is never exposed."""
    record = await run_in_threadpool(repository.get, prediction_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="prediction not found",
        )
    return PredictionRecordResponse.model_validate(record)


@router.patch(
    "/predictions/{prediction_id}/review",
    response_model=PredictionRecordResponse,
    dependencies=[Depends(require_admin_api_token)],
    responses={
        status.HTTP_401_UNAUTHORIZED: {
            "model": ErrorResponse,
            "description": "Missing or invalid administrative bearer token",
        },
        status.HTTP_404_NOT_FOUND: {
            "model": ErrorResponse,
            "description": "No prediction exists with this identifier",
        },
        422: {
            "model": ErrorResponse,
            "description": "The review status is not a permitted value or transition",
        },
        status.HTTP_409_CONFLICT: {
            "model": ErrorResponse,
            "description": "The prediction is already reviewed or the transition is not allowed",
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ErrorResponse,
            "description": "Prediction storage is unavailable or the admin API is disabled",
        },
    },
    summary="Update an analyst review status (admin)",
)
async def review_prediction(
    prediction_id: UUID,
    request: ReviewUpdateRequest,
    repository: PredictionRepositoryDep,
) -> PredictionRecordResponse:
    """Persist an analyst review transition and return the updated, non-sensitive record."""
    try:
        record = await run_in_threadpool(
            repository.set_review_status,
            prediction_id,
            request.review_status,
        )
    except InvalidReviewTransitionError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="prediction not found",
        )
    return PredictionRecordResponse.model_validate(record)


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    responses={
        413: {
            "model": ErrorResponse,
            "description": "The request body exceeds API_MAX_REQUEST_BYTES",
        },
        422: {
            "model": ErrorResponse,
            "description": "The transaction does not satisfy the model's feature contract",
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ErrorResponse,
            "description": "The model artifact or prediction storage is unavailable",
        },
    },
    summary="Score one transaction",
)
async def predict(
    request: PredictionRequest,
    settings: SettingsDep,
    service: InferenceServiceDep,
    repository: PredictionRepositoryDep,
) -> PredictionResponse:
    """Score one transaction, persist its record, then return the response."""
    result = await run_in_threadpool(
        service.predict,
        request.to_inference_request(),
        explain=request.explain,
    )
    response = PredictionResponse.from_result(result)
    await run_in_threadpool(
        repository.create_many,
        [_prediction_create(response, request, settings)],
    )
    return response


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    responses={
        413: {
            "model": ErrorResponse,
            "description": "The batch exceeds BATCH_MAX_SIZE or the request body limit",
        },
        422: {
            "model": ErrorResponse,
            "description": "The batch envelope is invalid, for example an empty items list",
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ErrorResponse,
            "description": "The model artifact or prediction storage is unavailable",
        },
    },
    summary="Score a bounded batch of transactions",
)
async def predict_batch(
    request: BatchPredictionRequest,
    settings: SettingsDep,
    service: InferenceServiceDep,
    repository: PredictionRepositoryDep,
) -> BatchPredictionResponse:
    """Score raw items independently, preserving input order and per-item errors."""
    if len(request.items) > settings.batch_max_size:
        raise HTTPException(
            status_code=413,
            detail=(
                f"batch contains {len(request.items)} items; "
                f"the maximum is {settings.batch_max_size}"
            ),
        )

    # Fail the whole request with a 503 if the model cannot be loaded; an input error in one
    # item must not be mistaken for model unavailability.
    model = await run_in_threadpool(lambda: service.model)

    results: list[BatchPredictionItemResult] = []
    rows_to_persist: list[PredictionCreate] = []
    for index, raw in enumerate(request.items):
        try:
            item = PredictionRequest.model_validate(raw)
        except ValidationError as exc:
            results.append(_batch_error(index, raw, safe_validation_detail(exc)))
            continue

        try:
            result = await run_in_threadpool(
                service.predict,
                item.to_inference_request(),
                explain=item.explain,
            )
        except InferenceModelError:
            raise
        except InferenceInputError as exc:
            results.append(_batch_error(index, raw, str(exc)))
        except InferenceError as exc:
            # For example, an explanation was requested for an unsupported model family.
            results.append(_batch_error(index, raw, str(exc)))
        else:
            prediction_response = PredictionResponse.from_result(result)
            results.append(
                BatchPredictionItemResult(
                    index=index,
                    transaction_id=item.transaction_id,
                    status="ok",
                    prediction=prediction_response,
                )
            )
            rows_to_persist.append(_prediction_create(prediction_response, item, settings))

    if rows_to_persist:
        await run_in_threadpool(repository.create_many, rows_to_persist)

    succeeded = sum(1 for result in results if result.status == "ok")
    return BatchPredictionResponse(
        model_version=model.model_version,
        model_sha256=model.sha256,
        model_family=model.family or "unknown",
        threshold=float(model.threshold),
        total=len(results),
        succeeded=succeeded,
        failed=len(results) - succeeded,
        results=results,
    )


def _validate_history_filters(
    *,
    created_from: datetime | None,
    created_to: datetime | None,
    min_risk_score: float | None,
    max_risk_score: float | None,
) -> None:
    """Reject ambiguous or empty filter ranges before the database query runs."""
    for name, value in (("created_from", created_from), ("created_to", created_to)):
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise HTTPException(
                status_code=422,
                detail=f"{name} must include a timezone",
            )
    if created_from is not None and created_to is not None and created_from > created_to:
        raise HTTPException(
            status_code=422,
            detail="created_from must not be after created_to",
        )
    if (
        min_risk_score is not None
        and max_risk_score is not None
        and min_risk_score > max_risk_score
    ):
        raise HTTPException(
            status_code=422,
            detail="min_risk_score must not be greater than max_risk_score",
        )


def _batch_error(index: int, raw: dict[str, Any], detail: str) -> BatchPredictionItemResult:
    """Build a safe per-item error, keeping a valid transaction id when one was supplied."""
    transaction_id = raw.get("transaction_id")
    safe_transaction_id = (
        transaction_id if isinstance(transaction_id, str) and transaction_id.strip() else None
    )
    return BatchPredictionItemResult(
        index=index,
        transaction_id=safe_transaction_id,
        status="error",
        error=detail,
    )


def _prediction_create(
    response: PredictionResponse,
    request: PredictionRequest,
    settings: Settings,
) -> PredictionCreate:
    """Build the persistence row, including an input snapshot only when explicitly enabled."""
    feature_snapshot = None
    if settings.persist_feature_snapshot:
        feature_snapshot = {
            "amount": request.amount,
            "features": dict(request.features),
        }
    return PredictionCreate(
        transaction_id=response.transaction_id,
        model_version=response.model_version,
        model_sha256=response.model_sha256,
        risk_score=response.probability,
        threshold=response.threshold,
        decision=PredictionDecision(response.decision),
        feature_snapshot=feature_snapshot,
    )


__all__ = ["router"]
