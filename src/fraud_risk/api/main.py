"""ASGI application factory.

Responsibility
    Build the FastAPI application, attach the process-wide settings and inference service,
    configure CORS from settings, register the versioned router and map inference failures to
    stable HTTP responses.

Design
    Construction performs no network I/O and loads no model. The inference service is lazy, so
    the process can start even when the artifact is missing; readiness then reports that fact
    accurately instead of failing startup.

Production
    Interactive docs and the OpenAPI document are disabled when ``app_env == "production"``;
    ``debug`` is always false. Scoring and health probes are public, while model metadata,
    prediction history and analyst review require ``API_ADMIN_TOKEN``. If that token is unset,
    the administrative routes fail closed with 503 rather than becoming anonymous.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from fraud_risk import __version__
from fraud_risk.api.dependencies import (
    INFERENCE_STATE_KEY,
    MONITORING_RECORDER_STATE_KEY,
    SETTINGS_STATE_KEY,
)
from fraud_risk.api.middleware import AccessLogMiddleware, RequestSizeLimitMiddleware
from fraud_risk.api.routes import router
from fraud_risk.api.validation import safe_validation_detail
from fraud_risk.config import Settings, get_settings
from fraud_risk.db.repository import PersistenceError
from fraud_risk.db.session import (
    DB_ENGINE_STATE_KEY,
    DB_SESSION_FACTORY_STATE_KEY,
    build_session_factory,
    create_db_engine,
)
from fraud_risk.models.inference import (
    FraudInferenceService,
    InferenceError,
    InferenceInputError,
    InferenceModelError,
)
from fraud_risk.monitoring.events import MonitoringEventRecorder

logger = logging.getLogger(__name__)

API_TITLE = "Fraud Detection & Risk Intelligence Platform API"
API_DESCRIPTION = (
    "Versioned HTTP contract for single and batch transaction scoring, model readiness, "
    "model metadata, prediction history, aggregate history counts, analyst review and the "
    "latest persisted drift report. Health, readiness and scoring are public; metadata, "
    "history, summary, review and monitoring operations require an administrative bearer "
    "token. Deploy behind TLS and bind to a private interface."
)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Dispose the process-wide engine when the application shuts down."""
    yield
    engine = getattr(app.state, DB_ENGINE_STATE_KEY, None)
    if engine is not None:
        engine.dispose()


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create a FastAPI application around the configured inference service.

    The service is constructed but not loaded: the first readiness check or prediction request
    performs artifact validation. This keeps startup independent of the artifact store while
    still making readiness truthful.
    """
    resolved = settings if settings is not None else get_settings()
    docs_enabled = resolved.app_env != "production"
    logging.getLogger("fraud_risk").setLevel(resolved.log_level)

    app = FastAPI(
        title=API_TITLE,
        description=API_DESCRIPTION,
        version=__version__,
        debug=False,
        docs_url="/docs" if docs_enabled else None,
        redoc_url=None,
        openapi_url="/openapi.json" if docs_enabled else None,
        lifespan=_lifespan,
    )

    db_engine = create_db_engine(resolved)
    session_factory = build_session_factory(db_engine)
    setattr(app.state, SETTINGS_STATE_KEY, resolved)
    setattr(app.state, INFERENCE_STATE_KEY, FraudInferenceService(settings=resolved))
    setattr(app.state, DB_ENGINE_STATE_KEY, db_engine)
    setattr(app.state, DB_SESSION_FACTORY_STATE_KEY, session_factory)
    setattr(
        app.state,
        MONITORING_RECORDER_STATE_KEY,
        MonitoringEventRecorder(
            session_factory,
            enabled=resolved.monitoring_capture_validation_failures,
        ),
    )

    # Added before CORS so CORSMiddleware stays outermost and can decorate a 413 response.
    app.add_middleware(
        RequestSizeLimitMiddleware,
        max_body_bytes=resolved.api_max_request_bytes,
    )

    origins = list(resolved.cors_origins)
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=False,
            allow_methods=["GET", "POST", "PATCH"],
            allow_headers=["Accept", "Content-Type", "Authorization"],
            max_age=600,
        )

    # Added last so it wraps CORS and the size limit: preflights and 413 responses are logged
    # too. It records no query string, headers, token or body.
    app.add_middleware(AccessLogMiddleware)

    async def _record_validation_failure(request: Request, reason_code: str) -> None:
        """Record a rejected request without reading its body or blocking the event loop."""
        recorder = getattr(request.app.state, MONITORING_RECORDER_STATE_KEY, None)
        if not getattr(recorder, "enabled", False):
            return
        record = getattr(recorder, "record_validation_failures", None)
        if not callable(record):
            return
        await run_in_threadpool(
            lambda: record(
                endpoint=request.url.path,
                reason_code=reason_code,
            )
        )

    @app.exception_handler(RequestValidationError)
    async def _handle_request_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Return a stable string instead of Pydantic's raw ``input`` echo."""
        await _record_validation_failure(request, "request_validation")
        return JSONResponse(
            status_code=422,
            content={"detail": safe_validation_detail(exc)},
        )

    @app.exception_handler(InferenceModelError)
    async def _handle_model_error(_request: Request, exc: InferenceModelError) -> JSONResponse:
        """Hide artifact paths and internals from the client while logging the cause."""
        logger.warning("inference model unavailable: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "model unavailable"},
        )

    @app.exception_handler(InferenceInputError)
    async def _handle_input_error(request: Request, exc: InferenceInputError) -> JSONResponse:
        """Return the safe validation message; it contains no full feature snapshot."""
        await _record_validation_failure(request, "inference_input")
        return JSONResponse(
            status_code=422,
            content={"detail": str(exc)},
        )

    @app.exception_handler(InferenceError)
    async def _handle_inference_error(_request: Request, exc: InferenceError) -> JSONResponse:
        """A requested capability (for example an explanation) cannot be served honestly."""
        logger.warning("inference request failed: %s", exc)
        return JSONResponse(
            status_code=422,
            content={"detail": str(exc)},
        )

    @app.exception_handler(PersistenceError)
    async def _handle_persistence_error(_request: Request, exc: PersistenceError) -> JSONResponse:
        """A failed read/write is not a successful prediction response."""
        logger.warning("prediction persistence unavailable: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "prediction storage unavailable"},
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected_error(_request: Request, exc: Exception) -> JSONResponse:
        """Return a stable 500 without a traceback, artifact path or payload in the response.

        Only the exception class reaches the log: third-party messages can contain SQL
        parameters or filesystem paths, which are neither needed for the client nor safe to
        assume are free of transaction data.
        """
        logger.error("unhandled API error (%s)", type(exc).__name__)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "internal server error"},
        )

    app.include_router(router, prefix=resolved.api_prefix)
    return app


app = create_app()


__all__ = ["API_DESCRIPTION", "API_TITLE", "app", "create_app"]
