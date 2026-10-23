"""FastAPI dependency providers.

Responsibility
    Expose the process-wide settings and inference service to route handlers without letting a
    route construct either object. Tests replace these providers with
    ``app.dependency_overrides`` instead of touching application internals.

Rules
    The providers only read objects attached to ``app.state`` by the application factory. They
    perform no I/O, so importing or resolving them cannot load a model or open a connection.
"""

from __future__ import annotations

import secrets
from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from fraud_risk.config import Settings
from fraud_risk.db.repository import PredictionRepository
from fraud_risk.db.session import DB_SESSION_FACTORY_STATE_KEY
from fraud_risk.models.inference import FraudInferenceService
from fraud_risk.monitoring.events import MonitoringEventRecorder
from fraud_risk.monitoring.performance_repository import PerformanceRepository
from fraud_risk.monitoring.repository import MonitoringRepository

SETTINGS_STATE_KEY = "settings"
INFERENCE_STATE_KEY = "inference_service"
MONITORING_RECORDER_STATE_KEY = "monitoring_recorder"

# `auto_error=False` lets us return one stable 401 shape for both a missing header and a
# malformed scheme, instead of FastAPI's default 403 for a missing bearer token.
_bearer_scheme = HTTPBearer(auto_error=False, scheme_name="Admin bearer token")


def get_app_settings(request: Request) -> Settings:
    """Return the settings object attached to this application instance."""
    settings = getattr(request.app.state, SETTINGS_STATE_KEY, None)
    if not isinstance(settings, Settings):
        msg = "application settings were not configured by the app factory"
        raise RuntimeError(msg)
    return settings


def get_inference_service(request: Request) -> FraudInferenceService:
    """Return the shared inference service attached to this application instance."""
    service = getattr(request.app.state, INFERENCE_STATE_KEY, None)
    if not isinstance(service, FraudInferenceService):
        msg = "an inference service was not configured by the app factory"
        raise RuntimeError(msg)
    return service


def get_db_session(request: Request) -> Iterator[Session]:
    """Yield one session per request, rolling back on failure and always closing it."""
    factory = getattr(request.app.state, DB_SESSION_FACTORY_STATE_KEY, None)
    if factory is None:
        msg = "a database session factory was not configured by the app factory"
        raise RuntimeError(msg)

    session: Session = factory()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_prediction_repository(
    session: Annotated[Session, Depends(get_db_session)],
) -> PredictionRepository:
    """Return a session-bound repository; tests override this provider directly."""
    return PredictionRepository(session)


def get_monitoring_repository(
    session: Annotated[Session, Depends(get_db_session)],
) -> MonitoringRepository:
    """Return a session-bound monitoring repository."""
    return MonitoringRepository(session)


def get_performance_repository(
    session: Annotated[Session, Depends(get_db_session)],
) -> PerformanceRepository:
    """Return a session-bound delayed-label performance repository."""
    return PerformanceRepository(session)


def get_monitoring_recorder(request: Request) -> MonitoringEventRecorder:
    """Return the process-wide validation-failure recorder."""
    recorder = getattr(request.app.state, MONITORING_RECORDER_STATE_KEY, None)
    if not isinstance(recorder, MonitoringEventRecorder):
        msg = "a monitoring event recorder was not configured by the app factory"
        raise RuntimeError(msg)
    return recorder


def require_admin_api_token(
    settings: Annotated[Settings, Depends(get_app_settings)],
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)],
) -> None:
    """Fail closed unless the request carries the configured administrative bearer token.

    There is deliberately no anonymous fallback: if ``API_ADMIN_TOKEN`` is unset, protected
    routes answer 503 rather than becoming public. Tokens are compared in constant time and
    never logged.
    """
    configured = (
        settings.api_admin_token.get_secret_value()
        if settings.api_admin_token is not None
        else None
    )
    if configured is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="administrative API is not configured",
        )

    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="invalid or missing credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise unauthorized
    if not secrets.compare_digest(credentials.credentials, configured):
        raise unauthorized


__all__ = [
    "INFERENCE_STATE_KEY",
    "MONITORING_RECORDER_STATE_KEY",
    "SETTINGS_STATE_KEY",
    "get_app_settings",
    "get_db_session",
    "get_inference_service",
    "get_monitoring_recorder",
    "get_monitoring_repository",
    "get_performance_repository",
    "get_prediction_repository",
    "require_admin_api_token",
]
