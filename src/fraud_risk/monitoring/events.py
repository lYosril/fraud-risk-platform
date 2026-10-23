"""Count-only recording of input-validation failures.

Responsibility
    Increment a persisted monitoring counter when the API rejects a malformed request. The
    recorder stores only the endpoint path, a safe reason code and a count. It never reads or
    writes the request body, headers, token or feature values.

Failure policy
    Monitoring must not turn a client's validation error into a 500. The recorder therefore
    swallows database failures after logging their class name. It does not retry: a missed
    monitoring event is preferable to delaying the response or duplicating a count.
"""

from __future__ import annotations

import logging

from sqlalchemy.exc import SQLAlchemyError

from fraud_risk.db.repository import PersistenceError
from fraud_risk.db.session import SessionFactory
from fraud_risk.monitoring.repository import MonitoringRepository

logger = logging.getLogger(__name__)


class MonitoringEventRecorder:
    """Persist count-only validation-failure events, or do nothing when disabled."""

    def __init__(self, session_factory: SessionFactory, *, enabled: bool) -> None:
        self._session_factory = session_factory
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        """Whether this recorder writes events."""
        return self._enabled

    def record_validation_failures(
        self,
        *,
        endpoint: str,
        reason_code: str,
        count: int = 1,
    ) -> None:
        """Record one aggregate event; never raise into the request path."""
        if not self._enabled or count <= 0:
            return
        safe_endpoint = endpoint.strip() or "unknown"
        safe_reason = reason_code.strip() or "validation_error"
        try:
            with self._session_factory() as session:
                MonitoringRepository(session).record_validation_failures(
                    endpoint=safe_endpoint,
                    reason_code=safe_reason,
                    count=count,
                )
        except (PersistenceError, SQLAlchemyError) as exc:
            logger.warning(
                "validation-failure monitoring write failed (%s)",
                type(exc).__name__,
            )


__all__ = ["MonitoringEventRecorder"]
