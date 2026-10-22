"""Persistence operations used by the prediction API.

Responsibility
    Provide the small write/read surface the API actually needs: create prediction rows in
    one transaction, read one by UUID, and update its analyst review status. A repository
    class is warranted here because these operations share a session and an explicit
    commit/rollback policy; it is not a generic DAO layer.

Rules
    - Commits happen here, after a complete unit of work. A database failure rolls back and
      raises :class:`PersistenceError`; the HTTP layer maps that to a safe 503.
    - Feature snapshots are optional and supplied only when the operator opted in.
    - This module never logs feature values or full transaction payloads.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from fraud_risk.db.models import Prediction, PredictionDecision, ReviewStatus

logger = logging.getLogger(__name__)


class PersistenceError(RuntimeError):
    """A database operation failed after the session was rolled back."""


@dataclass(frozen=True, slots=True)
class PredictionCreate:
    """The non-sensitive fields required to persist one prediction."""

    transaction_id: str
    model_version: str
    model_sha256: str
    risk_score: float
    threshold: float
    decision: PredictionDecision
    review_status: ReviewStatus = ReviewStatus.PENDING
    feature_snapshot: Mapping[str, Any] | None = field(default=None, repr=False)


class PredictionRepository:
    """Session-bound prediction persistence operations."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create_many(self, rows: Sequence[PredictionCreate]) -> list[Prediction]:
        """Insert rows in one transaction and return the persisted records."""
        if not rows:
            return []

        records = [
            Prediction(
                prediction_id=uuid4(),
                transaction_id=row.transaction_id,
                created_at=datetime.now(UTC),
                model_version=row.model_version,
                model_sha256=row.model_sha256,
                risk_score=float(row.risk_score),
                threshold=float(row.threshold),
                decision=PredictionDecision(row.decision).value,
                review_status=ReviewStatus(row.review_status).value,
                feature_snapshot=(
                    dict(row.feature_snapshot) if row.feature_snapshot is not None else None
                ),
            )
            for row in rows
        ]

        try:
            self._session.add_all(records)
            self._session.commit()
        except SQLAlchemyError as exc:
            self._session.rollback()
            # Log only the exception class: SQLAlchemy can include statement parameters in the
            # message, and an opt-in feature snapshot must never reach the logs.
            logger.warning("prediction persistence failed (%s)", type(exc).__name__)
            msg = "could not persist prediction records"
            raise PersistenceError(msg) from exc

        logger.info("persisted %d prediction record(s)", len(records))
        return records

    def get(self, prediction_id: UUID) -> Prediction | None:
        """Return one prediction by UUID, or ``None`` when it does not exist."""
        try:
            return self._session.get(Prediction, prediction_id)
        except SQLAlchemyError as exc:
            self._session.rollback()
            logger.warning("prediction read failed (%s)", type(exc).__name__)
            msg = "could not read prediction record"
            raise PersistenceError(msg) from exc

    def set_review_status(
        self,
        prediction_id: UUID,
        review_status: ReviewStatus,
    ) -> Prediction | None:
        """Update and commit an analyst review status, or return ``None`` if missing."""
        try:
            record = self._session.get(Prediction, prediction_id)
            if record is None:
                return None

            record.review_status = ReviewStatus(review_status).value
            self._session.commit()
        except SQLAlchemyError as exc:
            self._session.rollback()
            logger.warning("review-status update failed (%s)", type(exc).__name__)
            msg = "could not update prediction review status"
            raise PersistenceError(msg) from exc
        return record


__all__ = ["PersistenceError", "PredictionCreate", "PredictionRepository"]
