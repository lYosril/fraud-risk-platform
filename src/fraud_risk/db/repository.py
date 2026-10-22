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

from sqlalchemy import ColumnElement, and_, case, func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from fraud_risk.db.models import Prediction, PredictionDecision, ReviewStatus

logger = logging.getLogger(__name__)

#: Fixed-width score bands used by the dashboard overview. They are part of the API contract,
#: not a chart-only convenience, so every consumer sees the same buckets.
RISK_BIN_COUNT = 10


class PersistenceError(RuntimeError):
    """A database operation failed after the session was rolled back."""


class InvalidReviewTransitionError(ValueError):
    """An analyst review update would leave the permitted workflow."""


#: A review starts pending and ends in exactly one terminal decision. Model predictions are
#: deliberately absent from this table: a score never advances the human review state.
_ALLOWED_REVIEW_TARGETS: Mapping[ReviewStatus, frozenset[ReviewStatus]] = {
    ReviewStatus.PENDING: frozenset(
        {ReviewStatus.CONFIRMED_FRAUD, ReviewStatus.CONFIRMED_LEGITIMATE}
    ),
    ReviewStatus.CONFIRMED_FRAUD: frozenset(),
    ReviewStatus.CONFIRMED_LEGITIMATE: frozenset(),
}


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


@dataclass(frozen=True, slots=True)
class RiskBin:
    """The number of persisted predictions in one half-open risk-score band."""

    lower: float
    upper: float
    count: int


@dataclass(frozen=True, slots=True)
class PredictionSummary:
    """Aggregate history counts, without returning any transaction features."""

    total_predictions: int
    fraud_predictions: int
    legitimate_predictions: int
    pending_reviews: int
    confirmed_fraud: int
    confirmed_legitimate: int
    risk_bins: tuple[RiskBin, ...]


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

    def list_recent(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
        min_risk_score: float | None = None,
        max_risk_score: float | None = None,
        decision: PredictionDecision | None = None,
        review_status: ReviewStatus | None = None,
    ) -> list[Prediction]:
        """Return a filtered, newest-first prediction history page.

        Date bounds are inclusive. The caller (the API route) validates timezone
        awareness and range ordering; this method applies only the conditions it is given.
        """
        if limit < 1:
            msg = "limit must be at least 1"
            raise ValueError(msg)
        if offset < 0:
            msg = "offset must not be negative"
            raise ValueError(msg)

        conditions: list[ColumnElement[bool]] = []
        if created_from is not None:
            conditions.append(Prediction.created_at >= created_from)
        if created_to is not None:
            conditions.append(Prediction.created_at <= created_to)
        if min_risk_score is not None:
            conditions.append(Prediction.risk_score >= min_risk_score)
        if max_risk_score is not None:
            conditions.append(Prediction.risk_score <= max_risk_score)
        if decision is not None:
            conditions.append(Prediction.decision == PredictionDecision(decision).value)
        if review_status is not None:
            conditions.append(Prediction.review_status == ReviewStatus(review_status).value)

        statement = (
            select(Prediction)
            .where(*conditions)
            .order_by(Prediction.created_at.desc(), Prediction.prediction_id.desc())
            .offset(offset)
            .limit(limit)
        )
        try:
            return list(self._session.scalars(statement).all())
        except SQLAlchemyError as exc:
            self._session.rollback()
            logger.warning("prediction list failed (%s)", type(exc).__name__)
            msg = "could not read prediction history"
            raise PersistenceError(msg) from exc

    def summarize(self) -> PredictionSummary:
        """Return aggregate counts over every persisted prediction.

        The dashboard needs totals that do not depend on a paginated list. These are database
        aggregates, so the response stays bounded when the history grows. No feature snapshot
        participates in the query.
        """
        summary_columns: list[Any] = [
            func.count(Prediction.prediction_id).label("total_predictions"),
            _count_if(Prediction.decision == PredictionDecision.FRAUD.value).label(
                "fraud_predictions"
            ),
            _count_if(Prediction.decision == PredictionDecision.LEGITIMATE.value).label(
                "legitimate_predictions"
            ),
            _count_if(Prediction.review_status == ReviewStatus.PENDING.value).label(
                "pending_reviews"
            ),
            _count_if(Prediction.review_status == ReviewStatus.CONFIRMED_FRAUD.value).label(
                "confirmed_fraud"
            ),
            _count_if(Prediction.review_status == ReviewStatus.CONFIRMED_LEGITIMATE.value).label(
                "confirmed_legitimate"
            ),
        ]
        lower_bounds: list[float] = []
        upper_bounds: list[float] = []
        for index in range(RISK_BIN_COUNT):
            lower = index / RISK_BIN_COUNT
            upper = (index + 1) / RISK_BIN_COUNT
            condition = Prediction.risk_score >= lower
            condition = (
                and_(condition, Prediction.risk_score < upper)
                if index < RISK_BIN_COUNT - 1
                else and_(condition, Prediction.risk_score <= upper)
            )
            summary_columns.append(_count_if(condition).label(f"risk_bin_{index}"))
            lower_bounds.append(lower)
            upper_bounds.append(upper)

        try:
            row = self._session.execute(select(*summary_columns)).mappings().one()
        except SQLAlchemyError as exc:
            self._session.rollback()
            logger.warning("prediction summary failed (%s)", type(exc).__name__)
            msg = "could not summarize prediction history"
            raise PersistenceError(msg) from exc

        def count(label: str) -> int:
            return int(row[label] or 0)

        return PredictionSummary(
            total_predictions=count("total_predictions"),
            fraud_predictions=count("fraud_predictions"),
            legitimate_predictions=count("legitimate_predictions"),
            pending_reviews=count("pending_reviews"),
            confirmed_fraud=count("confirmed_fraud"),
            confirmed_legitimate=count("confirmed_legitimate"),
            risk_bins=tuple(
                RiskBin(
                    lower=lower,
                    upper=upper,
                    count=count(f"risk_bin_{index}"),
                )
                for index, (lower, upper) in enumerate(zip(lower_bounds, upper_bounds, strict=True))
            ),
        )

    def set_review_status(
        self,
        prediction_id: UUID,
        review_status: ReviewStatus,
    ) -> Prediction | None:
        """Persist one allowed analyst review transition, or return ``None`` if missing.

        A model prediction never advances this state. Only a pending row can be reviewed, and
        a reviewed row is terminal, so a second analyst cannot silently overwrite the first
        outcome. Invalid transitions raise :class:`InvalidReviewTransitionError` without a commit.
        """
        try:
            record = self._session.get(Prediction, prediction_id)
            if record is None:
                return None

            target = ReviewStatus(review_status)
            _validate_review_transition(record.review_status, target)
            record.review_status = target.value
            self._session.commit()
        except InvalidReviewTransitionError:
            raise
        except SQLAlchemyError as exc:
            self._session.rollback()
            logger.warning("review-status update failed (%s)", type(exc).__name__)
            msg = "could not update prediction review status"
            raise PersistenceError(msg) from exc
        return record


def _validate_review_transition(current_value: str, target: ReviewStatus) -> None:
    """Reject terminal-state overwrites and transitions back to pending."""
    try:
        current = ReviewStatus(current_value)
    except ValueError as exc:
        msg = "stored review status is not recognised"
        raise InvalidReviewTransitionError(msg) from exc

    if target not in _ALLOWED_REVIEW_TARGETS[current]:
        msg = f"review transition from {current.value} to {target.value} is not allowed"
        raise InvalidReviewTransitionError(msg)


def _count_if(condition: ColumnElement[bool]) -> Any:
    """Build a database-portable conditional count aggregate."""
    return func.coalesce(func.sum(case((condition, 1), else_=0)), 0)


__all__ = [
    "RISK_BIN_COUNT",
    "InvalidReviewTransitionError",
    "PersistenceError",
    "PredictionCreate",
    "PredictionRepository",
    "PredictionSummary",
    "RiskBin",
]
