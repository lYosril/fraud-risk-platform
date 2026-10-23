"""PostgreSQL reads and writes for monitoring.

Responsibility
    Provide the small session-bound surface the scheduled monitor and the API need:

    - Aggregate exact window traffic, decision and review-state counts.
    - Load a bounded score/feature-snapshot sample for distribution statistics.
    - Sum count-only validation-failure events.
    - Persist and read one drift report.

Rules
    - No feature row or request payload is returned to the API.
    - Every write commits once; on failure the session is rolled back and a
      :class:`~fraud_risk.db.repository.PersistenceError` is raised.
    - Aggregate counts are separate from the bounded sample, so a truncated sample never
      misrepresents prediction volume or model-version share.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from datetime import datetime
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from fraud_risk.db.models import DriftReport, MonitoringEvent, Prediction
from fraud_risk.db.repository import PersistenceError
from fraud_risk.monitoring.report import (
    AggregateCounts,
    AnalysisSample,
    ModelVersionUsage,
    MonitoringReport,
    feature_rows_from_snapshots,
)

logger = logging.getLogger(__name__)

INPUT_VALIDATION_EVENT = "input_validation_failure"


class MonitoringRepository:
    """Session-bound monitoring persistence operations."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def record_validation_failures(
        self,
        *,
        endpoint: str,
        reason_code: str,
        count: int = 1,
    ) -> None:
        """Append one count-only validation-failure event."""
        if count < 1:
            msg = "failure count must be at least 1"
            raise ValueError(msg)
        if not endpoint.strip():
            msg = "endpoint must not be blank"
            raise ValueError(msg)
        if not reason_code.strip():
            msg = "reason_code must not be blank"
            raise ValueError(msg)
        record = MonitoringEvent(
            event_id=uuid4(),
            event_type=INPUT_VALIDATION_EVENT,
            endpoint=endpoint.strip()[:256],
            reason_code=reason_code.strip()[:64],
            failure_count=int(count),
        )
        try:
            self._session.add(record)
            self._session.commit()
        except SQLAlchemyError as exc:
            self._session.rollback()
            logger.warning("monitoring event write failed (%s)", type(exc).__name__)
            msg = "could not persist monitoring event"
            raise PersistenceError(msg) from exc

    def window_aggregate(self, *, start: datetime, end: datetime) -> AggregateCounts:
        """Return exact counts for the inclusive window."""
        _validate_window(start, end)
        try:
            usage_rows = self._session.execute(
                select(
                    Prediction.model_version,
                    Prediction.decision,
                    func.count(Prediction.prediction_id),
                )
                .where(Prediction.created_at >= start, Prediction.created_at <= end)
                .group_by(Prediction.model_version, Prediction.decision)
            ).all()
            review_rows = self._session.execute(
                select(Prediction.review_status, func.count(Prediction.prediction_id))
                .where(Prediction.created_at >= start, Prediction.created_at <= end)
                .group_by(Prediction.review_status)
            ).all()
            failure_count = self._session.scalar(
                select(func.coalesce(func.sum(MonitoringEvent.failure_count), 0)).where(
                    MonitoringEvent.event_type == INPUT_VALIDATION_EVENT,
                    MonitoringEvent.created_at >= start,
                    MonitoringEvent.created_at <= end,
                )
            )
        except SQLAlchemyError as exc:
            self._session.rollback()
            logger.warning("monitoring aggregate read failed (%s)", type(exc).__name__)
            msg = "could not read monitoring aggregates"
            raise PersistenceError(msg) from exc

        by_version: dict[str, dict[str, int]] = {}
        for model_version, decision, count in usage_rows:
            entry = by_version.setdefault(
                str(model_version),
                {"fraud": 0, "legitimate": 0},
            )
            entry[str(decision)] = int(count)
        model_versions = tuple(
            ModelVersionUsage(
                model_version=version,
                total=counts["fraud"] + counts["legitimate"],
                fraud=counts["fraud"],
                legitimate=counts["legitimate"],
            )
            for version, counts in sorted(by_version.items())
        )
        prediction_count = sum(item.total for item in model_versions)
        fraud_count = sum(item.fraud for item in model_versions)
        legitimate_count = sum(item.legitimate for item in model_versions)
        review_status_counts = {str(status): int(count) for status, count in review_rows}
        return AggregateCounts(
            prediction_count=prediction_count,
            fraud_count=fraud_count,
            legitimate_count=legitimate_count,
            model_versions=model_versions,
            review_status_counts=review_status_counts,
            validation_failure_count=int(failure_count or 0),
        )

    def analysis_sample(
        self,
        *,
        start: datetime,
        end: datetime,
        limit: int,
        feature_columns: Sequence[str],
    ) -> AnalysisSample:
        """Load a bounded newest-first sample of scores and optional feature snapshots."""
        if limit < 1:
            msg = "sample limit must be at least 1"
            raise ValueError(msg)
        _validate_window(start, end)
        try:
            rows = self._session.execute(
                select(Prediction.risk_score, Prediction.feature_snapshot)
                .where(Prediction.created_at >= start, Prediction.created_at <= end)
                .order_by(Prediction.created_at.desc(), Prediction.prediction_id.desc())
                .limit(limit)
            ).all()
        except SQLAlchemyError as exc:
            self._session.rollback()
            logger.warning("monitoring sample read failed (%s)", type(exc).__name__)
            msg = "could not read monitoring sample"
            raise PersistenceError(msg) from exc

        scores = tuple(float(score) for score, _snapshot in rows)
        snapshots: list[Mapping[str, object] | None] = [
            snapshot if isinstance(snapshot, Mapping) else None for _score, snapshot in rows
        ]
        feature_rows, snapshot_count = feature_rows_from_snapshots(snapshots, feature_columns)
        return AnalysisSample(
            scores=scores,
            feature_rows=feature_rows,
            snapshot_count=snapshot_count,
        )

    def save_report(self, report: MonitoringReport) -> DriftReport:
        """Persist one report and return the ORM row."""
        record = DriftReport(
            report_id=report.report_id,
            generated_at=report.generated_at,
            window_start=report.window_start,
            window_end=report.window_end,
            reference_id=report.reference_id,
            status=report.status.value,
            data_drift_status=report.data_drift_status.value,
            data_quality_status=report.data_quality_status.value,
            prediction_count=report.prediction_count,
            validation_failure_count=report.validation_failure_count,
            feature_snapshot_count=report.feature_snapshot_count,
            feature_drift_count=report.feature_drift_count,
            payload=report.payload,
        )
        try:
            self._session.add(record)
            self._session.commit()
        except SQLAlchemyError as exc:
            self._session.rollback()
            logger.warning("drift-report write failed (%s)", type(exc).__name__)
            msg = "could not persist drift report"
            raise PersistenceError(msg) from exc
        return record

    def latest_report(self) -> DriftReport | None:
        """Return the newest persisted report, or ``None`` when none exists."""
        statement = (
            select(DriftReport)
            .order_by(DriftReport.generated_at.desc(), DriftReport.report_id.desc())
            .limit(1)
        )
        try:
            return self._session.scalars(statement).first()
        except SQLAlchemyError as exc:
            self._session.rollback()
            logger.warning("drift-report read failed (%s)", type(exc).__name__)
            msg = "could not read drift report"
            raise PersistenceError(msg) from exc


def _validate_window(start: datetime, end: datetime) -> None:
    for name, value in (("start", start), ("end", end)):
        if value.tzinfo is None or value.utcoffset() is None:
            msg = f"{name} must include a timezone"
            raise ValueError(msg)
    if end < start:
        msg = "end must not be before start"
        raise ValueError(msg)


__all__ = [
    "INPUT_VALIDATION_EVENT",
    "MonitoringRepository",
]
