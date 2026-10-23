"""PostgreSQL reads and writes for delayed-label performance reports.

Responsibility
    Persist one aggregate ``PerformanceEvaluationReport`` and return the newest report to the
    admin API. The repository never reads ``predictions`` or analyst review state: the labels
    and scores come from an explicit, separately supplied evaluation extract.

Rules
    Every write commits once. On a database failure the session is rolled back and a
    :class:`~fraud_risk.db.repository.PersistenceError` is raised. Only aggregate report
    payloads are stored; no feature row, identifier or raw label is written here.
"""

from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from fraud_risk.db.models import PerformanceReport
from fraud_risk.db.repository import PersistenceError
from fraud_risk.monitoring.performance import PerformanceEvaluationReport

logger = logging.getLogger(__name__)


class PerformanceRepository:
    """Session-bound persistence for aggregate performance reports."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def save_report(self, report: PerformanceEvaluationReport) -> PerformanceReport:
        """Persist one report and return the ORM row."""
        record = PerformanceReport(
            report_id=report.report_id,
            generated_at=report.generated_at,
            window_start=report.window_start,
            window_end=report.window_end,
            row_count=report.row_count,
            model_version_count=report.model_version_count,
            segment_count=report.segment_count,
            threshold=report.threshold,
            period=report.period,
            labels_verified=report.labels_verified,
            label_source=report.label_source,
            payload=report.payload,
        )
        try:
            self._session.add(record)
            self._session.commit()
        except SQLAlchemyError as exc:
            self._session.rollback()
            logger.warning("performance-report write failed (%s)", type(exc).__name__)
            msg = "could not persist performance report"
            raise PersistenceError(msg) from exc
        return record

    def latest_report(self) -> PerformanceReport | None:
        """Return the newest persisted report, or ``None`` when none exists."""
        statement = (
            select(PerformanceReport)
            .order_by(PerformanceReport.generated_at.desc(), PerformanceReport.report_id.desc())
            .limit(1)
        )
        try:
            return self._session.scalars(statement).first()
        except SQLAlchemyError as exc:
            self._session.rollback()
            logger.warning("performance-report read failed (%s)", type(exc).__name__)
            msg = "could not read performance report"
            raise PersistenceError(msg) from exc


__all__ = ["PerformanceRepository"]
