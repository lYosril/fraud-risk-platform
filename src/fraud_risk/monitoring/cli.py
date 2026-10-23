"""Command-line entry point for monitoring and delayed-label performance reports.

Examples
    ``python -m fraud_risk.monitoring run --window-hours 24``
    ``python -m fraud_risk.monitoring performance --path labeled.csv --threshold 0.5``

The reference profile is built by ``python -m fraud_risk.models.reference``. This module only
reads the persisted profile and stored prediction window, so it never imports the offline model
bundle and stays below the serving/model boundary in the package graph. The performance
subcommand reads an explicit, separately supplied scored evaluation extract; it never reads
the ``predictions`` table or analyst review state.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import Engine
from sqlalchemy.exc import SQLAlchemyError

from fraud_risk.config import PerformancePeriod, Settings, get_settings
from fraud_risk.data.errors import DataError
from fraud_risk.db.repository import PersistenceError
from fraud_risk.db.session import build_session_factory, create_db_engine
from fraud_risk.monitoring.performance import (
    PERFORMANCE_PERIODS,
    PerformanceReportError,
    build_performance_report,
    load_performance_dataset,
    write_performance_report,
)
from fraud_risk.monitoring.performance_repository import PerformanceRepository
from fraud_risk.monitoring.reference import (
    ReferenceProfileError,
    load_reference_profile,
)
from fraud_risk.monitoring.report import (
    MonitoringReportError,
    MonitoringThresholds,
    build_monitoring_report,
    write_report,
)
from fraud_risk.monitoring.repository import MonitoringRepository

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_INVALID = 1


def build_parser() -> argparse.ArgumentParser:
    """Build the monitoring CLI parser."""
    parser = argparse.ArgumentParser(
        prog="python -m fraud_risk.monitoring",
        description=(
            "Generate aggregate drift or delayed-label performance reports. No command writes "
            "raw transaction rows."
        ),
        epilog=(
            "Build the drift reference first with python -m fraud_risk.models.reference. "
            "Performance reports require an explicit scored, labeled evaluation extract."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="build and persist one monitoring report")
    run.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="reference profile path (default: Settings.monitoring_reference_path)",
    )
    run.add_argument(
        "--window-hours",
        type=int,
        default=None,
        help="window length in hours (default: Settings.monitoring_window_hours)",
    )
    run.add_argument(
        "--window-end",
        default=None,
        help="inclusive ISO-8601 UTC window end (default: now)",
    )
    run.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="JSON report directory (default: Settings.monitoring_report_dir)",
    )
    run.add_argument(
        "--no-persist",
        action="store_true",
        help="write the JSON report but do not insert a drift_reports row",
    )

    performance = subparsers.add_parser(
        "performance",
        help="build and persist one delayed-label performance report",
    )
    performance.add_argument(
        "--path",
        type=Path,
        required=True,
        help="scored, labeled evaluation CSV",
    )
    performance.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="evaluation threshold in [0, 1]; recorded with the report",
    )
    performance.add_argument(
        "--label-source",
        required=True,
        help="description of the verified outcome source (for example verified_chargebacks)",
    )
    performance.add_argument(
        "--labels-verified",
        action="store_true",
        help="assert that is_fraud is a confirmed outcome, not analyst review state",
    )
    performance.add_argument(
        "--period",
        choices=PERFORMANCE_PERIODS,
        default=None,
        help="evaluation period grouping (default: Settings.performance_period)",
    )
    performance.add_argument(
        "--window-start",
        default=None,
        help="optional inclusive ISO-8601 UTC start of the evaluation window",
    )
    performance.add_argument(
        "--window-end",
        default=None,
        help="optional inclusive ISO-8601 UTC end of the evaluation window",
    )
    performance.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="JSON report directory (default: Settings.performance_report_dir)",
    )
    performance.add_argument(
        "--no-persist",
        action="store_true",
        help="write the JSON report but do not insert a performance_reports row",
    )
    return parser


def main(argv: Sequence[str] | None = None, *, settings: Settings | None = None) -> int:
    """Run one monitoring command and return a process exit code."""
    options = build_parser().parse_args(argv)
    resolved_settings = settings or get_settings()
    logging.basicConfig(level=resolved_settings.log_level)
    if options.command == "performance":
        return _run_performance(resolved_settings, options)
    return _run_monitor(resolved_settings, options)


def _run_monitor(settings: Settings, options: argparse.Namespace) -> int:
    reference_path = Path(options.reference or settings.monitoring_reference_path)
    report_dir = Path(options.report_dir or settings.monitoring_report_dir)
    window_hours = (
        settings.monitoring_window_hours
        if options.window_hours is None
        else int(options.window_hours)
    )
    if window_hours < 1:
        logger.error("--window-hours must be at least 1")
        return EXIT_INVALID
    window_end = _parse_window_end(options.window_end)
    if window_end is None:
        logger.error("--window-end must be a timezone-aware ISO-8601 datetime")
        return EXIT_INVALID
    window_start = window_end - timedelta(hours=window_hours)

    try:
        reference = load_reference_profile(reference_path)
    except ReferenceProfileError as exc:
        logger.error("could not load reference profile: %s", exc)
        return EXIT_INVALID

    engine = create_db_engine(settings)
    try:
        session_factory = build_session_factory(engine)
        with session_factory() as session:
            repository = MonitoringRepository(session)
            aggregate = repository.window_aggregate(start=window_start, end=window_end)
            sample = repository.analysis_sample(
                start=window_start,
                end=window_end,
                limit=settings.monitoring_sample_limit,
                feature_columns=[item.name for item in reference.features],
            )
            report = build_monitoring_report(
                reference=reference,
                aggregate=aggregate,
                sample=sample,
                thresholds=MonitoringThresholds.from_settings(settings),
                window_start=window_start,
                window_end=window_end,
            )
            if not options.no_persist:
                repository.save_report(report)
    except (
        PersistenceError,
        MonitoringReportError,
        ReferenceProfileError,
        SQLAlchemyError,
        OSError,
        ValueError,
    ) as exc:
        logger.exception("monitoring run failed: %s", exc)
        return EXIT_INVALID
    finally:
        engine.dispose()

    try:
        path = write_report(report, report_dir)
    except (OSError, ValueError) as exc:
        logger.error("could not write monitoring report: %s", exc)
        return EXIT_INVALID
    logger.info(
        "monitoring report %s: status=%s predictions=%d validation_failures=%d path=%s",
        report.report_id,
        report.status.value,
        report.prediction_count,
        report.validation_failure_count,
        path,
    )
    return EXIT_OK


def _run_performance(settings: Settings, options: argparse.Namespace) -> int:
    path = Path(options.path)
    report_dir = Path(options.report_dir or settings.performance_report_dir)
    period: PerformancePeriod = options.period or settings.performance_period
    window_start = _parse_optional_datetime(options.window_start)
    window_end = _parse_optional_datetime(options.window_end)
    if options.window_start is not None and window_start is None:
        logger.error("--window-start must be a timezone-aware ISO-8601 datetime")
        return EXIT_INVALID
    if options.window_end is not None and window_end is None:
        logger.error("--window-end must be a timezone-aware ISO-8601 datetime")
        return EXIT_INVALID

    try:
        dataset = load_performance_dataset(path)
        report = build_performance_report(
            dataset=dataset,
            threshold=float(options.threshold),
            period=period,
            label_source=str(options.label_source),
            labels_verified=bool(options.labels_verified),
            window_start=window_start,
            window_end=window_end,
        )
    except (DataError, PerformanceReportError, OSError, ValueError) as exc:
        logger.error("could not build performance report: %s", exc)
        return EXIT_INVALID

    engine: Engine | None = None
    try:
        if not options.no_persist:
            engine = create_db_engine(settings)
            session_factory = build_session_factory(engine)
            with session_factory() as session:
                PerformanceRepository(session).save_report(report)
    except (
        PersistenceError,
        PerformanceReportError,
        SQLAlchemyError,
        OSError,
        ValueError,
    ) as exc:
        logger.exception("performance report persistence failed: %s", exc)
        return EXIT_INVALID
    finally:
        if engine is not None:
            engine.dispose()

    try:
        output_path = write_performance_report(report, report_dir)
    except OSError as exc:
        logger.error("could not write performance report: %s", exc)
        return EXIT_INVALID

    logger.info(
        "performance report %s: status=%s rows=%d versions=%d segments=%d path=%s",
        report.report_id,
        "ok" if report.labels_verified and not report.overall.undefined_metrics else "partial",
        report.row_count,
        report.model_version_count,
        report.segment_count,
        output_path,
    )
    return EXIT_OK


def _parse_window_end(value: str | None) -> datetime | None:
    if value is None:
        return datetime.now(UTC)
    parsed = _parse_optional_datetime(value)
    return None if parsed is None else parsed.replace(microsecond=0)


def _parse_optional_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(UTC)


__all__ = ["build_parser", "main"]
