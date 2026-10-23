"""Monitoring (architecture area 8).

Responsibility
    Detect drift in inputs and scores, record count-only data-quality signals, persist
    machine-readable reports, and state explicitly when performance cannot be measured
    because complete delayed labels are unavailable.

Entry points
    - ``python -m fraud_risk.models.reference`` builds a compact reference profile from the
      training split (offline model layer).
    - ``python -m fraud_risk.monitoring run`` scores one window against that profile, writes
      JSON and persists a ``drift_reports`` row.
    - ``python -m fraud_risk.monitoring performance`` evaluates a separately supplied,
      scored and labeled extract, writes JSON and persists a ``performance_reports`` row.
    - ``GET /api/v1/monitoring/summary`` and ``GET /api/v1/performance/summary`` expose the
      latest persisted reports to the dashboard.

Status
    Implemented for M8; see ``docs/MONITORING.md``, ``docs/PERFORMANCE.md`` and
    ``docs/ROADMAP.md``.
"""

from fraud_risk.monitoring.drift import DriftStatus
from fraud_risk.monitoring.performance import (
    LabeledEvaluationDataset,
    PerformanceEvaluationReport,
    PerformanceMetrics,
    PerformanceReportError,
    PerformanceSchemaError,
    build_performance_report,
    load_performance_dataset,
    prepare_performance_frame,
    write_performance_report,
)
from fraud_risk.monitoring.performance_repository import PerformanceRepository
from fraud_risk.monitoring.report import (
    AggregateCounts,
    AnalysisSample,
    MonitoringReport,
    MonitoringThresholds,
    build_monitoring_report,
)
from fraud_risk.monitoring.repository import MonitoringRepository

__all__ = [
    "AggregateCounts",
    "AnalysisSample",
    "DriftStatus",
    "LabeledEvaluationDataset",
    "MonitoringReport",
    "MonitoringRepository",
    "MonitoringThresholds",
    "PerformanceEvaluationReport",
    "PerformanceMetrics",
    "PerformanceReportError",
    "PerformanceRepository",
    "PerformanceSchemaError",
    "build_monitoring_report",
    "build_performance_report",
    "load_performance_dataset",
    "prepare_performance_frame",
    "write_performance_report",
]
