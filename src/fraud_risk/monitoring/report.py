"""Machine-readable monitoring reports.

Responsibility
    Combine compact reference distributions with a bounded window of stored predictions,
    validation-failure counts and optional feature snapshots into one JSON report. The report
    separates data drift from model performance explicitly:

    - Data drift: feature and score distribution shift.
    - Data quality: missing values and input-validation failures.
    - Performance: unavailable until complete delayed labels exist.

Rules
    - No raw transaction row is written to a report. Features are reduced to per-feature
      statistics and missing rates.
    - A drifted feature or score is not evidence of performance degradation.
    - An empty window is a valid report that says it is empty, not an error.
    - The same report object is written to disk and persisted in ``drift_reports``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import numpy as np

from fraud_risk.config import Settings
from fraud_risk.data.dataset import AMOUNT_COLUMN
from fraud_risk.monitoring.drift import (
    DriftStatus,
    bin_proportions,
    combine_statuses,
    finite_values,
    ks_p_value,
    ks_statistic,
    metric_status,
    missing_rate,
    psi_from_proportions,
)
from fraud_risk.monitoring.reference import ReferenceProfile

REPORT_SCHEMA_VERSION = 1


class MonitoringReportError(ValueError):
    """A window, sample or reference cannot produce a valid monitoring report."""


@dataclass(frozen=True, slots=True)
class MonitoringThresholds:
    """Version-controlled warning and alert bands used by one report."""

    feature_psi_warning: float
    feature_psi_alert: float
    score_psi_warning: float
    score_psi_alert: float
    ks_warning: float
    ks_alert: float
    missing_rate_warning: float
    missing_rate_alert: float
    validation_failure_rate_warning: float
    validation_failure_rate_alert: float
    sample_limit: int

    @classmethod
    def from_settings(cls, settings: Settings) -> MonitoringThresholds:
        """Build the thresholds from the single configuration object."""
        return cls(
            feature_psi_warning=settings.monitoring_feature_psi_warning,
            feature_psi_alert=settings.monitoring_feature_psi_alert,
            score_psi_warning=settings.monitoring_score_psi_warning,
            score_psi_alert=settings.monitoring_score_psi_alert,
            ks_warning=settings.monitoring_ks_warning,
            ks_alert=settings.monitoring_ks_alert,
            missing_rate_warning=settings.monitoring_missing_rate_warning,
            missing_rate_alert=settings.monitoring_missing_rate_alert,
            validation_failure_rate_warning=(settings.monitoring_validation_failure_rate_warning),
            validation_failure_rate_alert=settings.monitoring_validation_failure_rate_alert,
            sample_limit=settings.monitoring_sample_limit,
        )

    def to_dict(self) -> dict[str, float | int]:
        """Return the thresholds as a JSON-ready object."""
        return {
            "feature_psi_warning": self.feature_psi_warning,
            "feature_psi_alert": self.feature_psi_alert,
            "score_psi_warning": self.score_psi_warning,
            "score_psi_alert": self.score_psi_alert,
            "ks_warning": self.ks_warning,
            "ks_alert": self.ks_alert,
            "missing_rate_warning": self.missing_rate_warning,
            "missing_rate_alert": self.missing_rate_alert,
            "validation_failure_rate_warning": self.validation_failure_rate_warning,
            "validation_failure_rate_alert": self.validation_failure_rate_alert,
            "sample_limit": self.sample_limit,
        }


@dataclass(frozen=True, slots=True)
class ModelVersionUsage:
    """Prediction counts for one model version in the window."""

    model_version: str
    total: int
    fraud: int
    legitimate: int


@dataclass(frozen=True, slots=True)
class AggregateCounts:
    """Exact database aggregates for the window, independent of the bounded sample."""

    prediction_count: int
    fraud_count: int
    legitimate_count: int
    model_versions: tuple[ModelVersionUsage, ...]
    review_status_counts: Mapping[str, int]
    validation_failure_count: int


@dataclass(frozen=True, slots=True)
class AnalysisSample:
    """Bounded rows used for score and feature distributions."""

    scores: tuple[float, ...]
    feature_rows: tuple[Mapping[str, float | None], ...]
    snapshot_count: int


@dataclass(frozen=True, slots=True)
class MonitoringReport:
    """A persisted report plus the summary columns used by the API and dashboard."""

    report_id: UUID
    generated_at: datetime
    window_start: datetime
    window_end: datetime
    reference_id: str
    status: DriftStatus
    data_drift_status: DriftStatus
    data_quality_status: DriftStatus
    prediction_count: int
    validation_failure_count: int
    feature_snapshot_count: int
    feature_drift_count: int
    payload: dict[str, Any]

    def to_json(self) -> str:
        """Return stable UTF-8 JSON text."""
        return json.dumps(self.payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def build_monitoring_report(
    *,
    reference: ReferenceProfile,
    aggregate: AggregateCounts,
    sample: AnalysisSample | None,
    thresholds: MonitoringThresholds,
    window_start: datetime,
    window_end: datetime,
    generated_at: datetime | None = None,
    report_id: UUID | None = None,
) -> MonitoringReport:
    """Build a report from exact window aggregates and a bounded row sample."""
    _validate_window(window_start, window_end)
    if aggregate.prediction_count < 0:
        msg = "prediction_count must not be negative"
        raise MonitoringReportError(msg)
    if aggregate.validation_failure_count < 0:
        msg = "validation_failure_count must not be negative"
        raise MonitoringReportError(msg)

    resolved_generated_at = generated_at or datetime.now(UTC)
    resolved_id = report_id or uuid4()
    feature_snapshots_available = sample is not None and sample.snapshot_count > 0
    score_sample = tuple(sample.scores) if sample is not None else ()

    feature_section = _feature_section(
        reference=reference,
        sample=sample,
        available=feature_snapshots_available,
        thresholds=thresholds,
    )
    missing_section = _missing_section(
        reference=reference,
        sample=sample,
        available=feature_snapshots_available,
        thresholds=thresholds,
    )
    score_section = _score_section(
        reference=reference,
        scores=score_sample,
        prediction_count=aggregate.prediction_count,
        thresholds=thresholds,
    )
    validation_section = _validation_section(
        validation_failure_count=aggregate.validation_failure_count,
        prediction_count=aggregate.prediction_count,
        thresholds=thresholds,
    )
    traffic_section = _traffic_section(
        aggregate=aggregate,
        sampled_predictions=len(score_sample),
        sample_limit=thresholds.sample_limit,
    )
    performance_section = _performance_section(
        review_status_counts=aggregate.review_status_counts,
        prediction_count=aggregate.prediction_count,
    )

    feature_status = _status_from_section(feature_section)
    missing_status = _status_from_section(missing_section)
    score_status = _status_from_section(score_section)
    validation_status = _status_from_section(validation_section)
    data_drift_status = combine_statuses([feature_status, score_status])
    data_quality_status = combine_statuses([missing_status, validation_status])
    overall_status = combine_statuses([data_drift_status, data_quality_status])
    feature_drift_count = int(feature_section.get("breached_feature_count", 0))

    summary_reason = _summary_reason(
        prediction_count=aggregate.prediction_count,
        validation_failure_count=aggregate.validation_failure_count,
        feature_available=feature_snapshots_available,
        score_available=bool(score_sample),
    )
    payload: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_id": str(resolved_id),
        "generated_at": resolved_generated_at.astimezone(UTC).isoformat(),
        "window": {
            "start": window_start.astimezone(UTC).isoformat(),
            "end": window_end.astimezone(UTC).isoformat(),
            "hours": round((window_end - window_start).total_seconds() / 3600.0, 6),
        },
        "reference": {
            "reference_id": reference.reference_id,
            "generated_at": reference.generated_at.astimezone(UTC).isoformat(),
            "source": reference.source,
            "model_version": reference.model_version,
            "model_sha256": reference.model_sha256,
            "threshold": reference.threshold,
            "split_strategy": reference.split_strategy,
            "split_seed": reference.split_seed,
            "training_rows": reference.training_rows,
            "scored_rows": reference.scored_rows,
            "sample_method": reference.sample_method,
        },
        "summary": {
            "status": overall_status.value,
            "data_drift_status": data_drift_status.value,
            "data_quality_status": data_quality_status.value,
            "performance_status": DriftStatus.UNAVAILABLE.value,
            "reason": summary_reason,
            "prediction_count": aggregate.prediction_count,
            "validation_failure_count": aggregate.validation_failure_count,
            "feature_snapshot_count": 0 if sample is None else sample.snapshot_count,
            "feature_drift_count": feature_drift_count,
            "performance_note": (
                "Data drift is not model-performance degradation. Performance cannot be "
                "measured for the full window without complete delayed ground-truth labels."
            ),
        },
        "feature_drift": feature_section,
        "missing_values": missing_section,
        "score_drift": score_section,
        "validation_failures": validation_section,
        "traffic": traffic_section,
        "model_versions": traffic_section["model_versions"],
        "performance": performance_section,
        "thresholds": thresholds.to_dict(),
        "caveats": _caveats(
            feature_available=feature_snapshots_available,
            sample_truncated=aggregate.prediction_count > len(score_sample),
            sample_limit=thresholds.sample_limit,
        ),
    }

    return MonitoringReport(
        report_id=resolved_id,
        generated_at=resolved_generated_at,
        window_start=window_start,
        window_end=window_end,
        reference_id=reference.reference_id,
        status=overall_status,
        data_drift_status=data_drift_status,
        data_quality_status=data_quality_status,
        prediction_count=aggregate.prediction_count,
        validation_failure_count=aggregate.validation_failure_count,
        feature_snapshot_count=0 if sample is None else sample.snapshot_count,
        feature_drift_count=feature_drift_count,
        payload=payload,
    )


def feature_rows_from_snapshots(
    snapshots: Sequence[Mapping[str, Any] | None],
    feature_columns: Sequence[str],
) -> tuple[tuple[dict[str, float | None], ...], int]:
    """Reduce stored snapshot objects to feature rows and a snapshot count.

    The stored shape is ``{"amount": ..., "features": {...}}``. A malformed or partial row
    becomes a row of missing values rather than crashing a scheduled report; the missing-rate
    section will expose it. No identifier or label is read.
    """
    columns = tuple(feature_columns)
    rows: list[dict[str, float | None]] = []
    snapshot_count = 0
    for snapshot in snapshots:
        if snapshot is None:
            continue
        snapshot_count += 1
        nested = snapshot.get("features") if isinstance(snapshot, Mapping) else None
        feature_values = nested if isinstance(nested, Mapping) else {}
        row: dict[str, float | None] = {}
        for column in columns:
            if column == AMOUNT_COLUMN:
                raw: object = snapshot.get(AMOUNT_COLUMN.lower(), snapshot.get(AMOUNT_COLUMN))
            else:
                raw = feature_values.get(column)
            row[column] = _optional_number(raw)
        rows.append(row)
    return tuple(rows), snapshot_count


def write_report(report: MonitoringReport, directory: Path) -> Path:
    """Write the JSON report and update ``latest.json``, returning the timestamped path."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = report.generated_at.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = directory / f"drift_{stamp}_{str(report.report_id)[:8]}.json"
    text = report.to_json()
    path.write_text(text, encoding="utf-8")
    (directory / "latest.json").write_text(text, encoding="utf-8")
    return path


def _feature_section(
    *,
    reference: ReferenceProfile,
    sample: AnalysisSample | None,
    available: bool,
    thresholds: MonitoringThresholds,
) -> dict[str, Any]:
    if not available or sample is None or not sample.feature_rows:
        return {
            "available": False,
            "status": DriftStatus.UNAVAILABLE.value,
            "reason": (
                "No feature snapshots in the window. Set PERSIST_FEATURE_SNAPSHOT=true and "
                "collect a new window to enable feature drift."
            ),
            "features": [],
            "breached_feature_count": 0,
        }

    rows = sample.feature_rows
    features: list[dict[str, Any]] = []
    statuses: list[DriftStatus] = []
    for distribution in reference.features:
        raw_values = [row.get(distribution.name) for row in rows]
        numeric = [
            number
            for number in (_optional_number(value) for value in raw_values)
            if number is not None
        ]
        missing = 0.0 if not raw_values else (len(raw_values) - len(numeric)) / len(raw_values)
        psi: float | None = None
        ks: float | None = None
        p_value: float | None = None
        if numeric:
            current_proportions = bin_proportions(numeric, distribution.bin_edges)
            psi = psi_from_proportions(distribution.bin_proportions, current_proportions)
            ks = ks_statistic(distribution.quantile_sample, numeric)
            p_value = ks_p_value(ks, distribution.count, len(numeric))
        psi_status = (
            None
            if psi is None
            else metric_status(
                psi,
                warning=thresholds.feature_psi_warning,
                alert=thresholds.feature_psi_alert,
            )
        )
        ks_status = (
            None
            if ks is None
            else metric_status(
                ks,
                warning=thresholds.ks_warning,
                alert=thresholds.ks_alert,
            )
        )
        missing_status = metric_status(
            missing,
            warning=thresholds.missing_rate_warning,
            alert=thresholds.missing_rate_alert,
        )
        available_statuses = [
            item for item in (psi_status, ks_status, missing_status) if item is not None
        ]
        status = combine_statuses(available_statuses)
        statuses.append(status)
        features.append(
            {
                "feature": distribution.name,
                "available": bool(numeric),
                "status": status.value,
                "psi": psi,
                "ks_statistic": ks,
                "ks_p_value": p_value,
                "missing_rate": missing,
                "reference_mean": distribution.mean,
                "current_mean": float(np.mean(numeric)) if numeric else None,
                "sample_count": len(numeric),
            }
        )

    features.sort(key=lambda item: (-_status_rank(item["status"]), -(item["psi"] or 0.0)))
    breached = [
        item for item in features if item["status"] in {DriftStatus.WARNING, DriftStatus.ALERT}
    ]
    return {
        "available": True,
        "status": combine_statuses(statuses).value,
        "reason": None,
        "sample_rows": len(rows),
        "breached_feature_count": len(breached),
        "features": features,
    }


def _missing_section(
    *,
    reference: ReferenceProfile,
    sample: AnalysisSample | None,
    available: bool,
    thresholds: MonitoringThresholds,
) -> dict[str, Any]:
    if not available or sample is None or not sample.feature_rows:
        return {
            "available": False,
            "status": DriftStatus.UNAVAILABLE.value,
            "reason": "Missing-value monitoring needs feature snapshots in the window.",
            "overall_missing_rate": None,
            "features": [],
        }

    columns = tuple(item.name for item in reference.features)
    rows = sample.feature_rows
    cell_count = len(rows) * len(columns)
    missing_cells = 0
    feature_rows: list[dict[str, Any]] = []
    statuses: list[DriftStatus] = []
    for column in columns:
        values = [row.get(column) for row in rows]
        rate = missing_rate(values)
        missing_cells += sum(1 for value in values if not _is_finite_number(value))
        status = metric_status(
            rate,
            warning=thresholds.missing_rate_warning,
            alert=thresholds.missing_rate_alert,
        )
        statuses.append(status)
        feature_rows.append(
            {
                "feature": column,
                "missing_rate": rate,
                "status": status.value,
            }
        )
    overall = missing_cells / cell_count if cell_count else None
    overall_status = metric_status(
        overall,
        warning=thresholds.missing_rate_warning,
        alert=thresholds.missing_rate_alert,
    )
    return {
        "available": True,
        "status": combine_statuses([overall_status, *statuses]).value,
        "reason": None,
        "snapshot_rows": len(rows),
        "feature_cells": cell_count,
        "overall_missing_rate": overall,
        "features": feature_rows,
    }


def _score_section(
    *,
    reference: ReferenceProfile,
    scores: Sequence[float],
    prediction_count: int,
    thresholds: MonitoringThresholds,
) -> dict[str, Any]:
    numeric = finite_values(np.asarray(scores, dtype="float64"))
    if prediction_count == 0 or numeric.size == 0:
        return {
            "available": False,
            "status": DriftStatus.UNAVAILABLE.value,
            "reason": "No prediction scores in the window.",
            "sample_count": 0,
            "psi": None,
            "ks_statistic": None,
            "ks_p_value": None,
            "current_mean": None,
            "reference_mean": reference.score.mean,
            "mean_shift": None,
            "quantiles": {},
        }

    current_proportions = bin_proportions(numeric, reference.score.bin_edges)
    psi = psi_from_proportions(reference.score.bin_proportions, current_proportions)
    ks = ks_statistic(reference.score.quantile_sample, numeric)
    p_value = ks_p_value(ks, reference.score.count, int(numeric.size))
    psi_status = metric_status(
        psi,
        warning=thresholds.score_psi_warning,
        alert=thresholds.score_psi_alert,
    )
    ks_status = metric_status(ks, warning=thresholds.ks_warning, alert=thresholds.ks_alert)
    current_mean = float(np.mean(numeric))
    quantile_points = np.quantile(numeric, [0.5, 0.9, 0.99])
    return {
        "available": True,
        "status": combine_statuses([psi_status, ks_status]).value,
        "reason": None,
        "sample_count": int(numeric.size),
        "psi": psi,
        "ks_statistic": ks,
        "ks_p_value": p_value,
        "current_mean": current_mean,
        "reference_mean": reference.score.mean,
        "mean_shift": current_mean - reference.score.mean,
        "quantiles": {
            "p50": float(quantile_points[0]),
            "p90": float(quantile_points[1]),
            "p99": float(quantile_points[2]),
        },
    }


def _validation_section(
    *,
    validation_failure_count: int,
    prediction_count: int,
    thresholds: MonitoringThresholds,
) -> dict[str, Any]:
    attempts = validation_failure_count + prediction_count
    rate = (validation_failure_count / attempts) if attempts else None
    return {
        "available": attempts > 0,
        "status": (
            metric_status(
                rate,
                warning=thresholds.validation_failure_rate_warning,
                alert=thresholds.validation_failure_rate_alert,
            ).value
            if attempts
            else DriftStatus.UNAVAILABLE.value
        ),
        "reason": None if attempts else "No scoring attempts or validation failures in window.",
        "count": validation_failure_count,
        "successful_predictions": prediction_count,
        "attempts": attempts,
        "rate": rate,
    }


def _traffic_section(
    *,
    aggregate: AggregateCounts,
    sampled_predictions: int,
    sample_limit: int,
) -> dict[str, Any]:
    total = aggregate.prediction_count
    versions = [
        {
            "model_version": item.model_version,
            "total": item.total,
            "fraud": item.fraud,
            "legitimate": item.legitimate,
            "share": (item.total / total) if total else 0.0,
        }
        for item in sorted(
            aggregate.model_versions,
            key=lambda item: (-item.total, item.model_version),
        )
    ]
    return {
        "predictions": total,
        "sampled_predictions": sampled_predictions,
        "sample_limit": sample_limit,
        "sample_truncated": total > sampled_predictions,
        "decision_mix": {
            "fraud": aggregate.fraud_count,
            "legitimate": aggregate.legitimate_count,
            "fraud_share": (aggregate.fraud_count / total) if total else None,
        },
        "model_versions": versions,
    }


def _performance_section(
    *,
    review_status_counts: Mapping[str, int],
    prediction_count: int,
) -> dict[str, Any]:
    confirmed_fraud = int(review_status_counts.get("confirmed_fraud", 0))
    confirmed_legitimate = int(review_status_counts.get("confirmed_legitimate", 0))
    reviewed = confirmed_fraud + confirmed_legitimate
    if prediction_count == 0:
        reason = "No predictions in the window, so no delayed labels can exist yet."
    elif reviewed == 0:
        reason = (
            "No analyst-confirmed outcomes in the window. Delayed ground-truth labels are "
            "unavailable, so performance degradation cannot be measured."
        )
    else:
        reason = (
            "Analyst-confirmed outcomes cover only a non-random reviewed subset "
            f"({reviewed}/{prediction_count} predictions). They are not complete delayed "
            "ground truth, so no overall PR-AUC, recall or accuracy is reported."
        )
    return {
        "available": False,
        "status": DriftStatus.UNAVAILABLE.value,
        "reason": reason,
        "reviewed_predictions": reviewed,
        "confirmed_fraud": confirmed_fraud,
        "confirmed_legitimate": confirmed_legitimate,
        "coverage": (reviewed / prediction_count) if prediction_count else None,
        "metrics": None,
    }


def _summary_reason(
    *,
    prediction_count: int,
    validation_failure_count: int,
    feature_available: bool,
    score_available: bool,
) -> str:
    if prediction_count == 0 and validation_failure_count == 0:
        return "The monitoring window is empty: no predictions or validation failures."
    parts: list[str] = []
    if prediction_count == 0:
        parts.append("No predictions were stored; validation failures were still counted.")
    if not score_available and prediction_count:
        parts.append("No score sample was available.")
    if not feature_available and prediction_count:
        parts.append(
            "Feature drift is unavailable because PERSIST_FEATURE_SNAPSHOT is disabled or "
            "no feature snapshots were stored."
        )
    return " ".join(parts) if parts else "At least one monitoring signal is available."


def _caveats(
    *,
    feature_available: bool,
    sample_truncated: bool,
    sample_limit: int,
) -> list[str]:
    caveats = [
        "Data drift is not model-performance degradation; performance needs delayed labels.",
        "PSI uses reference-derived quantile bins and floors empty bins at 1e-6.",
        "KS is computed against a 101-point reference quantile sample and is approximate.",
    ]
    if not feature_available:
        caveats.append(
            "Feature distribution and missing-value monitoring require opt-in feature "
            "snapshots; none were available for this window."
        )
    if sample_truncated:
        caveats.append(
            "Score and feature distributions use the newest "
            f"{sample_limit} prediction rows; aggregate traffic counts are exact."
        )
    return caveats


def _status_from_section(section: Mapping[str, Any]) -> DriftStatus:
    value = section.get("status")
    if not isinstance(value, str):
        return DriftStatus.UNAVAILABLE
    try:
        return DriftStatus(value)
    except ValueError:
        return DriftStatus.UNAVAILABLE


def _status_rank(value: object) -> int:
    return {
        DriftStatus.ALERT.value: 3,
        DriftStatus.WARNING.value: 2,
        DriftStatus.OK.value: 1,
    }.get(str(value), 0)


def _validate_window(start: datetime, end: datetime) -> None:
    for name, value in (("window_start", start), ("window_end", end)):
        if value.tzinfo is None or value.utcoffset() is None:
            msg = f"{name} must include a timezone"
            raise MonitoringReportError(msg)
    if end <= start:
        msg = "window_end must be after window_start"
        raise MonitoringReportError(msg)


def _optional_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _is_finite_number(value: object) -> bool:
    return _optional_number(value) is not None


__all__ = [
    "REPORT_SCHEMA_VERSION",
    "AggregateCounts",
    "AnalysisSample",
    "ModelVersionUsage",
    "MonitoringReport",
    "MonitoringReportError",
    "MonitoringThresholds",
    "build_monitoring_report",
    "feature_rows_from_snapshots",
    "write_report",
]
