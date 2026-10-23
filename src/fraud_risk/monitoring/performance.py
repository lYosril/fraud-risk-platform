"""Repeatable, aggregate-only performance reports from verified delayed labels.

Responsibility
    Turn a *scored, labeled evaluation extract* into the performance numbers the dashboard
    reports: precision, recall, F1, PR-AUC, ROC-AUC, the confusion matrix and the
    false-positive rate, grouped by model version and evaluation period.

Input contract
    The CSV must contain exactly the model feature columns (``V1..V28`` and ``Amount``), the
    verified target ``is_fraud``, the score served at prediction time ``risk_score``, the
    model identity ``model_version`` and the scoring timestamp ``scored_at``. Scores are
    never recomputed with a later model: a report describes the scores that actually served
    each row. ``review_status`` and the raw ``Class`` column are rejected because analyst
    review state is not confirmed ground truth.

Undefined metrics
    A metric that cannot be computed is represented as
    ``{"value": null, "defined": false, "reason": "..."}``. It is never coerced to zero or
    silently omitted. PR-AUC and ROC-AUC require both classes; precision requires a predicted
    positive; recall requires an actual positive; F1 requires both precision and recall; and
    the false-positive rate requires an actual negative.

Rules
    - Reports contain aggregates, metric definitions and provenance only. No feature row,
      transaction identifier or raw label is written.
    - The workflow does not read the ``predictions`` table, ``review_status`` or the frozen
      final-evaluation metrics. It is a separate routine evaluation path.
    - A chosen evaluation threshold is recorded with the report. The threshold is not
      calibrated here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score, roc_auc_score

from fraud_risk.config import PerformancePeriod
from fraud_risk.data.dataset import AMOUNT_COLUMN, FEATURE_COLUMNS
from fraud_risk.data.ingestion import load_csv

REPORT_SCHEMA_VERSION = 1

TARGET_COLUMN = "is_fraud"
SCORE_COLUMN = "risk_score"
MODEL_VERSION_COLUMN = "model_version"
SCORED_AT_COLUMN = "scored_at"

#: The exact feature schema the model was trained on. ``Time`` is deliberately absent.
PERFORMANCE_FEATURE_COLUMNS: tuple[str, ...] = (*FEATURE_COLUMNS, AMOUNT_COLUMN)

#: The exact, closed input schema. Extra columns are rejected rather than ignored.
PERFORMANCE_REQUIRED_COLUMNS: tuple[str, ...] = (
    *PERFORMANCE_FEATURE_COLUMNS,
    TARGET_COLUMN,
    SCORE_COLUMN,
    MODEL_VERSION_COLUMN,
    SCORED_AT_COLUMN,
)

PERFORMANCE_PERIODS: tuple[PerformancePeriod, ...] = ("day", "week", "month")

#: Stable definitions recorded with every report. A reader must not have to guess a formula.
METRIC_DEFINITIONS: dict[str, str] = {
    "threshold": (
        "A row is predicted positive when risk_score >= threshold. The threshold is supplied "
        "by the operator and recorded with the report; it is not recalibrated here."
    ),
    "confusion_matrix": (
        "Counts at the recorded threshold with labels 0/1: true_negative, false_positive, "
        "false_negative and true_positive."
    ),
    "precision": "true_positive / (true_positive + false_positive).",
    "recall": "true_positive / (true_positive + false_negative).",
    "f1": "2 * precision * recall / (precision + recall).",
    "pr_auc": (
        "Average precision (scikit-learn average_precision_score) from the positive-class "
        "scores. It is undefined unless both classes are present."
    ),
    "roc_auc": (
        "Area under the ROC curve from the positive-class scores. It is undefined unless both "
        "classes are present."
    ),
    "false_positive_rate": "false_positive / (false_positive + true_negative).",
}

_ISO_TIMEZONE_SUFFIX = r"(?:Z|[+-]\d{2}:\d{2})$"
_CONTROL_CHARACTERS = r"[\x00-\x1f\x7f]"


class PerformanceReportError(ValueError):
    """An evaluation extract or report request cannot produce a valid report."""


class PerformanceSchemaError(PerformanceReportError):
    """The evaluation extract does not match the exact performance input schema."""


@dataclass(frozen=True, slots=True)
class LabeledEvaluationDataset:
    """A validated, normalized evaluation frame plus its content fingerprint."""

    frame: pd.DataFrame
    source_sha256: str
    source_rows: int


@dataclass(frozen=True, slots=True)
class MetricValue:
    """A metric value with an explicit undefined state."""

    value: float | None
    defined: bool
    reason: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON shape used by disk, API and dashboard."""
        return {"value": self.value, "defined": self.defined, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    """Every reported metric for one row group at one threshold."""

    rows: int
    positives: int
    negatives: int
    threshold: float
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int
    precision: MetricValue
    recall: MetricValue
    f1: MetricValue
    pr_auc: MetricValue
    roc_auc: MetricValue
    false_positive_rate: MetricValue

    @property
    def undefined_metrics(self) -> tuple[str, ...]:
        """Return the metric names that are explicitly undefined."""
        values = {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "pr_auc": self.pr_auc,
            "roc_auc": self.roc_auc,
            "false_positive_rate": self.false_positive_rate,
        }
        return tuple(name for name, metric in values.items() if not metric.defined)

    def to_dict(self) -> dict[str, Any]:
        """Return the aggregate metric block; no row-level value is included."""
        return {
            "rows": self.rows,
            "positives": self.positives,
            "negatives": self.negatives,
            "threshold": self.threshold,
            "confusion_matrix": {
                "true_negative": self.true_negatives,
                "false_positive": self.false_positives,
                "false_negative": self.false_negatives,
                "true_positive": self.true_positives,
            },
            "precision": self.precision.to_dict(),
            "recall": self.recall.to_dict(),
            "f1": self.f1.to_dict(),
            "pr_auc": self.pr_auc.to_dict(),
            "roc_auc": self.roc_auc.to_dict(),
            "false_positive_rate": self.false_positive_rate.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ModelVersionPerformance:
    """Metrics for one model version across all selected periods."""

    model_version: str
    metrics: PerformanceMetrics

    def to_dict(self) -> dict[str, Any]:
        """Return the version summary block."""
        return {"model_version": self.model_version, **self.metrics.to_dict()}


@dataclass(frozen=True, slots=True)
class PerformanceSegment:
    """Metrics for one (model version, evaluation period) segment."""

    model_version: str
    period_start: datetime
    period_end: datetime
    period_label: str
    metrics: PerformanceMetrics

    def to_dict(self) -> dict[str, Any]:
        """Return the segment block with an explicit, exclusive period end."""
        return {
            "model_version": self.model_version,
            "period": {
                "label": self.period_label,
                "start": self.period_start.astimezone(UTC).isoformat(),
                "end": self.period_end.astimezone(UTC).isoformat(),
                "end_inclusive": False,
            },
            **self.metrics.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class PerformanceEvaluationReport:
    """A persisted aggregate report plus the summary columns used by the API."""

    report_id: UUID
    generated_at: datetime
    window_start: datetime
    window_end: datetime
    row_count: int
    model_version_count: int
    segment_count: int
    threshold: float
    period: PerformancePeriod
    labels_verified: bool
    label_source: str
    overall: PerformanceMetrics
    model_versions: tuple[ModelVersionPerformance, ...]
    segments: tuple[PerformanceSegment, ...]
    payload: dict[str, Any]

    def to_json(self) -> str:
        """Return stable UTF-8 JSON text."""
        return json.dumps(self.payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def prepare_performance_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the exact schema and return a normalized frame without raw-label leakage.

    The returned frame has only the required columns, with features and scores as float64,
    the target as int8, timestamps as timezone-aware UTC and model versions as strings.
    """
    if frame.columns.duplicated().any():
        duplicates = sorted({str(column) for column in frame.columns[frame.columns.duplicated()]})
        msg = f"duplicate column name(s): {', '.join(duplicates)}"
        raise PerformanceSchemaError(msg)

    columns = tuple(str(column) for column in frame.columns)
    if "review_status" in columns:
        msg = (
            "review_status is analyst workflow state, not confirmed ground truth; supply a "
            "verified is_fraud target instead"
        )
        raise PerformanceSchemaError(msg)
    if "Class" in columns:
        msg = (
            "the raw Class column is not accepted; export the verified target as is_fraud "
            "and include the evaluation provenance columns"
        )
        raise PerformanceSchemaError(msg)

    missing = [column for column in PERFORMANCE_REQUIRED_COLUMNS if column not in columns]
    unexpected = [column for column in columns if column not in PERFORMANCE_REQUIRED_COLUMNS]
    if missing or unexpected:
        parts: list[str] = []
        if missing:
            parts.append(f"missing: {', '.join(missing)}")
        if unexpected:
            parts.append(f"unexpected: {', '.join(unexpected)}")
        msg = (
            "evaluation dataset does not match the exact performance schema "
            f"({'; '.join(parts)}). Required columns are "
            f"{', '.join(PERFORMANCE_REQUIRED_COLUMNS)}"
        )
        raise PerformanceSchemaError(msg)

    if len(frame) == 0:
        msg = "the evaluation dataset has no data rows"
        raise PerformanceSchemaError(msg)

    missing_values = frame[list(PERFORMANCE_REQUIRED_COLUMNS)].isna().sum()
    null_columns = [str(column) for column, count in missing_values.items() if int(count) > 0]
    if null_columns:
        msg = f"null values are not allowed in required column(s): {', '.join(null_columns)}"
        raise PerformanceSchemaError(msg)

    normalized = pd.DataFrame(index=frame.index)
    for column in PERFORMANCE_FEATURE_COLUMNS:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if bool(numeric.isna().any()):
            msg = f"feature column {column!r} must contain finite numeric values"
            raise PerformanceSchemaError(msg)
        values = np.asarray(numeric.to_numpy(), dtype="float64")
        if not bool(np.isfinite(values).all()):
            msg = f"feature column {column!r} must contain finite numeric values"
            raise PerformanceSchemaError(msg)
        normalized[column] = values

    if bool((np.asarray(normalized[AMOUNT_COLUMN], dtype="float64") < 0.0).any()):
        msg = f"{AMOUNT_COLUMN} must be non-negative"
        raise PerformanceSchemaError(msg)

    raw_target = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    if bool(raw_target.isna().any()):
        msg = f"{TARGET_COLUMN} must contain only the labels 0 and 1"
        raise PerformanceSchemaError(msg)
    target_values = np.asarray(raw_target.to_numpy(), dtype="float64")
    if not bool(np.isin(target_values, [0.0, 1.0]).all()):
        msg = f"{TARGET_COLUMN} must contain only the labels 0 and 1"
        raise PerformanceSchemaError(msg)
    normalized[TARGET_COLUMN] = target_values.astype("int8")

    raw_score = pd.to_numeric(frame[SCORE_COLUMN], errors="coerce")
    if bool(raw_score.isna().any()):
        msg = f"{SCORE_COLUMN} must contain finite scores in [0, 1]"
        raise PerformanceSchemaError(msg)
    score_values = np.asarray(raw_score.to_numpy(), dtype="float64")
    if not bool(np.isfinite(score_values).all()):
        msg = f"{SCORE_COLUMN} must contain finite scores in [0, 1]"
        raise PerformanceSchemaError(msg)
    if bool(((score_values < 0.0) | (score_values > 1.0)).any()):
        msg = f"{SCORE_COLUMN} must be in [0, 1]"
        raise PerformanceSchemaError(msg)
    normalized[SCORE_COLUMN] = score_values

    version_text = frame[MODEL_VERSION_COLUMN].astype("string").str.strip()
    if bool(version_text.isna().any()) or bool((version_text == "").any()):
        msg = f"{MODEL_VERSION_COLUMN} must contain a non-empty value on every row"
        raise PerformanceSchemaError(msg)
    if bool(version_text.str.len().gt(128).any()):
        msg = f"{MODEL_VERSION_COLUMN} values must be at most 128 characters"
        raise PerformanceSchemaError(msg)
    if bool(version_text.str.contains(_CONTROL_CHARACTERS, regex=True).any()):
        msg = f"{MODEL_VERSION_COLUMN} must not contain control characters"
        raise PerformanceSchemaError(msg)
    normalized[MODEL_VERSION_COLUMN] = version_text.astype("object")

    scored_at = frame[SCORED_AT_COLUMN]
    if pd.api.types.is_datetime64_any_dtype(scored_at):
        if getattr(scored_at.dt, "tz", None) is None:
            msg = f"{SCORED_AT_COLUMN} must include a timezone offset"
            raise PerformanceSchemaError(msg)
        parsed_time = pd.to_datetime(scored_at, utc=True, errors="coerce")
    else:
        raw_time = scored_at.astype("string")
        has_offset = raw_time.str.contains(_ISO_TIMEZONE_SUFFIX, regex=True, na=False)
        if not bool(has_offset.all()):
            msg = (
                f"{SCORED_AT_COLUMN} must be ISO-8601 with an explicit timezone offset "
                "(for example 2026-10-23T12:00:00Z)"
            )
            raise PerformanceSchemaError(msg)
        parsed_time = pd.to_datetime(raw_time, utc=True, errors="coerce", format="mixed")
    if bool(parsed_time.isna().any()):
        msg = f"{SCORED_AT_COLUMN} contains an unparseable timestamp"
        raise PerformanceSchemaError(msg)
    normalized[SCORED_AT_COLUMN] = parsed_time.dt.tz_convert("UTC")

    return normalized.reset_index(drop=True)


def load_performance_dataset(path: Path) -> LabeledEvaluationDataset:
    """Load a CSV evaluation extract, validate it and fingerprint the file."""
    resolved = Path(path)
    frame = prepare_performance_frame(load_csv(resolved))
    return LabeledEvaluationDataset(
        frame=frame,
        source_sha256=_sha256_file(resolved),
        source_rows=len(frame),
    )


def build_performance_report(
    *,
    dataset: LabeledEvaluationDataset,
    threshold: float,
    period: PerformancePeriod,
    label_source: str,
    labels_verified: bool,
    window_start: datetime | None = None,
    window_end: datetime | None = None,
    generated_at: datetime | None = None,
    report_id: UUID | None = None,
) -> PerformanceEvaluationReport:
    """Build one aggregate report from a validated evaluation extract."""
    _validate_threshold(threshold)
    _validate_period(period)
    _validate_label_source(label_source)
    _validate_sha256(dataset.source_sha256)
    _validate_optional_window(window_start, window_end)

    frame = dataset.frame
    if len(frame) == 0:
        msg = "the evaluation dataset has no data rows"
        raise PerformanceReportError(msg)

    timestamps = [value.to_pydatetime().astimezone(UTC) for value in frame[SCORED_AT_COLUMN]]
    selected_indices = [
        index
        for index, timestamp in enumerate(timestamps)
        if (window_start is None or timestamp >= window_start)
        and (window_end is None or timestamp <= window_end)
    ]
    if not selected_indices:
        msg = "the requested evaluation window contains no labeled rows"
        raise PerformanceReportError(msg)

    labels = np.asarray(frame[TARGET_COLUMN].to_numpy(), dtype="int8")
    scores = np.asarray(frame[SCORE_COLUMN].to_numpy(), dtype="float64")
    versions = [str(value) for value in frame[MODEL_VERSION_COLUMN].to_list()]

    selected_array = np.asarray(selected_indices, dtype="int64")
    selected_labels = labels[selected_array]
    selected_scores = scores[selected_array]
    selected_timestamps = [timestamps[index] for index in selected_indices]
    observed_start = min(selected_timestamps)
    observed_end = max(selected_timestamps)

    model_index: dict[str, list[int]] = {}
    segment_index: dict[tuple[str, datetime], list[int]] = {}
    segment_bounds: dict[tuple[str, datetime], tuple[datetime, datetime]] = {}
    for index in selected_indices:
        version = versions[index]
        start, end = period_bounds(timestamps[index], period)
        key = (version, start)
        model_index.setdefault(version, []).append(index)
        segment_index.setdefault(key, []).append(index)
        segment_bounds[key] = (start, end)

    overall = _compute_metrics(selected_labels, selected_scores, threshold)
    model_versions = tuple(
        ModelVersionPerformance(
            model_version=version,
            metrics=_metrics_for_indices(model_index[version], labels, scores, threshold),
        )
        for version in sorted(model_index)
    )
    segments = tuple(
        PerformanceSegment(
            model_version=version,
            period_start=segment_bounds[(version, start)][0],
            period_end=segment_bounds[(version, start)][1],
            period_label=period_label(start, period),
            metrics=_metrics_for_indices(
                segment_index[(version, start)], labels, scores, threshold
            ),
        )
        for version, start in sorted(segment_index)
    )

    resolved_generated_at = generated_at or datetime.now(UTC)
    resolved_report_id = report_id or uuid4()
    warnings = _warnings(
        overall=overall,
        labels_verified=labels_verified,
        segments=segments,
    )
    status = "ok" if labels_verified and not overall.undefined_metrics else "partial"
    resolved_label_source = label_source.strip()

    payload: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_id": str(resolved_report_id),
        "generated_at": resolved_generated_at.astimezone(UTC).isoformat(),
        "status": status,
        "summary": {
            "status": status,
            "row_count": len(selected_indices),
            "dataset_row_count": dataset.source_rows,
            "rows_excluded_by_window": dataset.source_rows - len(selected_indices),
            "model_version_count": len(model_versions),
            "segment_count": len(segments),
            "threshold": round(threshold, 6),
            "period": period,
            "labels_verified": labels_verified,
            "undefined_overall_metrics": list(overall.undefined_metrics),
            "warnings": warnings,
        },
        "window": {
            "start": observed_start.isoformat(),
            "end": observed_end.isoformat(),
            "requested_start": (
                None if window_start is None else window_start.astimezone(UTC).isoformat()
            ),
            "requested_end": (
                None if window_end is None else window_end.astimezone(UTC).isoformat()
            ),
            "period": period,
            "hours": round((observed_end - observed_start).total_seconds() / 3600.0, 6),
        },
        "dataset": {
            "sha256": dataset.source_sha256,
            "rows": dataset.source_rows,
            "feature_columns": list(PERFORMANCE_FEATURE_COLUMNS),
            "target_column": TARGET_COLUMN,
            "score_column": SCORE_COLUMN,
            "model_version_column": MODEL_VERSION_COLUMN,
            "timestamp_column": SCORED_AT_COLUMN,
        },
        "evaluation": {
            "threshold": round(threshold, 6),
            "period": period,
            "label_source": resolved_label_source,
            "labels_verified": labels_verified,
            "label_provenance": {
                "source": resolved_label_source,
                "verified": labels_verified,
                "analyst_review_status_used": False,
                "target_column": TARGET_COLUMN,
                "note": (
                    "Only the explicit is_fraud target in the supplied evaluation extract is "
                    "used. The API review_status field is never read as ground truth."
                ),
            },
            "score_source": (
                "risk_score supplied with the evaluation extract; scores are not recomputed "
                "with a later model"
            ),
        },
        "overall": overall.to_dict(),
        "by_model_version": [item.to_dict() for item in model_versions],
        "segments": [item.to_dict() for item in segments],
        "metric_definitions": dict(METRIC_DEFINITIONS),
        "caveats": _caveats(labels_verified=labels_verified),
    }

    return PerformanceEvaluationReport(
        report_id=resolved_report_id,
        generated_at=resolved_generated_at,
        window_start=observed_start,
        window_end=observed_end,
        row_count=len(selected_indices),
        model_version_count=len(model_versions),
        segment_count=len(segments),
        threshold=round(threshold, 6),
        period=period,
        labels_verified=labels_verified,
        label_source=resolved_label_source,
        overall=overall,
        model_versions=model_versions,
        segments=segments,
        payload=payload,
    )


def write_performance_report(report: PerformanceEvaluationReport, directory: Path) -> Path:
    """Write the JSON report and update ``latest.json``, returning the timestamped path."""
    resolved = Path(directory)
    resolved.mkdir(parents=True, exist_ok=True)
    stamp = report.generated_at.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = resolved / f"performance_{stamp}_{str(report.report_id)[:8]}.json"
    text = report.to_json()
    path.write_text(text, encoding="utf-8")
    (resolved / "latest.json").write_text(text, encoding="utf-8")
    return path


def period_bounds(timestamp: datetime, period: PerformancePeriod) -> tuple[datetime, datetime]:
    """Return the inclusive-start, exclusive-end bounds of a UTC calendar period."""
    _validate_period(period)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        msg = "timestamp must include a timezone"
        raise PerformanceReportError(msg)
    value = timestamp.astimezone(UTC)
    if period == "day":
        start = value.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
    elif period == "week":
        day_start = value.replace(hour=0, minute=0, second=0, microsecond=0)
        start = day_start - timedelta(days=day_start.weekday())
        end = start + timedelta(days=7)
    else:
        start = value.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end = (start + timedelta(days=32)).replace(day=1)
    return start, end


def period_label(start: datetime, period: PerformancePeriod) -> str:
    """Return a short, sortable label for a period start."""
    _validate_period(period)
    if period == "day":
        return start.astimezone(UTC).strftime("%Y-%m-%d")
    if period == "week":
        return start.astimezone(UTC).strftime("%G-W%V")
    return start.astimezone(UTC).strftime("%Y-%m")


def _compute_metrics(
    labels: NDArray[np.int8],
    scores: NDArray[np.float64],
    threshold: float,
) -> PerformanceMetrics:
    if labels.size == 0 or labels.size != scores.size:
        msg = "labels and scores must be non-empty and the same length"
        raise PerformanceReportError(msg)
    predictions = (scores >= threshold).astype("int8")
    true_positives = int(np.count_nonzero((labels == 1) & (predictions == 1)))
    false_positives = int(np.count_nonzero((labels == 0) & (predictions == 1)))
    false_negatives = int(np.count_nonzero((labels == 1) & (predictions == 0)))
    true_negatives = int(np.count_nonzero((labels == 0) & (predictions == 0)))
    positives = true_positives + false_negatives
    negatives = false_positives + true_negatives

    precision_value: float | None = None
    if true_positives + false_positives > 0:
        precision_value = true_positives / (true_positives + false_positives)
        precision = _defined(precision_value)
    else:
        precision = _undefined("no rows were predicted positive at this threshold")

    recall_value: float | None = None
    if positives > 0:
        recall_value = true_positives / positives
        recall = _defined(recall_value)
    else:
        recall = _undefined("no actual positives in this group")

    if precision_value is None or recall_value is None:
        f1 = _undefined("F1 requires defined precision and recall")
    else:
        denominator = precision_value + recall_value
        f1 = _defined(
            0.0 if denominator == 0 else 2.0 * precision_value * recall_value / denominator
        )

    if positives > 0 and negatives > 0:
        pr_auc = _defined(cast("float", average_precision_score(labels, scores)))
        roc_auc = _defined(cast("float", roc_auc_score(labels, scores)))
    else:
        pr_auc = _undefined("PR-AUC requires both positive and negative labels")
        roc_auc = _undefined("ROC-AUC requires both positive and negative labels")

    if false_positives + true_negatives > 0:
        false_positive_rate = _defined(false_positives / (false_positives + true_negatives))
    else:
        false_positive_rate = _undefined("no actual negatives in this group")

    return PerformanceMetrics(
        rows=int(labels.size),
        positives=positives,
        negatives=negatives,
        threshold=round(threshold, 6),
        true_negatives=true_negatives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        true_positives=true_positives,
        precision=precision,
        recall=recall,
        f1=f1,
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        false_positive_rate=false_positive_rate,
    )


def _metrics_for_indices(
    indices: list[int],
    labels: NDArray[np.int8],
    scores: NDArray[np.float64],
    threshold: float,
) -> PerformanceMetrics:
    selected = np.asarray(indices, dtype="int64")
    return _compute_metrics(labels[selected], scores[selected], threshold)


def _defined(value: float) -> MetricValue:
    number = float(value)
    if not np.isfinite(number):
        msg = "a metric produced a non-finite value"
        raise PerformanceReportError(msg)
    return MetricValue(value=round(number, 6), defined=True, reason=None)


def _undefined(reason: str) -> MetricValue:
    return MetricValue(value=None, defined=False, reason=reason)


def _warnings(
    *,
    overall: PerformanceMetrics,
    labels_verified: bool,
    segments: tuple[PerformanceSegment, ...],
) -> list[str]:
    warnings: list[str] = []
    if not labels_verified:
        warnings.append(
            "The labels are not marked verified. Treat these metrics as provisional until a "
            "confirmed outcome source is recorded."
        )
    for name in overall.undefined_metrics:
        metric = getattr(overall, name)
        warnings.append(f"Overall {name} is undefined: {metric.reason}.")
    partial_segments = sum(1 for segment in segments if segment.metrics.undefined_metrics)
    if partial_segments:
        warnings.append(
            f"{partial_segments} of {len(segments)} segments have at least one undefined "
            "metric; see the segment metric objects for the reason."
        )
    return warnings


def _caveats(*, labels_verified: bool) -> list[str]:
    caveats = [
        "This report contains aggregates only; no feature row, transaction identifier or raw "
        "label is written.",
        "Labels must be the final verified outcome for the same transaction and model score. "
        "Analyst review status is not a substitute for a confirmed outcome.",
        "Scores are the values served at prediction time and are not recomputed with a later "
        "model. A model change after scoring is not reflected in this report.",
        "PR-AUC and ROC-AUC are undefined when a group contains only one class. Undefined is "
        "reported as null, never as zero.",
        "This routine report is separate from the one-time final test evaluation. Do not "
        "select a model or tune a threshold on these labels.",
        "Small segments have wide uncertainty even when a metric is defined; use the reported "
        "row and class counts when interpreting differences.",
    ]
    if not labels_verified:
        caveats.append(
            "The operator did not mark the labels verified, so the report is explicitly "
            "provisional."
        )
    return caveats


def _validate_threshold(threshold: float) -> None:
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        msg = "threshold must be a number"
        raise PerformanceReportError(msg)
    if not np.isfinite(float(threshold)):
        msg = "threshold must be finite"
        raise PerformanceReportError(msg)
    if not 0.0 <= float(threshold) <= 1.0:
        msg = "threshold must be in [0, 1]"
        raise PerformanceReportError(msg)


def _validate_period(period: str) -> None:
    if period not in PERFORMANCE_PERIODS:
        msg = f"period must be one of {', '.join(PERFORMANCE_PERIODS)} (got {period!r})"
        raise PerformanceReportError(msg)


def _validate_label_source(label_source: str) -> None:
    if not isinstance(label_source, str) or not label_source.strip():
        msg = "label_source must be a non-empty description of the verified label source"
        raise PerformanceReportError(msg)
    if len(label_source.strip()) > 128:
        msg = "label_source must be at most 128 characters"
        raise PerformanceReportError(msg)
    if pd.Series([label_source]).str.contains(_CONTROL_CHARACTERS, regex=True).any():
        msg = "label_source must not contain control characters"
        raise PerformanceReportError(msg)


def _validate_sha256(value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        msg = "source_sha256 must be a lowercase SHA-256 hex digest"
        raise PerformanceReportError(msg)


def _validate_optional_window(
    start: datetime | None,
    end: datetime | None,
) -> None:
    for name, value in (("window_start", start), ("window_end", end)):
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            msg = f"{name} must include a timezone"
            raise PerformanceReportError(msg)
    if start is not None and end is not None and end < start:
        msg = "window_end must not be before window_start"
        raise PerformanceReportError(msg)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "METRIC_DEFINITIONS",
    "MODEL_VERSION_COLUMN",
    "PERFORMANCE_FEATURE_COLUMNS",
    "PERFORMANCE_PERIODS",
    "PERFORMANCE_REQUIRED_COLUMNS",
    "REPORT_SCHEMA_VERSION",
    "SCORED_AT_COLUMN",
    "SCORE_COLUMN",
    "TARGET_COLUMN",
    "LabeledEvaluationDataset",
    "MetricValue",
    "ModelVersionPerformance",
    "PerformanceEvaluationReport",
    "PerformanceMetrics",
    "PerformanceReportError",
    "PerformanceSchemaError",
    "PerformanceSegment",
    "build_performance_report",
    "load_performance_dataset",
    "period_bounds",
    "period_label",
    "prepare_performance_frame",
    "write_performance_report",
]
