"""Model evaluation: ranking metrics, operating policies and the metrics they imply.

Responsibility
    Turn labels, predicted probabilities and a threshold into the numbers the project
    reports: precision, recall, F1, the confusion matrix, per-class metrics, PR-AUC (ranking
    quality), ROC-AUC (secondary), the false-positive rate, the number of alerts, and the
    expected cost of the errors. Also provide the majority-class baseline that shows why
    accuracy is not a usable headline, and the validation-only threshold rules.

Probabilities versus decisions
    A model produces probabilities; a policy turns them into binary decisions. The two are
    deliberately separate here: :func:`binary_decisions` is the only conversion, PR-AUC and
    ROC-AUC use the probabilities, and precision/recall/F1/confusion/FPR/alerts use the
    decisions. A probability is not a decision until a threshold is supplied.

Rules
    - The model-selection criterion is business-oriented: expected cost, using an explicit
      :class:`CostPolicy`. PR-AUC is reported and breaks ties; accuracy never selects.
    - Operating policies are explicit and serialisable: expected cost, a minimum recall
      target, or a limited alert-review capacity. No threshold is universally optimal; each
      policy encodes a stated trade-off and is selected on validation data only.
    - Every number is computed from the arrays passed in. Nothing here invents, smooths or
      defaults a metric; an undefined input raises :class:`MetricError`.
    - ``accuracy`` exists as a property, and is serialised only inside the deliberate
      majority-class comparison. It is never a headline or a selection criterion.
    - Threshold selection sees validation data only; the caller is responsible for never
      passing test labels here before the model and threshold are frozen.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

# Inputs accept pandas Series, plain sequences and numpy arrays; they are normalised once.
LabelsInput = pd.Series | NDArray[Any] | Sequence[int]
ScoresInput = pd.Series | NDArray[Any] | Sequence[float]

#: Human-readable names for the two classes, used in the serialised per-class block.
CLASS_NAMES = {0: "legitimate", 1: "fraud"}


class MetricError(ValueError):
    """The labels, scores or cost policy cannot produce a meaningful metric."""


@dataclass(frozen=True)
class CostPolicy:
    """The relative cost of the two error types, in abstract cost units.

    A false negative is a fraudulent transaction that was approved (loss, chargeback, no
    review). A false positive is a legitimate transaction that was challenged or blocked
    (analyst time, customer friction). The weights are policy, not something a model learns:
    they are configuration, recorded with every run, and changing them re-ranks candidates.
    """

    false_negative_cost: float
    false_positive_cost: float

    def __post_init__(self) -> None:
        for name, value in (
            ("false_negative_cost", self.false_negative_cost),
            ("false_positive_cost", self.false_positive_cost),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                msg = f"{name} must be a number (got {value!r})"
                raise MetricError(msg)
            if not np.isfinite(value) or value <= 0:
                msg = f"{name} must be finite and strictly positive (got {value!r})"
                raise MetricError(msg)

    @property
    def kind(self) -> str:
        """Discriminator used in serialised decision records."""
        return "expected_cost"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "false_negative_cost": float(self.false_negative_cost),
            "false_positive_cost": float(self.false_positive_cost),
            "unit": "cost units, not currency",
            "source": "provisional policy; see docs/ARCHITECTURE.md 13.2",
        }


@dataclass(frozen=True)
class MinRecallPolicy:
    """Meet a minimum fraud recall target with the highest achievable precision.

    This is the "do not miss more than X% of fraud" policy. Among every distinct validation
    cut that reaches ``min_recall``, the selected cut has the highest precision; ties choose
    fewer alerts, then the higher threshold. The policy does not guarantee that the target is
    achievable in production, and it says nothing about the cost of the false positives it
    accepts.
    """

    min_recall: float

    def __post_init__(self) -> None:
        if isinstance(self.min_recall, bool) or not isinstance(self.min_recall, (int, float)):
            msg = f"min_recall must be a number (got {self.min_recall!r})"
            raise MetricError(msg)
        if not np.isfinite(self.min_recall) or not 0.0 < self.min_recall <= 1.0:
            msg = f"min_recall must be in (0, 1] (got {self.min_recall!r})"
            raise MetricError(msg)

    @property
    def kind(self) -> str:
        """Discriminator used in serialised decision records."""
        return "min_recall"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "min_recall": float(self.min_recall),
            "source": "operating policy; selected on validation data only",
        }


@dataclass(frozen=True)
class AlertCapacityPolicy:
    """Use at most a fixed review capacity, maximising recall within it.

    The capacity is either an absolute validation alert count (``max_alerts``) or a share of
    validation rows (``max_alert_rate``), with the rate rounded down. Exactly one is
    supplied. Among cuts that fit the capacity, the selected cut has the highest recall;
    ties choose higher precision, fewer alerts, then the higher threshold. A group of tied
    scores is never split, so a capacity can be under-used rather than broken.

    This is a review-queue constraint, not a claim that the capacity will hold on another
    volume. The test evaluation reports the frozen threshold's alert count next to the
    validation capacity; it never re-tunes to fit the test volume.
    """

    max_alerts: int | None = None
    max_alert_rate: float | None = None

    def __post_init__(self) -> None:
        if self.max_alerts is None and self.max_alert_rate is None:
            msg = "alert capacity needs max_alerts or max_alert_rate"
            raise MetricError(msg)
        if self.max_alerts is not None and self.max_alert_rate is not None:
            msg = "supply only one of max_alerts and max_alert_rate"
            raise MetricError(msg)

        if self.max_alerts is not None:
            if isinstance(self.max_alerts, bool) or not isinstance(self.max_alerts, int):
                msg = f"max_alerts must be an int (got {self.max_alerts!r})"
                raise MetricError(msg)
            if self.max_alerts < 0:
                msg = f"max_alerts must be non-negative (got {self.max_alerts!r})"
                raise MetricError(msg)
        else:
            rate = self.max_alert_rate
            if isinstance(rate, bool) or not isinstance(rate, (int, float)):
                msg = f"max_alert_rate must be a number (got {rate!r})"
                raise MetricError(msg)
            if not np.isfinite(rate) or not 0.0 <= rate <= 1.0:
                msg = f"max_alert_rate must be in [0, 1] (got {rate!r})"
                raise MetricError(msg)

    @property
    def kind(self) -> str:
        """Discriminator used in serialised decision records."""
        return "alert_capacity"

    def capacity(self, rows: int) -> int:
        """Return the alert allowance for a validation split of ``rows`` rows."""
        if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
            msg = f"rows must be a positive int (got {rows!r})"
            raise MetricError(msg)
        if self.max_alerts is not None:
            return int(self.max_alerts)
        rate = self.max_alert_rate
        if rate is None:  # pragma: no cover - __post_init__ guarantees exactly one
            msg = "alert capacity is not configured"
            raise MetricError(msg)
        return int(np.floor(float(rate) * rows))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "max_alerts": self.max_alerts,
            "max_alert_rate": self.max_alert_rate,
            "basis": "absolute_alert_count" if self.max_alerts is not None else "alert_rate",
            "source": "operating policy; selected on validation data only",
        }


#: Any policy that can turn validation probabilities into a frozen operating threshold.
ThresholdPolicy = CostPolicy | MinRecallPolicy | AlertCapacityPolicy


@dataclass(frozen=True)
class ClassMetrics:
    """Precision, recall and F1 for one class, computed from the same hard predictions."""

    label: int
    name: str
    precision: float
    recall: float
    f1: float
    support: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "name": self.name,
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "f1": round(self.f1, 6),
            "support": self.support,
        }


@dataclass(frozen=True)
class ClassificationMetrics:
    """Binary-classification metrics at one threshold, plus the threshold-free AUCs."""

    rows: int
    positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int
    precision: float
    recall: float
    f1: float
    pr_auc: float
    roc_auc: float
    false_positive_rate: float
    per_class: tuple[ClassMetrics, ClassMetrics]

    @property
    def accuracy(self) -> float:
        """Share of correct hard predictions.

        Deliberately a property rather than a serialised field: it is meaningful only next
        to the majority-class baseline, never on its own.
        """
        return (self.true_positives + self.true_negatives) / self.rows

    @property
    def alerts(self) -> int:
        """Number of transactions the policy would send for review or block."""
        return self.true_positives + self.false_positives

    @property
    def alert_rate(self) -> float:
        """Share of rows that become alerts."""
        return self.alerts / self.rows

    def to_dict(self) -> dict[str, Any]:
        """Serialise the reported metrics. Accuracy is intentionally absent."""
        return {
            "rows": self.rows,
            "positives": self.positives,
            "confusion_matrix": {
                "true_negative": self.true_negatives,
                "false_positive": self.false_positives,
                "false_negative": self.false_negatives,
                "true_positive": self.true_positives,
            },
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "f1": round(self.f1, 6),
            "pr_auc": round(self.pr_auc, 6),
            "roc_auc": round(self.roc_auc, 6),
            "false_positive_rate": round(self.false_positive_rate, 6),
            "alerts": self.alerts,
            "alert_rate": round(self.alert_rate, 8),
            "per_class": {metrics.name: metrics.to_dict() for metrics in self.per_class},
        }


@dataclass(frozen=True)
class ThresholdDecision:
    """A frozen threshold plus the validation evidence and rule that produced it."""

    threshold: float
    policy: ThresholdPolicy
    rule: str
    metrics: ClassificationMetrics
    diagnostics: dict[str, Any]

    @property
    def alerts(self) -> int:
        """Validation alerts at the selected threshold."""
        return self.metrics.alerts

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_on": "validation",
            "policy": self.policy.to_dict(),
            "threshold": float(self.threshold),
            "rule": self.rule,
            "metrics": self.metrics.to_dict(),
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class MetricsComparison:
    """A model's metrics next to the majority-class baseline on the same rows."""

    model: ClassificationMetrics
    majority_class: ClassificationMetrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.to_dict(),
            "majority_class": self.majority_class.to_dict(),
            "accuracy_comparison": {
                "model": round(self.model.accuracy, 6),
                "majority_class": round(self.majority_class.accuracy, 6),
                "note": (
                    "Accuracy is shown only to expose the majority-class trap; it is not a "
                    "headline metric, a selection criterion or a promotion gate."
                ),
            },
        }


def binary_decisions(y_score: ScoresInput, *, threshold: float) -> NDArray[np.int8]:
    """Turn probabilities into hard predictions; this is the only threshold comparison.

    A row is predicted positive when ``score >= threshold``. The threshold may exceed 1,
    which predicts nothing (used by a policy whose cheapest cut is "approve everything").
    Probability ranking metrics never call this function; decision metrics always do.
    """
    scores = _validate_scores(y_score)
    cutoff = _validate_threshold(threshold)
    return (scores >= cutoff).astype("int8")


def classification_metrics(
    y_true: LabelsInput,
    y_score: ScoresInput,
    *,
    threshold: float,
) -> ClassificationMetrics:
    """Compute every reported metric from labels, positive-class scores and a threshold.

    ``y_score`` must be a probability-like score in ``[0, 1]``. Decisions come exclusively
    from :func:`binary_decisions`; PR-AUC and ROC-AUC come from the probabilities. Both
    classes must be present: those ranking metrics are undefined otherwise, and returning zero
    would hide that.
    """
    labels, scores = _validate_arrays(y_true, y_score)
    predictions = binary_decisions(scores, threshold=threshold)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    true_negatives, false_positives, false_negatives, true_positives = (
        int(value) for value in matrix.ravel()
    )

    per_class = _per_class_metrics(labels, predictions)
    fraud = per_class[1]
    false_positive_rate = _ratio(false_positives, false_positives + true_negatives)

    return ClassificationMetrics(
        rows=int(labels.size),
        positives=int(true_positives + false_negatives),
        true_negatives=true_negatives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        true_positives=true_positives,
        precision=fraud.precision,
        recall=fraud.recall,
        f1=fraud.f1,
        pr_auc=average_precision(labels, scores),
        roc_auc=float(roc_auc_score(labels, scores)),
        false_positive_rate=false_positive_rate,
        per_class=per_class,
    )


def average_precision(
    y_true: LabelsInput,
    y_score: ScoresInput,
) -> float:
    """PR-AUC (average precision), the ranking metric reported for every candidate."""
    labels, scores = _validate_arrays(y_true, y_score)
    return float(average_precision_score(labels, scores))


def majority_class_metrics(y_true: LabelsInput) -> ClassificationMetrics:
    """Metrics for "never fraud", the baseline every model must beat.

    All scores are zero, so every row is predicted negative. The resulting accuracy is the
    negative prevalence - which is exactly why 99% accuracy can mean a useless model. PR-AUC
    reduces to the prevalence and ROC-AUC is 0.5.
    """
    labels = np.asarray(y_true, dtype="int8")
    return classification_metrics(labels, np.zeros(labels.size, dtype="float64"), threshold=0.5)


def expected_cost(metrics: ClassificationMetrics, policy: CostPolicy) -> float:
    """Expected cost of the errors in ``metrics`` under ``policy``, in cost units."""
    return metrics.false_negatives * float(
        policy.false_negative_cost
    ) + metrics.false_positives * float(policy.false_positive_cost)


def cost_impact(metrics: ClassificationMetrics, policy: CostPolicy) -> dict[str, Any]:
    """Machine-readable error counts plus their cost and operational meaning."""
    false_negative_cost = metrics.false_negatives * float(policy.false_negative_cost)
    false_positive_cost = metrics.false_positives * float(policy.false_positive_cost)
    return {
        "false_negatives": {
            "count": metrics.false_negatives,
            "unit_cost": float(policy.false_negative_cost),
            "cost_units": round(false_negative_cost, 2),
            "meaning": (
                "fraudulent transactions approved: direct loss or chargeback, and the "
                "fraud is never reviewed"
            ),
        },
        "false_positives": {
            "count": metrics.false_positives,
            "unit_cost": float(policy.false_positive_cost),
            "cost_units": round(false_positive_cost, 2),
            "meaning": (
                "legitimate transactions challenged or blocked: analyst time, customer "
                "friction and lost good business"
            ),
        },
        "total_cost_units": round(false_negative_cost + false_positive_cost, 2),
    }


@dataclass(frozen=True)
class _ThresholdSweep:
    """Every distinct cut point and the counts it produces, in descending score order."""

    thresholds: NDArray[np.float64]
    true_positives: NDArray[np.int64]
    false_positives: NDArray[np.int64]
    positives: int
    rows: int
    approve_none_threshold: float

    @property
    def false_negatives(self) -> NDArray[np.int64]:
        return (self.positives - self.true_positives).astype("int64")

    @property
    def alerts(self) -> NDArray[np.int64]:
        return (self.true_positives + self.false_positives).astype("int64")

    @property
    def recall(self) -> NDArray[np.float64]:
        return (self.true_positives / self.positives).astype("float64")

    @property
    def precision(self) -> NDArray[np.float64]:
        alerts = self.alerts.astype("float64")
        values: NDArray[np.float64] = np.divide(
            self.true_positives.astype("float64"),
            alerts,
            out=np.zeros(alerts.shape, dtype="float64"),
            where=alerts > 0,
        )
        return values


def _threshold_sweep(
    labels: NDArray[np.int8],
    scores: NDArray[np.float64],
) -> _ThresholdSweep:
    """Sweep every distinct score cut, never splitting a group of tied scores."""
    positives = int(labels.sum())
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    true_positives = np.cumsum(sorted_labels).astype("int64")
    false_positives = np.cumsum(1 - sorted_labels).astype("int64")

    # Only the last row of a group of tied scores is a distinct cut point.
    group_end = np.empty(sorted_scores.size, dtype=bool)
    group_end[:-1] = sorted_scores[1:] != sorted_scores[:-1]
    group_end[-1] = True

    return _ThresholdSweep(
        thresholds=sorted_scores[group_end].astype("float64"),
        true_positives=true_positives[group_end],
        false_positives=false_positives[group_end],
        positives=positives,
        rows=int(labels.size),
        approve_none_threshold=float(np.nextafter(float(scores.max()), np.inf)),
    )


def _select_cost_threshold(
    sweep: _ThresholdSweep,
    policy: CostPolicy,
) -> tuple[float, dict[str, Any]]:
    """Return the cost-minimising threshold and its diagnostics."""
    costs = sweep.false_negatives * float(
        policy.false_negative_cost
    ) + sweep.false_positives * float(policy.false_positive_cost)
    best_cost = float(costs.min())
    # Thresholds descend, so the last minimum is the lowest threshold (highest recall).
    at_best = np.flatnonzero(costs == best_cost)
    threshold = float(sweep.thresholds[int(at_best[-1])])

    approve_all_cost = sweep.positives * float(policy.false_negative_cost)
    approves_none = approve_all_cost < best_cost
    if approves_none:
        threshold = sweep.approve_none_threshold

    return threshold, {
        "candidate_thresholds": int(sweep.thresholds.size),
        "minimum_expected_cost_units": round(best_cost, 6),
        "approve_all_cost_units": round(approve_all_cost, 6),
        "selected_approves_none": approves_none,
        "tie_break": "equal cost chooses the lower threshold (higher recall)",
    }


def select_cost_threshold(
    y_true: LabelsInput,
    y_score: ScoresInput,
    policy: CostPolicy,
) -> float:
    """Return the threshold that minimises expected cost on the given (validation) rows.

    The sweep covers every distinct score and the degenerate "approve everything" cut above
    the maximum score. Ties between thresholds are resolved towards the lower threshold
    (higher recall): at equal expected cost, taking the extra alert protects against the
    costlier error. The rule is fixed before the test split is scored; see
    ``models.training`` and ``models.final_evaluation``.
    """
    labels, scores = _validate_arrays(y_true, y_score)
    threshold, _ = _select_cost_threshold(_threshold_sweep(labels, scores), policy)
    return threshold


def select_operating_threshold(
    y_true: LabelsInput,
    y_score: ScoresInput,
    policy: ThresholdPolicy,
) -> ThresholdDecision:
    """Select a threshold on ``(y_true, y_score)`` under an explicit operating policy.

    The caller must pass validation rows. The returned decision records the exact threshold,
    the policy, the selection rule and the validation metrics. Every candidate cut is a
    distinct score value, so a group of tied scores is never split. The selected threshold
    is a policy choice, not a universally optimal value; test data must not reach this
    function before the decision is frozen.
    """
    labels, scores = _validate_arrays(y_true, y_score)
    sweep = _threshold_sweep(labels, scores)

    if isinstance(policy, CostPolicy):
        threshold, diagnostics = _select_cost_threshold(sweep, policy)
        rule = (
            "minimum expected validation cost; equal costs choose the lower threshold "
            "(higher recall); predicting no fraud is used only when strictly cheaper"
        )
    elif isinstance(policy, MinRecallPolicy):
        threshold, diagnostics, rule = _select_min_recall_threshold(sweep, policy)
    else:
        # ThresholdPolicy is a closed union; mypy narrows this to AlertCapacityPolicy.
        threshold, diagnostics, rule = _select_capacity_threshold(sweep, policy)

    metrics = classification_metrics(labels, scores, threshold=threshold)
    diagnostics = {
        **diagnostics,
        "alerts": metrics.alerts,
        "alert_rate": round(metrics.alert_rate, 8),
    }
    return ThresholdDecision(
        threshold=threshold,
        policy=policy,
        rule=rule,
        metrics=metrics,
        diagnostics=diagnostics,
    )


def _select_min_recall_threshold(
    sweep: _ThresholdSweep,
    policy: MinRecallPolicy,
) -> tuple[float, dict[str, Any], str]:
    """Highest precision among cuts that meet the recall target."""
    feasible = np.flatnonzero(sweep.recall >= float(policy.min_recall))
    if feasible.size == 0:  # pragma: no cover - predicting all rows reaches recall 1.0
        msg = f"no validation threshold reaches min_recall={policy.min_recall!r}"
        raise MetricError(msg)

    precision = sweep.precision
    alerts = sweep.alerts
    # Two cuts with the same precision: the one with fewer alerts is cheaper to operate.
    best_index = max(
        (int(index) for index in feasible),
        key=lambda index: (
            float(precision[index]),
            -int(alerts[index]),
            float(sweep.thresholds[index]),
        ),
    )
    threshold = float(sweep.thresholds[best_index])
    return (
        threshold,
        {
            "candidate_thresholds": int(sweep.thresholds.size),
            "feasible_thresholds": int(feasible.size),
            "minimum_recall": float(policy.min_recall),
            "achieved_recall": float(sweep.recall[best_index]),
            "selected_approves_none": False,
            "tie_break": "equal precision chooses fewer alerts, then the higher threshold",
        },
        (
            "highest validation precision among thresholds meeting the minimum recall target; "
            "equal precision chooses fewer alerts, then the higher threshold"
        ),
    )


def _select_capacity_threshold(
    sweep: _ThresholdSweep,
    policy: AlertCapacityPolicy,
) -> tuple[float, dict[str, Any], str]:
    """Highest recall among cuts that fit the review capacity."""
    capacity = policy.capacity(sweep.rows)
    feasible = np.flatnonzero(sweep.alerts <= capacity)

    # "Approve everything" is always feasible when no cut catches a fraud; it has zero
    # alerts, so it wins the fewer-alerts tie-break against an uninformative alert.
    threshold = sweep.approve_none_threshold
    best_key = (0.0, 0.0, 0, sweep.approve_none_threshold)
    recall = sweep.recall
    precision = sweep.precision
    alerts = sweep.alerts
    for raw_index in feasible:
        index = int(raw_index)
        key = (
            float(recall[index]),
            float(precision[index]),
            -int(alerts[index]),
            float(sweep.thresholds[index]),
        )
        if key > best_key:
            best_key = key
            threshold = float(sweep.thresholds[index])

    return (
        threshold,
        {
            "candidate_thresholds": int(sweep.thresholds.size),
            "capacity": capacity,
            "capacity_basis": "absolute_alert_count"
            if policy.max_alerts is not None
            else "alert_rate",
            "feasible_thresholds": int(feasible.size),
            "selected_approves_none": bool(threshold == sweep.approve_none_threshold),
            "note": (
                "a group of tied scores is never split, so alerts can be below capacity; the "
                "capacity is derived from validation volume and is not re-tuned on test"
            ),
            "tie_break": (
                "equal recall chooses higher precision, fewer alerts, then higher threshold"
            ),
        },
        (
            "highest validation recall within the alert capacity; equal recall chooses higher "
            "precision, fewer alerts, then the higher threshold"
        ),
    )


def _per_class_metrics(
    labels: NDArray[np.int8],
    predictions: NDArray[np.int8],
) -> tuple[ClassMetrics, ClassMetrics]:
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=[0, 1],
        zero_division=0,
    )
    return (
        ClassMetrics(
            label=0,
            name=CLASS_NAMES[0],
            precision=float(precision[0]),
            recall=float(recall[0]),
            f1=float(f1[0]),
            support=int(support[0]),
        ),
        ClassMetrics(
            label=1,
            name=CLASS_NAMES[1],
            precision=float(precision[1]),
            recall=float(recall[1]),
            f1=float(f1[1]),
            support=int(support[1]),
        ),
    )


def _validate_scores(y_score: ScoresInput) -> NDArray[np.float64]:
    """Validate a probability vector independently of any labels."""
    scores = np.asarray(y_score, dtype="float64")
    if scores.ndim != 1 or scores.size == 0:
        msg = "scores must be a non-empty one-dimensional sequence"
        raise MetricError(msg)
    if not np.isfinite(scores).all():
        msg = "scores must all be finite"
        raise MetricError(msg)
    if (scores < 0.0).any() or (scores > 1.0).any():
        msg = "scores must be probabilities in [0, 1]"
        raise MetricError(msg)
    return scores


def _validate_threshold(threshold: float) -> float:
    """Validate the one number that separates a probability from a decision."""
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not np.isfinite(threshold)
        or threshold < 0.0
    ):
        msg = f"threshold must be a finite non-negative number (got {threshold!r})"
        raise MetricError(msg)
    return float(threshold)


def _validate_arrays(
    y_true: LabelsInput,
    y_score: ScoresInput,
) -> tuple[NDArray[np.int8], NDArray[np.float64]]:
    labels = np.asarray(y_true, dtype="int8")
    scores = _validate_scores(y_score)

    if labels.ndim != 1 or labels.size == 0:
        msg = "labels must be a non-empty one-dimensional sequence"
        raise MetricError(msg)
    if scores.shape != labels.shape:
        msg = f"labels and scores must have the same shape (got {labels.shape} and {scores.shape})"
        raise MetricError(msg)
    if not np.isin(labels, [0, 1]).all():
        msg = "labels must contain only 0 and 1"
        raise MetricError(msg)
    if labels.min() == labels.max():
        msg = "both classes must be present; metrics are undefined for a single-class sample"
        raise MetricError(msg)

    return labels, scores


def _ratio(numerator: float | int, denominator: float | int) -> float:
    """Return ``0.0`` instead of dividing by zero (no predicted positives, etc.)."""
    return float(numerator) / float(denominator) if denominator else 0.0
