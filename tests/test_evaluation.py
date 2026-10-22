"""Tests for the metric layer.

All inputs are small hand-built arrays, so every expected number can be verified by hand. The
tests pin the definitions the project relies on: PR-AUC as average precision, the
false-positive rate over the negative rows, per-class metrics from the same hard predictions,
and a cost-optimal threshold that never consults accuracy.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from fraud_risk.models.evaluation import (
    AlertCapacityPolicy,
    CostPolicy,
    MetricError,
    MetricsComparison,
    MinRecallPolicy,
    average_precision,
    binary_decisions,
    classification_metrics,
    cost_impact,
    expected_cost,
    majority_class_metrics,
    select_cost_threshold,
    select_operating_threshold,
)


def test_perfect_predictions_score_one() -> None:
    metrics = classification_metrics([0, 0, 1, 1], [0.01, 0.02, 0.98, 0.99], threshold=0.5)

    assert (metrics.true_negatives, metrics.false_positives) == (2, 0)
    assert (metrics.false_negatives, metrics.true_positives) == (0, 2)
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.pr_auc == pytest.approx(1.0)
    assert metrics.roc_auc == pytest.approx(1.0)
    assert metrics.false_positive_rate == 0.0


def test_all_negative_predictions_are_the_majority_class_case() -> None:
    metrics = classification_metrics([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4], threshold=0.5)

    assert (metrics.true_negatives, metrics.false_positives) == (2, 0)
    assert (metrics.false_negatives, metrics.true_positives) == (2, 0)
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0
    assert metrics.false_positive_rate == 0.0
    # AUCs are threshold-free: the ranking here is perfect even though nothing is flagged,
    # which is exactly why they must be read next to precision/recall at a real threshold.
    assert metrics.pr_auc == pytest.approx(1.0)
    assert metrics.roc_auc == pytest.approx(1.0)


def test_false_positive_rate_is_a_share_of_the_negative_rows() -> None:
    metrics = classification_metrics(
        [0, 0, 0, 0, 1],
        [0.9, 0.8, 0.1, 0.2, 0.7],
        threshold=0.5,
    )

    assert (metrics.true_negatives, metrics.false_positives) == (2, 2)
    assert metrics.false_positive_rate == pytest.approx(0.5)
    assert metrics.precision == pytest.approx(1 / 3)
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.f1 == pytest.approx(0.5)


def test_threshold_moves_the_confusion_matrix_but_not_the_aucs() -> None:
    labels = [0, 0, 1, 1, 0, 1]
    scores = [0.1, 0.35, 0.4, 0.6, 0.7, 0.9]

    low = classification_metrics(labels, scores, threshold=0.35)
    high = classification_metrics(labels, scores, threshold=0.8)

    assert low.true_positives > high.true_positives
    assert low.precision < high.precision
    assert low.recall > high.recall
    assert low.pr_auc == high.pr_auc
    assert low.roc_auc == high.roc_auc


def test_per_class_metrics_come_from_the_same_predictions() -> None:
    metrics = classification_metrics(
        [0, 0, 0, 1, 1, 1],
        [0.2, 0.4, 0.6, 0.5, 0.8, 0.9],
        threshold=0.5,
    )

    legitimate = metrics.per_class[0]
    fraud = metrics.per_class[1]

    assert legitimate.name == "legitimate"
    assert (legitimate.support, fraud.support) == (3, 3)
    assert legitimate.precision == pytest.approx(1.0)  # 2 TN, 0 FN
    assert legitimate.recall == pytest.approx(2 / 3)  # 1 FP out of 3 negatives
    assert fraud.precision == pytest.approx(3 / 4)  # 1 FP among 4 predicted fraud
    assert fraud.recall == pytest.approx(1.0)
    assert fraud.f1 == pytest.approx(6 / 7)
    assert metrics.precision == fraud.precision
    assert metrics.recall == fraud.recall

    record = metrics.to_dict()
    assert set(record["per_class"]) == {"legitimate", "fraud"}
    assert record["per_class"]["fraud"]["support"] == 3


def test_accuracy_is_available_but_not_serialised() -> None:
    metrics = classification_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], threshold=0.5)

    assert metrics.accuracy == pytest.approx(1.0)

    record = metrics.to_dict()
    assert "accuracy" not in json.dumps(record)
    assert set(record) == {
        "rows",
        "positives",
        "confusion_matrix",
        "precision",
        "recall",
        "f1",
        "pr_auc",
        "roc_auc",
        "false_positive_rate",
        "alerts",
        "alert_rate",
        "per_class",
    }
    assert set(record["confusion_matrix"]) == {
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
    }


def test_binary_decisions_are_a_separate_threshold_operation() -> None:
    scores = [0.1, 0.5, 0.9]

    decisions = binary_decisions(scores, threshold=0.5)

    assert decisions.dtype == np.int8
    assert decisions.tolist() == [0, 1, 1]
    # The probability ranking is unchanged; only the decision layer moves.
    low = classification_metrics([0, 1, 1], scores, threshold=0.5)
    high = classification_metrics([0, 1, 1], scores, threshold=0.8)
    assert low.pr_auc == high.pr_auc
    assert low.roc_auc == high.roc_auc
    assert low.true_positives != high.true_positives


def test_alert_count_and_rate_match_the_confusion_matrix() -> None:
    metrics = classification_metrics(
        [0, 0, 0, 1, 1, 1],
        [0.2, 0.4, 0.6, 0.5, 0.8, 0.9],
        threshold=0.5,
    )

    assert metrics.alerts == metrics.true_positives + metrics.false_positives
    assert metrics.alerts == 4
    assert metrics.alert_rate == pytest.approx(4 / 6)
    assert metrics.to_dict()["alerts"] == 4
    assert metrics.to_dict()["alert_rate"] == pytest.approx(4 / 6, abs=1e-8)


def test_majority_class_baseline_exposes_the_accuracy_trap() -> None:
    labels = [0] * 95 + [1] * 5

    metrics = majority_class_metrics(labels)

    assert metrics.accuracy == pytest.approx(0.95)
    assert metrics.recall == 0.0  # not one fraud found
    assert metrics.precision == 0.0
    assert metrics.f1 == 0.0
    assert metrics.pr_auc == pytest.approx(0.05)
    assert metrics.roc_auc == pytest.approx(0.5)
    assert metrics.per_class[1].recall == 0.0
    assert metrics.per_class[0].recall == 1.0


def test_comparison_reports_accuracy_only_next_to_the_baseline() -> None:
    labels = [0, 0, 0, 0, 1]
    model = classification_metrics(labels, [0.1, 0.2, 0.3, 0.4, 0.9], threshold=0.5)

    record = MetricsComparison(
        model=model,
        majority_class=majority_class_metrics(labels),
    ).to_dict()

    assert set(record) == {"model", "majority_class", "accuracy_comparison"}
    assert record["accuracy_comparison"]["model"] == pytest.approx(1.0)
    assert record["accuracy_comparison"]["majority_class"] == pytest.approx(0.8)
    assert "not a headline" in record["accuracy_comparison"]["note"]


def test_average_precision_matches_the_reported_pr_auc() -> None:
    labels = [0, 1, 0, 1, 0, 1]
    scores = [0.2, 0.8, 0.1, 0.4, 0.3, 0.9]

    metrics = classification_metrics(labels, scores, threshold=0.5)

    assert average_precision(labels, scores) == pytest.approx(metrics.pr_auc)


def test_a_threshold_above_the_score_range_predicts_nothing() -> None:
    metrics = classification_metrics([0, 1], [0.2, 0.8], threshold=1.5)

    assert metrics.true_positives == 0
    assert metrics.false_negatives == 1
    assert metrics.true_negatives == 1
    assert metrics.false_positives == 0


@pytest.mark.parametrize(
    ("labels", "scores", "threshold"),
    [
        ([0, 1], [0.1], 0.5),
        ([0, 2], [0.1, 0.9], 0.5),
        ([0, 0], [0.1, 0.9], 0.5),
        ([0, 1], [0.1, 0.9], -0.1),
        ([0, 1], [0.1, 0.9], float("nan")),
        ([0, 1], [0.1, float("nan")], 0.5),
        ([0, 1], [0.1, 1.5], 0.5),
    ],
)
def test_invalid_input_is_rejected(
    labels: list[int], scores: list[float], threshold: float
) -> None:
    with pytest.raises(MetricError):
        classification_metrics(labels, scores, threshold=threshold)


def test_empty_input_is_rejected() -> None:
    with pytest.raises(MetricError):
        classification_metrics([], [], threshold=0.5)
    with pytest.raises(MetricError):
        select_cost_threshold([], [], CostPolicy(100.0, 1.0))


# ---------------------------------------------------------------------------
# Cost policy and threshold rule
# ---------------------------------------------------------------------------
def test_cost_policy_requires_positive_finite_weights() -> None:
    for false_negative_cost, false_positive_cost in (
        (0.0, 1.0),
        (-1.0, 1.0),
        (100.0, 0.0),
        (float("nan"), 1.0),
        (100.0, float("inf")),
    ):
        with pytest.raises(MetricError):
            CostPolicy(false_negative_cost, false_positive_cost)

    with pytest.raises(MetricError):
        CostPolicy("100", 1.0)  # type: ignore[arg-type]


def test_expected_cost_uses_the_false_negative_and_false_positive_counts() -> None:
    metrics = classification_metrics(
        [0, 0, 0, 1, 1, 1],
        [0.2, 0.4, 0.6, 0.5, 0.8, 0.9],
        threshold=0.5,
    )
    policy = CostPolicy(false_negative_cost=100.0, false_positive_cost=1.0)

    # 1 false positive, no false negatives at this threshold.
    assert metrics.false_negatives == 0
    assert metrics.false_positives == 1
    assert expected_cost(metrics, policy) == pytest.approx(1.0)

    impact = cost_impact(metrics, policy)
    assert impact["false_negatives"]["count"] == 0
    assert impact["false_positives"]["count"] == 1
    assert impact["total_cost_units"] == pytest.approx(1.0)
    assert "meaning" in impact["false_negatives"]
    assert "meaning" in impact["false_positives"]


def test_cost_impact_counts_match_the_confusion_matrix() -> None:
    metrics = classification_metrics(
        [0, 0, 1, 1, 1],
        [0.9, 0.1, 0.8, 0.4, 0.7],
        threshold=0.5,
    )
    policy = CostPolicy(false_negative_cost=10.0, false_positive_cost=3.0)

    impact = cost_impact(metrics, policy)

    assert impact["false_negatives"]["count"] == metrics.false_negatives
    assert impact["false_positives"]["count"] == metrics.false_positives
    assert impact["false_negatives"]["cost_units"] == pytest.approx(metrics.false_negatives * 10.0)
    assert impact["false_positives"]["cost_units"] == pytest.approx(metrics.false_positives * 3.0)
    assert impact["total_cost_units"] == pytest.approx(expected_cost(metrics, policy))


def test_cost_threshold_minimises_expected_cost() -> None:
    labels = [0, 1, 0, 1, 1, 0]
    scores = [0.2, 0.9, 0.4, 0.8, 0.6, 0.1]
    policy = CostPolicy(false_negative_cost=10.0, false_positive_cost=1.0)

    threshold = select_cost_threshold(labels, scores, policy)
    metrics = classification_metrics(labels, scores, threshold=threshold)

    assert threshold == pytest.approx(0.6)
    assert metrics.false_negatives == 0
    assert metrics.false_positives == 0
    assert expected_cost(metrics, policy) == 0.0


def test_cost_threshold_breaks_ties_towards_higher_recall() -> None:
    # Both extreme cuts cost 1 unit: approve all three or flag only the top fraud.
    labels = [1, 0, 1]
    scores = [0.9, 0.5, 0.1]
    policy = CostPolicy(false_negative_cost=1.0, false_positive_cost=1.0)

    threshold = select_cost_threshold(labels, scores, policy)

    assert threshold == pytest.approx(0.1)
    metrics = classification_metrics(labels, scores, threshold=threshold)
    assert metrics.recall == 1.0


def test_cost_threshold_can_approve_everything_when_false_alarms_are_huge() -> None:
    labels = [0, 0, 0, 1]
    scores = [0.9, 0.8, 0.7, 0.4]
    policy = CostPolicy(false_negative_cost=1.0, false_positive_cost=100.0)

    threshold = select_cost_threshold(labels, scores, policy)

    assert threshold > max(scores)
    metrics = classification_metrics(labels, scores, threshold=threshold)
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 1
    assert expected_cost(metrics, policy) == 1.0


def test_binary_decisions_reject_invalid_scores_and_thresholds() -> None:
    for scores, threshold in (
        ([], 0.5),
        ([0.1, float("nan")], 0.5),
        ([0.1, 1.5], 0.5),
        ([0.1, 0.9], -0.1),
        ([0.1, 0.9], float("nan")),
    ):
        with pytest.raises(MetricError):
            binary_decisions(scores, threshold=threshold)


# ---------------------------------------------------------------------------
# Operating policies
# ---------------------------------------------------------------------------
def test_min_recall_policy_maximises_precision_above_the_target() -> None:
    labels = [0, 1, 0, 1, 1, 0]
    scores = [0.2, 0.9, 0.4, 0.8, 0.6, 0.1]
    policy = MinRecallPolicy(min_recall=2 / 3)

    decision = select_operating_threshold(labels, scores, policy)
    metrics = classification_metrics(labels, scores, threshold=decision.threshold)

    # 0.8 catches the two top frauds with no false positives and meets the target.
    assert decision.threshold == pytest.approx(0.8)
    assert metrics.recall >= policy.min_recall
    assert metrics.precision == 1.0
    assert metrics.alerts == 2
    assert decision.metrics.to_dict() == metrics.to_dict()
    assert decision.to_dict()["selected_on"] == "validation"
    assert decision.to_dict()["policy"]["kind"] == "min_recall"


def test_min_recall_prefers_precision_not_extra_recall() -> None:
    labels = [0, 1, 0, 0, 1, 0]
    scores = [0.9, 0.8, 0.75, 0.72, 0.7, 0.6]

    decision = select_operating_threshold(
        labels,
        scores,
        MinRecallPolicy(min_recall=0.5),
    )

    # 0.8 catches one fraud with one false positive. Lower cuts meet the target but add more
    # false positives, so they have lower precision even when they catch the second fraud.
    assert decision.threshold == pytest.approx(0.8)
    assert decision.metrics.recall == pytest.approx(0.5)
    assert decision.metrics.precision == pytest.approx(0.5)
    assert decision.metrics.alerts == 2


def test_alert_capacity_policy_never_exceeds_the_capacity() -> None:
    labels = [0, 1, 0, 1, 1, 0]
    scores = [0.2, 0.9, 0.4, 0.8, 0.6, 0.1]

    two = select_operating_threshold(
        labels,
        scores,
        AlertCapacityPolicy(max_alerts=2),
    )
    one = select_operating_threshold(
        labels,
        scores,
        AlertCapacityPolicy(max_alerts=1),
    )

    assert two.alerts == 2
    assert two.threshold == pytest.approx(0.8)
    assert two.metrics.recall == pytest.approx(2 / 3)
    assert one.alerts == 1
    assert one.threshold == pytest.approx(0.9)
    assert one.metrics.recall == pytest.approx(1 / 3)


def test_alert_capacity_zero_selects_no_alerts() -> None:
    labels = [0, 1, 0, 1]
    scores = [0.9, 0.8, 0.4, 0.1]

    decision = select_operating_threshold(
        labels,
        scores,
        AlertCapacityPolicy(max_alerts=0),
    )

    assert decision.threshold > max(scores)
    assert decision.alerts == 0
    assert decision.metrics.recall == 0.0
    assert decision.diagnostics["capacity"] == 0


def test_alert_capacity_never_splits_tied_scores() -> None:
    labels = [1, 0, 1]
    scores = [0.9, 0.9, 0.8]

    # The top tie contains two rows, so a capacity of one cannot be used without splitting
    # it. The policy approves everything rather than break the tie.
    one = select_operating_threshold(
        labels,
        scores,
        AlertCapacityPolicy(max_alerts=1),
    )
    two = select_operating_threshold(
        labels,
        scores,
        AlertCapacityPolicy(max_alerts=2),
    )

    assert one.alerts == 0
    assert one.threshold > max(scores)
    assert two.alerts == 2
    assert two.threshold == pytest.approx(0.9)


def test_alert_capacity_rate_is_rounded_down() -> None:
    policy = AlertCapacityPolicy(max_alert_rate=0.25)

    assert policy.capacity(6) == 1
    assert policy.capacity(7) == 1
    assert policy.capacity(8) == 2
    assert policy.to_dict()["basis"] == "alert_rate"


def test_operating_policy_parameters_are_validated() -> None:
    for value in (0.0, -0.1, 1.1, float("nan"), True, "0.8"):
        with pytest.raises(MetricError):
            MinRecallPolicy(value)  # type: ignore[arg-type]

    with pytest.raises(MetricError):
        AlertCapacityPolicy()
    with pytest.raises(MetricError):
        AlertCapacityPolicy(max_alerts=1, max_alert_rate=0.01)
    with pytest.raises(MetricError):
        AlertCapacityPolicy(max_alerts=-1)
    with pytest.raises(MetricError):
        AlertCapacityPolicy(max_alerts=1.5)  # type: ignore[arg-type]
    with pytest.raises(MetricError):
        AlertCapacityPolicy(max_alert_rate=-0.1)
    with pytest.raises(MetricError):
        AlertCapacityPolicy(max_alert_rate=1.1)
    with pytest.raises(MetricError):
        AlertCapacityPolicy(max_alert_rate=float("nan"))
    with pytest.raises(MetricError):
        AlertCapacityPolicy(max_alert_rate=0.5).capacity(0)


def test_cost_policy_decision_matches_the_cost_threshold() -> None:
    labels = [0, 1, 0, 1, 1, 0]
    scores = [0.2, 0.9, 0.4, 0.8, 0.6, 0.1]
    policy = CostPolicy(false_negative_cost=10.0, false_positive_cost=1.0)

    decision = select_operating_threshold(labels, scores, policy)

    assert decision.threshold == select_cost_threshold(labels, scores, policy)
    assert (
        decision.metrics.to_dict()
        == classification_metrics(
            labels,
            scores,
            threshold=decision.threshold,
        ).to_dict()
    )
    assert decision.to_dict()["policy"]["kind"] == "expected_cost"
    assert decision.to_dict()["metrics"]["alerts"] == decision.alerts


def test_operating_threshold_rejects_empty_input() -> None:
    with pytest.raises(MetricError):
        select_operating_threshold([], [], MinRecallPolicy(0.5))
    with pytest.raises(MetricError):
        select_operating_threshold([], [], AlertCapacityPolicy(max_alerts=1))
