"""Final evaluation: freeze a model and an operating threshold, then score test once.

Responsibility
    Consume the winning comparison bundle, select an operating threshold on validation under
    an explicit policy, copy the bundle into an immutable local artifact directory, record
    the decision *before* any test label is used, and score the test split exactly once.

Why a separate stage
    Model selection (``models.training``) answers "which candidate?". The operating policy
    answers "where do we cut the alerts?". They are deliberately separate decisions: a
    different review capacity or recall target changes the threshold, not the fitted model.
    The model was selected under validation expected cost; this stage does not re-select it.

Protocol
    1. Validate the bundle schema, feature columns and cost policy.
    2. Score validation once with the frozen pipeline.
    3. Select a threshold under :class:`CostPolicy`, :class:`MinRecallPolicy` or
       :class:`AlertCapacityPolicy` using validation labels only.
    4. Copy the bundle to ``frozen_model.joblib`` and write ``decision.json`` with the model
       SHA-256, policy, exact threshold, validation metrics and split definition.
    5. Only then read the test labels and score test once. Write ``metrics.json`` and
       ``run.json``; ``decision.json`` is not rewritten.

Artifacts (all git-ignored under ``artifacts/final_evaluation/``)
    ``decision.json`` - the frozen decision, written before test evaluation.
    ``frozen_model.joblib`` - byte-identical copy of the model bundle that was evaluated.
    ``metrics.json`` - validation and test probabilities/decisions, metrics, per-class
    metrics, cost impact and the majority-class baseline.
    ``run.json`` - evaluation configuration and provenance, including the full split record.

Limits
    A threshold is a policy choice, not a universally optimal value, and a high metric on
    one temporal split is not evidence that the system is ready for real financial decisions.
    The reliability/Brier calibration comparison, MLflow tracking of this evaluator and the
    three-way review/block policy remain later work. The test split was already scored by the
    comparison stage, so this final report is a repeat evaluation, not a pristine holdout.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from numpy.typing import NDArray

from fraud_risk.data.dataset import DEFAULT_DATASET_PATH
from fraud_risk.data.errors import DataError
from fraud_risk.data.ingestion import load_csv
from fraud_risk.data.validation import validate_dataset
from fraud_risk.features.engineering import FeatureSchemaError
from fraud_risk.models.bundle import (
    DEFAULT_COMPARISON_MODEL_PATH,
    FrozenModel,
    ModelBundleError,
    load_model_bundle,
    sha256_file,
)
from fraud_risk.models.evaluation import (
    AlertCapacityPolicy,
    ClassificationMetrics,
    CostPolicy,
    MetricError,
    MetricsComparison,
    MinRecallPolicy,
    ThresholdDecision,
    ThresholdPolicy,
    classification_metrics,
    cost_impact,
    majority_class_metrics,
    select_operating_threshold,
)
from fraud_risk.models.splitting import (
    EXIT_INVALID,
    EXIT_MISSING,
    EXIT_OK,
    DatasetSplit,
    DatasetSplits,
    SplitConfig,
    SplitError,
    SplitStrategy,
    split_dataset,
)

DEFAULT_MODEL_PATH = DEFAULT_COMPARISON_MODEL_PATH
DEFAULT_OUTPUT_DIR = Path("artifacts") / "final_evaluation"
DECISION_FILENAME = "decision.json"
METRICS_FILENAME = "metrics.json"
RUN_FILENAME = "run.json"
FROZEN_MODEL_FILENAME = "frozen_model.joblib"
SCHEMA_VERSION = 1


class FinalEvaluationError(ValueError):
    """The frozen model, split or policy cannot produce an honest final evaluation."""


@dataclass(frozen=True)
class FinalEvaluation:
    """The frozen decision, both evaluations and the artifact location."""

    model: FrozenModel
    policy: ThresholdPolicy
    decision: ThresholdDecision
    validation: MetricsComparison
    test: MetricsComparison
    impact_policy: CostPolicy
    artifact_dir: Path
    frozen_model_path: Path


def load_bundle(path: Path = DEFAULT_MODEL_PATH) -> FrozenModel:
    """Load and validate the comparison bundle, using this stage's error type."""
    try:
        return load_model_bundle(path)
    except ModelBundleError as exc:
        raise FinalEvaluationError(str(exc)) from exc


def evaluate_final(
    splits: DatasetSplits,
    model: FrozenModel,
    policy: ThresholdPolicy,
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    impact_policy: CostPolicy | None = None,
    dataset_path: Path | None = None,
) -> FinalEvaluation:
    """Select on validation, freeze the decision and score test once.

    Raises :class:`FinalEvaluationError` when the bundle schema or feature contract does not
    match the split, validation cannot support a decision, or the test split is unusable.
    The test split is not inspected until ``decision.json`` and ``frozen_model.joblib`` have
    been written.
    """
    if tuple(model.feature_columns) != tuple(splits.feature_columns):
        msg = (
            "model bundle feature columns do not match the split schema: "
            f"{len(model.feature_columns)} vs {len(splits.feature_columns)} columns"
        )
        raise FinalEvaluationError(msg)

    _require_both_classes(splits.validation, "validation")
    if impact_policy is None:
        impact_policy = policy if isinstance(policy, CostPolicy) else model.cost_policy

    # Everything up to the test split is validation-only.
    validation_scores = _predict_probabilities(
        model.pipeline,
        splits.validation.X,
        context="validation",
    )
    decision = select_operating_threshold(splits.validation.y, validation_scores, policy)
    validation = MetricsComparison(
        model=decision.metrics,
        majority_class=majority_class_metrics(splits.validation.y),
    )

    artifact_dir = Path(output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    frozen_model_path = artifact_dir / FROZEN_MODEL_FILENAME
    shutil.copyfile(model.path, frozen_model_path)
    frozen_hash = sha256_file(frozen_model_path)
    if frozen_hash != model.sha256:
        msg = "frozen model copy does not match the loaded bundle"
        raise FinalEvaluationError(msg)

    created_at = datetime.now(tz=UTC)
    decision_record = _decision_record(
        model=model,
        policy=policy,
        decision=decision,
        validation=validation,
        validation_scores=validation_scores,
        impact_policy=impact_policy,
        splits=splits,
        created_at=created_at,
        frozen_hash=frozen_hash,
    )
    decision_path = artifact_dir / DECISION_FILENAME
    _write_json(decision_path, decision_record)
    decision_hash = sha256_file(decision_path)

    # The decision is frozen. Only now may test labels or test features be used.
    _require_both_classes(splits.test, "test")
    test_scores = _predict_probabilities(model.pipeline, splits.test.X, context="test")
    test = MetricsComparison(
        model=classification_metrics(
            splits.test.y,
            test_scores,
            threshold=decision.threshold,
        ),
        majority_class=majority_class_metrics(splits.test.y),
    )

    result = FinalEvaluation(
        model=model,
        policy=policy,
        decision=decision,
        validation=validation,
        test=test,
        impact_policy=impact_policy,
        artifact_dir=artifact_dir,
        frozen_model_path=frozen_model_path,
    )
    _write_json(
        artifact_dir / METRICS_FILENAME,
        _metrics_record(
            result=result,
            validation_scores=validation_scores,
            test_scores=test_scores,
            created_at=created_at,
        ),
    )
    _write_json(
        artifact_dir / RUN_FILENAME,
        _run_record(
            result=result,
            splits=splits,
            created_at=created_at,
            dataset_path=dataset_path,
            frozen_hash=frozen_hash,
            decision_hash=decision_hash,
        ),
    )
    return result


def _decision_record(
    *,
    model: FrozenModel,
    policy: ThresholdPolicy,
    decision: ThresholdDecision,
    validation: MetricsComparison,
    validation_scores: NDArray[np.float64],
    impact_policy: CostPolicy,
    splits: DatasetSplits,
    created_at: datetime,
    frozen_hash: str,
) -> dict[str, Any]:
    record = decision.to_dict()
    record.update(
        {
            "schema_version": SCHEMA_VERSION,
            "created_at": created_at.isoformat(),
            "status": "frozen before test evaluation",
            "model": {
                "source": str(model.path),
                "frozen_copy": FROZEN_MODEL_FILENAME,
                "sha256": model.sha256,
                "frozen_copy_sha256": frozen_hash,
                "seed": model.seed,
                "family": model.metadata.get("family"),
                "params": model.metadata.get("params"),
                "imbalance": model.metadata.get("imbalance"),
                "scaling": model.metadata.get("scaling"),
                "calibration": model.metadata.get("calibration"),
                "bundle_threshold": model.threshold,
                "bundle_threshold_role": (
                    "comparison-stage threshold recorded with the bundle; the operating "
                    "threshold above is re-selected under the stated policy"
                ),
            },
            "impact_cost_policy": impact_policy.to_dict(),
            "validation_comparison": validation.to_dict(),
            "validation_probability_summary": _probability_summary(validation_scores),
            "selection_split": _selection_split_record(splits),
            "test_data": (
                "not inspected for selection; the test split is scored once after this file "
                "is written"
            ),
        }
    )
    return record


def _selection_split_record(splits: DatasetSplits) -> dict[str, Any]:
    """Describe the partition without reading the test labels before the freeze."""
    return {
        "strategy": splits.config.strategy.value,
        "seed": splits.config.seed,
        "fractions": {
            "train": splits.config.train_fraction,
            "validation": splits.config.validation_fraction,
            "test": splits.config.test_fraction,
        },
        "row_count": splits.row_count,
        "feature_columns": list(splits.feature_columns),
        "resampling": "none",
        "train": splits.train.describe(),
        "validation": splits.validation.describe(),
        "test": "not inspected before the decision was frozen; see run.json",
    }


def _metrics_record(
    *,
    result: FinalEvaluation,
    validation_scores: NDArray[np.float64],
    test_scores: NDArray[np.float64],
    created_at: datetime,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": created_at.isoformat(),
        "policy": result.policy.to_dict(),
        "threshold": result.decision.threshold,
        "rule": result.decision.rule,
        "selected_on": "validation",
        "impact_cost_policy": result.impact_policy.to_dict(),
        "validation": _evaluation_record(
            result.validation,
            validation_scores,
            result.decision.threshold,
            result.impact_policy,
        ),
        "test": _evaluation_record(
            result.test,
            test_scores,
            result.decision.threshold,
            result.impact_policy,
        ),
        "frozen_model": {
            "file": FROZEN_MODEL_FILENAME,
            "sha256": result.model.sha256,
        },
        "notes": {
            "probability": (
                "probability_summary describes model outputs; it is not a decision. "
                "decision_summary is the binary alert set at the frozen threshold."
            ),
            "test_set": (
                "the test split was scored once in this stage, after decision.json was "
                "written; the comparison stage also scored it once, so this is not a "
                "pristine holdout for future tuning"
            ),
            "status": (
                "single binary operating cut for evaluation; the three-way review/block "
                "policy and production readiness remain out of scope"
            ),
        },
    }


def _evaluation_record(
    comparison: MetricsComparison,
    scores: NDArray[np.float64],
    threshold: float,
    impact_policy: CostPolicy,
) -> dict[str, Any]:
    record = comparison.to_dict()
    record["probability_summary"] = _probability_summary(scores)
    record["decision_summary"] = {
        "threshold": threshold,
        "rule": "predict fraud when score >= threshold",
        "alerts": comparison.model.alerts,
        "alert_rate": round(comparison.model.alert_rate, 8),
        "note": (
            "a single binary review/block cut for evaluation; the three-way review/block "
            "policy remains M6 work"
        ),
    }
    record["model_cost_impact"] = cost_impact(comparison.model, impact_policy)
    return record


def _probability_summary(scores: NDArray[np.float64]) -> dict[str, Any]:
    """Summarise probabilities without storing row-level scores."""
    if scores.ndim != 1 or scores.size == 0 or not np.isfinite(scores).all():
        msg = "probabilities must be a non-empty finite vector"
        raise FinalEvaluationError(msg)
    quantiles = np.quantile(scores, [0.5, 0.9, 0.95, 0.99])
    return {
        "count": int(scores.size),
        "minimum": round(float(scores.min()), 8),
        "mean": round(float(scores.mean()), 8),
        "p50": round(float(quantiles[0]), 8),
        "p90": round(float(quantiles[1]), 8),
        "p95": round(float(quantiles[2]), 8),
        "p99": round(float(quantiles[3]), 8),
        "maximum": round(float(scores.max()), 8),
        "note": "probabilities are model outputs, not decisions",
    }


def _run_record(
    *,
    result: FinalEvaluation,
    splits: DatasetSplits,
    created_at: datetime,
    dataset_path: Path | None,
    frozen_hash: str,
    decision_hash: str,
) -> dict[str, Any]:
    policy_kind = result.policy.kind
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": created_at.isoformat(),
        "dataset": {
            "path": str(dataset_path) if dataset_path is not None else None,
            "rows": splits.row_count,
            "features": len(splits.feature_columns),
        },
        "split": splits.describe(),
        "model": {
            "source": str(result.model.path),
            "frozen_copy": FROZEN_MODEL_FILENAME,
            "sha256": result.model.sha256,
            "frozen_copy_sha256": frozen_hash,
            "seed": result.model.seed,
            "bundle_threshold": result.model.threshold,
            "family": result.model.metadata.get("family"),
            "params": result.model.metadata.get("params"),
            "imbalance": result.model.metadata.get("imbalance"),
            "scaling": result.model.metadata.get("scaling"),
            "calibration": result.model.metadata.get("calibration"),
        },
        "selection": {
            "data": "validation",
            "policy": result.policy.to_dict(),
            "threshold": result.decision.threshold,
            "rule": result.decision.rule,
            "diagnostics": result.decision.diagnostics,
            "bundle_threshold": result.model.threshold,
            "matches_bundle_threshold": bool(result.decision.threshold == result.model.threshold),
        },
        "impact_cost_policy": result.impact_policy.to_dict(),
        "test_evaluation": {
            "model_scorings_in_this_run": 1,
            "performed_after_decision_freeze": True,
            "decision_file_sha256": decision_hash,
            "prior_test_evaluation": (
                "the comparison stage scored this split once before this final evaluation; "
                "this is a repeat final report, not an independent holdout"
            ),
        },
        "supported_policies": ["expected_cost", "min_recall", "alert_capacity"],
        "selected_policy": policy_kind,
        "library_versions": {
            "scikit-learn": str(sklearn.__version__),
            "xgboost": str(xgboost.__version__),
            "pandas": str(pd.__version__),
            "numpy": str(np.__version__),
            "joblib": str(joblib.__version__),
        },
        "python_version": platform.python_version(),
        "git_revision": _git_revision(),
        "artifacts": {
            "decision": DECISION_FILENAME,
            "frozen_model": FROZEN_MODEL_FILENAME,
            "metrics": METRICS_FILENAME,
            "run": RUN_FILENAME,
        },
        "notes": {
            "test_set": (
                "scored once after the model and threshold were frozen; no test label was "
                "used in selection. The split is not pristine because the comparison stage "
                "already evaluated it; future tuning needs validation or a fresh holdout."
            ),
            "threshold": (
                "the threshold is a policy choice selected on validation, not a universally "
                "optimal value"
            ),
            "calibration": (
                "the threshold is a validated ranking cut; the reliability/Brier comparison "
                "remains M4 work, so the absolute probability scale is not certified"
            ),
            "cost": (
                "cost weights are provisional business assumptions (ARCHITECTURE 13.2); for "
                "non-cost policies the cost impact is informational, not the selection rule"
            ),
            "accuracy": (
                "accuracy appears only next to the majority-class baseline; it never selects "
                "a model or a threshold"
            ),
            "reproducibility": (
                "the model SHA-256 and split seed/config are recorded; tree parallelism "
                "changes floating-point summation order, so bit-exact replication on another "
                "thread count is not guaranteed"
            ),
            "status": (
                "final binary evaluation implemented; not a production-ready financial "
                "decision system. Three-way review/block policy, MLflow tracking of this "
                "evaluator and the calibration comparison remain later work."
            ),
        },
    }


def _predict_probabilities(
    model: Any,
    features: pd.DataFrame,
    *,
    context: str,
) -> NDArray[np.float64]:
    """Score one split once and return the positive-class probabilities."""
    try:
        probabilities = np.asarray(model.predict_proba(features), dtype="float64")
    except Exception as exc:
        msg = f"model failed to score the {context} split: {exc}"
        raise FinalEvaluationError(msg) from exc
    expected_shape = (len(features), 2)
    if probabilities.ndim != 2 or probabilities.shape != expected_shape:
        msg = (
            f"expected a two-column probability matrix with shape {expected_shape} "
            f"for the {context} split, got {probabilities.shape}"
        )
        raise FinalEvaluationError(msg)
    scores = probabilities[:, 1]
    if not np.isfinite(scores).all():
        msg = f"{context} probabilities contain non-finite values"
        raise FinalEvaluationError(msg)
    if (scores < 0.0).any() or (scores > 1.0).any():
        msg = f"{context} probabilities are outside [0, 1]"
        raise FinalEvaluationError(msg)
    return scores


def _require_both_classes(split: DatasetSplit, name: str) -> None:
    if int(split.y.nunique()) < 2:
        msg = (
            f"the {name} split contains only one class; metrics and threshold selection are "
            "undefined. Use a larger split or a different split strategy."
        )
        raise FinalEvaluationError(msg)


def _write_json(path: Path, record: dict[str, Any]) -> None:
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def _git_revision() -> str | None:
    """Best-effort repository revision; ``None`` when git is unavailable or not a repo."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607 - git is resolved from PATH by design
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = completed.stdout.strip()
    return revision if completed.returncode == 0 and revision else None


# ---------------------------------------------------------------------------
# CLI: python -m fraud_risk.models.final_evaluation
# ---------------------------------------------------------------------------
def _emit(text: str) -> None:
    """Write CLI output to stdout (``print`` is reserved for the dashboard)."""
    sys.stdout.write(text + "\n")


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the final-evaluation stage."""
    parser = argparse.ArgumentParser(
        prog="python -m fraud_risk.models.final_evaluation",
        description=(
            "Load the frozen comparison bundle, select an operating threshold on validation, "
            "freeze the model and threshold, then score the test split once."
        ),
        epilog=(
            "Exit codes: 0 ok, 1 invalid data/split/policy or an unusable validation split, "
            "2 missing dataset or model. The default policy is expected cost using the "
            "weights recorded in the bundle. Test data is never used to choose the threshold. "
            "A selected threshold is a policy choice, not a universally optimal value."
        ),
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"validated dataset file (default: {DEFAULT_DATASET_PATH})",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"comparison bundle to freeze (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"final-evaluation artifact directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--strategy",
        choices=[strategy.value for strategy in SplitStrategy],
        default=None,
        help="override the split strategy; default reads run.json next to the model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="override the split seed; default reads run.json next to the model",
    )
    parser.add_argument(
        "--policy",
        choices=("cost", "min-recall", "alert-capacity"),
        default="cost",
        help="operating policy (default: cost)",
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default=None,
        help="minimum fraud recall for --policy min-recall, e.g. 0.80",
    )
    parser.add_argument(
        "--max-alerts",
        type=int,
        default=None,
        help="absolute validation alert capacity for --policy alert-capacity",
    )
    parser.add_argument(
        "--max-alert-rate",
        type=float,
        default=None,
        help="validation alert capacity as a share of rows for --policy alert-capacity",
    )
    return parser


def _resolve_split_plan(
    model_path: Path,
    *,
    strategy: str | None,
    seed: int | None,
    fallback_feature_columns: Sequence[str],
) -> tuple[SplitConfig, tuple[str, ...]]:
    """Read the training split record, or build one from explicit CLI arguments."""
    run_path = model_path.parent / RUN_FILENAME
    split_record: dict[str, Any] | None = None
    if run_path.is_file():
        try:
            loaded = json.loads(run_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            msg = f"could not read split provenance {run_path}: {exc}"
            raise FinalEvaluationError(msg) from exc
        if not isinstance(loaded, dict) or not isinstance(loaded.get("split"), dict):
            msg = f"{run_path} does not contain a split record"
            raise FinalEvaluationError(msg)
        split_record = cast(dict[str, Any], loaded["split"])

    if split_record is None and (strategy is None or seed is None):
        msg = (
            "no run.json split record next to the model; pass both --strategy and --seed to "
            "recreate the split"
        )
        raise FinalEvaluationError(msg)

    if split_record is None:
        if strategy is None or seed is None:  # pragma: no cover - guarded above
            msg = "strategy and seed are required"
            raise FinalEvaluationError(msg)
        return SplitConfig(strategy=SplitStrategy(strategy), seed=seed), tuple(
            fallback_feature_columns
        )

    split_strategy = strategy if strategy is not None else split_record.get("strategy")
    if not isinstance(split_strategy, str):
        msg = f"{run_path} has no valid split strategy"
        raise FinalEvaluationError(msg)
    split_seed = seed if seed is not None else split_record.get("seed")
    if isinstance(split_seed, bool) or not isinstance(split_seed, int):
        msg = f"{run_path} has no valid split seed"
        raise FinalEvaluationError(msg)

    raw_fractions = split_record.get("fractions")
    if not isinstance(raw_fractions, dict):
        msg = f"{run_path} has no split fractions"
        raise FinalEvaluationError(msg)
    try:
        config = SplitConfig(
            strategy=SplitStrategy(split_strategy),
            seed=split_seed,
            train_fraction=float(raw_fractions["train"]),
            validation_fraction=float(raw_fractions["validation"]),
            test_fraction=float(raw_fractions["test"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        msg = f"{run_path} contains invalid split fractions"
        raise FinalEvaluationError(msg) from exc

    raw_features = split_record.get("feature_columns")
    if raw_features is None:
        features = tuple(fallback_feature_columns)
    elif (
        isinstance(raw_features, list)
        and raw_features
        and all(isinstance(column, str) and column for column in raw_features)
    ):
        features = tuple(str(column) for column in raw_features)
    else:
        msg = f"{run_path} has invalid feature columns"
        raise FinalEvaluationError(msg)
    return config, features


def _policy_from_options(options: argparse.Namespace, model: FrozenModel) -> ThresholdPolicy:
    policy = str(options.policy)
    if policy == "cost":
        if any(
            value is not None
            for value in (options.min_recall, options.max_alerts, options.max_alert_rate)
        ):
            msg = "--policy cost does not accept recall or capacity arguments"
            raise FinalEvaluationError(msg)
        return model.cost_policy

    if policy == "min-recall":
        if options.min_recall is None:
            msg = "--policy min-recall requires --min-recall"
            raise FinalEvaluationError(msg)
        if options.max_alerts is not None or options.max_alert_rate is not None:
            msg = "--policy min-recall does not accept capacity arguments"
            raise FinalEvaluationError(msg)
        try:
            return MinRecallPolicy(float(options.min_recall))
        except MetricError as exc:
            raise FinalEvaluationError(str(exc)) from exc

    if policy == "alert-capacity":
        if options.min_recall is not None:
            msg = "--policy alert-capacity does not accept --min-recall"
            raise FinalEvaluationError(msg)
        if (options.max_alerts is None) == (options.max_alert_rate is None):
            msg = "--policy alert-capacity requires exactly one of --max-alerts/--max-alert-rate"
            raise FinalEvaluationError(msg)
        try:
            return AlertCapacityPolicy(
                max_alerts=options.max_alerts,
                max_alert_rate=options.max_alert_rate,
            )
        except MetricError as exc:
            raise FinalEvaluationError(str(exc)) from exc

    msg = f"unknown policy {policy!r}"  # pragma: no cover - argparse constrains this
    raise FinalEvaluationError(msg)


def _describe_policy(policy: ThresholdPolicy) -> str:
    if isinstance(policy, CostPolicy):
        return (
            f"expected cost (FN {policy.false_negative_cost:g} units, "
            f"FP {policy.false_positive_cost:g} units)"
        )
    if isinstance(policy, MinRecallPolicy):
        return f"minimum recall {policy.min_recall:g}"
    if policy.max_alerts is not None:
        return f"alert capacity {policy.max_alerts} validation alerts"
    return f"alert capacity {policy.max_alert_rate:g} of validation rows"


def _format_metrics(label: str, metrics: ClassificationMetrics) -> str:
    matrix = metrics.to_dict()["confusion_matrix"]
    return (
        f"  {label:<10} precision {metrics.precision:.4f}  recall {metrics.recall:.4f}  "
        f"f1 {metrics.f1:.4f}  pr-auc {metrics.pr_auc:.4f}  roc-auc {metrics.roc_auc:.4f}  "
        f"fpr {metrics.false_positive_rate:.6f}  alerts {metrics.alerts}  "
        f"cm [[{matrix['true_negative']}, {matrix['false_positive']}], "
        f"[{matrix['false_negative']}, {matrix['true_positive']}]]"
    )


def _print_result(result: FinalEvaluation, *, dataset_path: Path) -> None:
    validation_impact = cost_impact(result.validation.model, result.impact_policy)
    test_impact = cost_impact(result.test.model, result.impact_policy)
    _emit(f"Dataset    : {dataset_path}")
    _emit(f"Model      : {result.model.path}  sha256 {result.model.sha256[:12]}...")
    _emit(f"Policy     : {_describe_policy(result.policy)}")
    _emit(
        f"Threshold  : {result.decision.threshold!r} (selected on validation; frozen before test)"
    )
    _emit("Validation (selection):")
    _emit(_format_metrics("model", result.validation.model))
    _emit(_format_metrics("majority", result.validation.majority_class))
    _emit("Test (scored once after freeze):")
    _emit(_format_metrics("model", result.test.model))
    _emit(_format_metrics("majority", result.test.majority_class))
    _emit(
        f"Cost impact: validation {validation_impact['total_cost_units']:g} units, "
        f"test {test_impact['total_cost_units']:g} units (provisional weights)"
    )
    _emit(
        f"Artifacts  : {result.artifact_dir} "
        f"({DECISION_FILENAME}, {FROZEN_MODEL_FILENAME}, {METRICS_FILENAME}, {RUN_FILENAME})"
    )
    _emit(
        "Note       : threshold is a validation-selected policy choice, not a universal "
        "optimum; the test split is not a pristine holdout and the system is not "
        "production-ready."
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the final evaluation and return a process exit code."""
    options = build_parser().parse_args(argv)
    dataset_path = Path(options.path)
    model_path = Path(options.model)

    if not dataset_path.is_file():
        _emit(f"Dataset not found: {dataset_path}")
        _emit("Place it at data/raw/creditcard.csv or pass --path (see data/README.md).")
        return EXIT_MISSING
    if not model_path.is_file():
        _emit(f"Model bundle not found: {model_path}")
        _emit("Run python -m fraud_risk.models.training first, or pass --model.")
        return EXIT_MISSING

    try:
        model = load_bundle(model_path)
        split_config, feature_columns = _resolve_split_plan(
            model_path,
            strategy=options.strategy,
            seed=options.seed,
            fallback_feature_columns=model.feature_columns,
        )
        policy = _policy_from_options(options, model)
        frame = load_csv(dataset_path)
        validate_dataset(frame, path=dataset_path)
        splits = split_dataset(frame, split_config, feature_columns=feature_columns)
        result = evaluate_final(
            splits,
            model,
            policy,
            output_dir=Path(options.output_dir),
            dataset_path=dataset_path,
        )
    except (DataError, FeatureSchemaError, SplitError, FinalEvaluationError, MetricError) as exc:
        _emit(f"Final evaluation: FAILED\n  {type(exc).__name__}: {exc}")
        return EXIT_INVALID

    _print_result(result, dataset_path=dataset_path)
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())
