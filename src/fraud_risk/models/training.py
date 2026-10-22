"""Model training: a reproducible three-family comparison.

Responsibility
    Fit logistic regression, random forest and XGBoost candidates on the training split,
    rank them on the validation split with a business-oriented criterion, freeze the winner
    and its threshold, score the test split once, and write a self-contained local artifact
    bundle plus machine-readable comparison, metrics and provenance records.

The protocol (identical for every family)
    1. One shared feature pipeline per candidate
       (:func:`fraud_risk.features.engineering.build_preprocessor`): standardisation for the
       linear model, passthrough for the scale-invariant tree models.
    2. A sigmoid calibrator fitted on internal folds of the *training* data, so no candidate
       is compared on a score whose probability scale was fitted on validation or test.
    3. Class imbalance handled by training-only reweighting:
       ``class_weight`` for logistic regression and random forest, ``scale_pos_weight``
       (negatives / positives, computed from the training labels) for XGBoost. No synthetic
       rows, no undersampling, no SMOTE - resampling is applied nowhere.
    4. Every candidate's threshold is chosen on validation to minimise expected cost under
       an explicit :class:`fraud_risk.models.evaluation.CostPolicy` (a missed fraud versus a
       false alarm), and the candidate with the lowest validation cost wins. Ties break on
       validation PR-AUC, then grid order. Accuracy never participates.
    5. The winning candidate and its threshold are frozen, and the test split is scored
       exactly once. No test label is read during fitting, selection or thresholding.

Computational limits
    Bounded grids rather than an open search: four logistic-regression settings, two random
    forests (200 trees, depth 12, leaf 5) and two XGBoost settings (300 rounds, depth 6,
    histogram). Calibration is three-fold. ``n_jobs`` defaults to all cores; the fit time of
    every candidate is recorded so the comparison can be budgeted. The caps are configuration
    fields of :class:`TrainingConfig`, so a fixture-sized run can use smaller ones in tests.

Artifacts
    ``artifacts/comparison/`` (git-ignored) receives ``model.joblib`` (the winning pipeline,
    threshold, feature schema and settings), ``comparison.json`` (every candidate's
    validation score, cost, threshold and metrics), ``metrics.json`` (validation and test at
    the frozen threshold, including per-class metrics and the cost impact of each error type)
    and ``run.json`` (seed, feature schema, split record, grids, imbalance methods, timings,
    library versions, git revision, run fingerprint and the MLflow run identity when
    tracking is enabled). A later run replaces the directory. MLflow records the same run
    parameters, validation/test metrics and the four artifact files; it never registers or
    deploys a model. The calibration comparison remains M4 work.

Status
    Three-family comparison implemented. The binary operating-policy evaluation lives in
    ``models.final_evaluation``; the three-way review/block policy remains M6 work. MLflow
    tracking is integrated for this trainer and the calibration comparison remains M4 work
    (see ``docs/ROADMAP.md``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
import warnings
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from numpy.typing import NDArray
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from fraud_risk.config import get_settings
from fraud_risk.data.dataset import DEFAULT_DATASET_PATH
from fraud_risk.data.errors import DataError
from fraud_risk.data.ingestion import load_csv
from fraud_risk.data.validation import validate_dataset
from fraud_risk.features.engineering import (
    FeatureSchemaError,
    ScalingStrategy,
    build_preprocessor,
)
from fraud_risk.models.evaluation import (
    ClassificationMetrics,
    CostPolicy,
    MetricError,
    MetricsComparison,
    average_precision,
    classification_metrics,
    cost_impact,
    expected_cost,
    majority_class_metrics,
    select_cost_threshold,
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
from fraud_risk.tracking import (
    REGISTRY_NOTE,
    TrackingConfig,
    TrackingError,
    TrackingRecord,
    TrackingSession,
    build_run_fingerprint,
    build_run_name,
    track_experiment,
)

DEFAULT_ARTIFACT_DIR = Path("artifacts") / "comparison"
MODEL_FILENAME = "model.joblib"
METRICS_FILENAME = "metrics.json"
RUN_FILENAME = "run.json"
COMPARISON_FILENAME = "comparison.json"

SELECTION_CRITERION = (
    "minimum expected cost on the validation split at each candidate's cost-optimal "
    "threshold; ties broken by higher validation PR-AUC, then grid order"
)

#: Hard caps that keep a run bounded on a laptop; exposed on TrainingConfig for fixtures.
RF_MAX_DEPTH = 12
RF_MIN_SAMPLES_LEAF = 5
XGB_LEARNING_RATE = 0.1
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE_BYTREE = 0.8


class TrainingError(ValueError):
    """The comparison cannot be trained or evaluated honestly on these splits/config."""


class ModelFamily(StrEnum):
    """The estimator families compared under one protocol."""

    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"


@dataclass(frozen=True)
class TrainingConfig:
    """The reproducibility and computational contract of a comparison run."""

    seed: int
    cost_policy: CostPolicy
    families: tuple[ModelFamily, ...] = (
        ModelFamily.LOGISTIC_REGRESSION,
        ModelFamily.RANDOM_FOREST,
        ModelFamily.XGBOOST,
    )
    c_values: tuple[float, ...] = (0.1, 1.0)
    calibration_folds: int = 3
    max_iter: int = 2000
    rf_n_estimators: int = 200
    rf_max_depth: int = RF_MAX_DEPTH
    rf_min_samples_leaf: int = RF_MIN_SAMPLES_LEAF
    xgb_n_estimators: int = 300
    xgb_max_depth: int = 6
    xgb_learning_rate: float = XGB_LEARNING_RATE
    n_jobs: int = -1

    def __post_init__(self) -> None:
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed < 2**32
        ):
            msg = f"seed must be an int in [0, 2**32) (got {self.seed!r})"
            raise TrainingError(msg)
        if not self.families:
            msg = "families must contain at least one model family"
            raise TrainingError(msg)
        if len(set(self.families)) != len(self.families):
            msg = f"families must be unique (got {self.families!r})"
            raise TrainingError(msg)
        if not self.c_values:
            msg = "c_values must contain at least one candidate"
            raise TrainingError(msg)
        if any(not value > 0 for value in self.c_values):
            msg = f"every C must be strictly positive (got {self.c_values!r})"
            raise TrainingError(msg)
        if self.calibration_folds < 2:
            msg = f"calibration_folds must be at least 2 (got {self.calibration_folds!r})"
            raise TrainingError(msg)
        if self.max_iter < 1:
            msg = f"max_iter must be at least 1 (got {self.max_iter!r})"
            raise TrainingError(msg)
        if self.rf_n_estimators < 1 or self.xgb_n_estimators < 1:
            msg = "tree counts must be at least 1"
            raise TrainingError(msg)
        if self.rf_max_depth < 1 or self.xgb_max_depth < 1:
            msg = "tree depths must be at least 1"
            raise TrainingError(msg)
        if self.rf_min_samples_leaf < 1:
            msg = "rf_min_samples_leaf must be at least 1"
            raise TrainingError(msg)
        if not 0 < self.xgb_learning_rate <= 1:
            msg = f"xgb_learning_rate must be in (0, 1] (got {self.xgb_learning_rate!r})"
            raise TrainingError(msg)
        if self.n_jobs == 0 or self.n_jobs < -1:
            msg = f"n_jobs must be -1 or a positive integer (got {self.n_jobs!r})"
            raise TrainingError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the run configuration for records, fingerprints and tracking params."""
        return {
            "seed": self.seed,
            "cost_policy": self.cost_policy.to_dict(),
            "families": [family.value for family in self.families],
            "c_values": list(self.c_values),
            "calibration_folds": self.calibration_folds,
            "max_iter": self.max_iter,
            "rf_n_estimators": self.rf_n_estimators,
            "rf_max_depth": self.rf_max_depth,
            "rf_min_samples_leaf": self.rf_min_samples_leaf,
            "xgb_n_estimators": self.xgb_n_estimators,
            "xgb_max_depth": self.xgb_max_depth,
            "xgb_learning_rate": self.xgb_learning_rate,
            "n_jobs": self.n_jobs,
        }


@dataclass(frozen=True)
class Candidate:
    """One fully specified grid point, before fitting."""

    name: str
    family: ModelFamily
    params: dict[str, Any]
    imbalance: dict[str, Any]
    scaling: ScalingStrategy

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family.value,
            "params": dict(self.params),
            "imbalance": dict(self.imbalance),
            "scaling": self.scaling.value,
        }


@dataclass(frozen=True)
class CandidateScore:
    """The validation outcome of one fitted candidate."""

    candidate: Candidate
    fit_seconds: float
    converged: bool
    average_precision: float
    threshold: float
    expected_cost: float
    metrics: ClassificationMetrics

    def to_dict(self, cost_policy: CostPolicy) -> dict[str, Any]:
        record = self.candidate.to_dict()
        record.update(
            {
                "fit_seconds": round(self.fit_seconds, 3),
                "converged": self.converged,
                "validation": {
                    "average_precision": round(self.average_precision, 6),
                    "threshold": round(self.threshold, 6),
                    "expected_cost_units": round(self.expected_cost, 2),
                    "metrics": self.metrics.to_dict(),
                    "cost_impact": cost_impact(self.metrics, cost_policy),
                },
            }
        )
        return record


@dataclass(frozen=True)
class ComparisonRun:
    """The frozen winner, every candidate score, both evaluations and the artifact path."""

    model: Any
    selected: CandidateScore
    candidates: tuple[CandidateScore, ...]
    threshold: float
    cost_policy: CostPolicy
    validation: MetricsComparison
    test: MetricsComparison
    artifact_dir: Path
    run_fingerprint: str
    tracking: TrackingRecord | None = None


def train_and_compare(
    splits: DatasetSplits,
    config: TrainingConfig,
    *,
    output_dir: Path = DEFAULT_ARTIFACT_DIR,
    tracking: TrackingConfig | None = None,
) -> ComparisonRun:
    """Fit, compare, freeze, evaluate once and persist the local comparison bundle.

    ``tracking`` is deliberately optional: unit tests and offline runs can call the trainer
    without an MLflow backend, and the CLI enables it by default. When supplied, the run is
    started before fitting and terminated ``FAILED`` if fitting, artifact writing or artifact
    upload raises, so a partial training attempt is never recorded as a success.

    Raises :class:`TrainingError` when a split is unusable (for example validation without a
    single fraud) or when calibration needs more minority rows than the training split has.
    """
    _require_both_classes(splits.train, "train")
    _require_both_classes(splits.validation, "validation")
    _require_both_classes(splits.test, "test")
    _require_minority_rows(splits.train, "train", config.calibration_folds)

    split_record = splits.describe()
    git_revision = _git_revision()
    run_fingerprint = build_run_fingerprint(
        seed=config.seed,
        split=split_record,
        feature_columns=splits.feature_columns,
        config=config.to_dict(),
        git_revision=git_revision,
    )

    with ExitStack() as stack:
        tracker: TrackingSession | None = None
        if tracking is not None:
            tracker = stack.enter_context(
                track_experiment(
                    tracking,
                    run_name=build_run_name(tracking.experiment_name, run_fingerprint),
                    run_fingerprint=run_fingerprint,
                    tags=_tracking_tags(
                        splits=splits,
                        split_record=split_record,
                        config=config,
                        git_revision=git_revision,
                        run_fingerprint=run_fingerprint,
                    ),
                    params=_tracking_params(
                        splits=splits,
                        split_record=split_record,
                        config=config,
                        git_revision=git_revision,
                    ),
                )
            )

        best_model: Any = None
        best: CandidateScore | None = None
        results: list[CandidateScore] = []

        for candidate in _candidates(splits.train, config):
            pipeline, score = _fit_candidate(splits, candidate, config)
            results.append(score)
            if best is None or _is_better(score, best):
                best_model, best = pipeline, score

        if best is None or best_model is None:
            msg = "the candidate grid is empty"
            raise TrainingError(msg)

        # The winner and its threshold are frozen from here on. This is the only test evaluation.
        validation = MetricsComparison(
            model=best.metrics,
            majority_class=majority_class_metrics(splits.validation.y),
        )
        test_scores = _positive_scores(best_model, splits.test.X)
        test = MetricsComparison(
            model=classification_metrics(splits.test.y, test_scores, threshold=best.threshold),
            majority_class=majority_class_metrics(splits.test.y),
        )

        created_at = datetime.now(tz=UTC)
        artifact_dir = Path(output_dir)
        run = ComparisonRun(
            model=best_model,
            selected=best,
            candidates=tuple(results),
            threshold=best.threshold,
            cost_policy=config.cost_policy,
            validation=validation,
            test=test,
            artifact_dir=artifact_dir,
            run_fingerprint=run_fingerprint,
            tracking=tracker.record if tracker is not None else None,
        )
        _write_artifacts(
            run=run,
            splits=splits,
            config=config,
            created_at=created_at,
            git_revision=git_revision,
        )
        if tracker is not None:
            tracker.log_params(_selected_tracking_params(run))
            tracker.log_metrics(_tracking_metrics(run))
            tracker.log_artifacts(
                [
                    artifact_dir / MODEL_FILENAME,
                    artifact_dir / COMPARISON_FILENAME,
                    artifact_dir / METRICS_FILENAME,
                    artifact_dir / RUN_FILENAME,
                ],
                artifact_path="artifacts",
            )
        return run


def _candidates(train: DatasetSplit, config: TrainingConfig) -> list[Candidate]:
    """Enumerate the bounded grid in a deterministic order."""
    candidates: list[Candidate] = []

    if ModelFamily.LOGISTIC_REGRESSION in config.families:
        for class_weight in (None, "balanced"):
            for c_value in config.c_values:
                candidates.append(
                    Candidate(
                        name=f"logistic_regression:class_weight={class_weight},C={c_value}",
                        family=ModelFamily.LOGISTIC_REGRESSION,
                        params={
                            "C": c_value,
                            "class_weight": class_weight,
                            "solver": "lbfgs",
                            "max_iter": config.max_iter,
                        },
                        imbalance={"method": "class_weight", "value": class_weight},
                        scaling=ScalingStrategy.STANDARD,
                    )
                )

    if ModelFamily.RANDOM_FOREST in config.families:
        for class_weight in (None, "balanced_subsample"):
            candidates.append(
                Candidate(
                    name=f"random_forest:class_weight={class_weight}",
                    family=ModelFamily.RANDOM_FOREST,
                    params={
                        "n_estimators": config.rf_n_estimators,
                        "max_depth": config.rf_max_depth,
                        "min_samples_leaf": config.rf_min_samples_leaf,
                        "max_features": "sqrt",
                        "class_weight": class_weight,
                    },
                    imbalance={"method": "class_weight", "value": class_weight},
                    scaling=ScalingStrategy.NONE,
                )
            )

    if ModelFamily.XGBOOST in config.families:
        positives = int(train.y.sum())
        negatives = int(len(train.y) - positives)
        ratio = negatives / positives
        for scale_pos_weight in (1.0, ratio):
            label = "1.0" if scale_pos_weight == 1.0 else f"{scale_pos_weight:.4f}"
            candidates.append(
                Candidate(
                    name=f"xgboost:scale_pos_weight={label}",
                    family=ModelFamily.XGBOOST,
                    params={
                        "n_estimators": config.xgb_n_estimators,
                        "max_depth": config.xgb_max_depth,
                        "learning_rate": config.xgb_learning_rate,
                        "subsample": XGB_SUBSAMPLE,
                        "colsample_bytree": XGB_COLSAMPLE_BYTREE,
                        "tree_method": "hist",
                        "eval_metric": "aucpr",
                        "scale_pos_weight": scale_pos_weight,
                        "verbosity": 0,
                    },
                    imbalance={
                        "method": "scale_pos_weight",
                        "value": round(scale_pos_weight, 6),
                        "derivation": "training negatives / training positives"
                        if scale_pos_weight != 1.0
                        else "no reweighting",
                    },
                    scaling=ScalingStrategy.NONE,
                )
            )

    return candidates


def _is_better(candidate: CandidateScore, incumbent: CandidateScore) -> bool:
    """Business criterion first, ranking quality as the tie-break."""
    if candidate.expected_cost != incumbent.expected_cost:
        return candidate.expected_cost < incumbent.expected_cost
    return candidate.average_precision > incumbent.average_precision


def _fit_candidate(
    splits: DatasetSplits,
    candidate: Candidate,
    config: TrainingConfig,
) -> tuple[Any, CandidateScore]:
    pipeline = _build_pipeline(candidate, config)
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipeline.fit(splits.train.X, splits.train.y)
    fit_seconds = time.perf_counter() - started
    converged = not any(issubclass(warning.category, ConvergenceWarning) for warning in caught)

    scores = _positive_scores(pipeline, splits.validation.X)
    threshold = select_cost_threshold(
        splits.validation.y,
        scores,
        config.cost_policy,
    )
    metrics = classification_metrics(splits.validation.y, scores, threshold=threshold)
    score = CandidateScore(
        candidate=candidate,
        fit_seconds=fit_seconds,
        converged=converged,
        average_precision=average_precision(splits.validation.y, scores),
        threshold=threshold,
        expected_cost=expected_cost(metrics, config.cost_policy),
        metrics=metrics,
    )
    return pipeline, score


def _build_pipeline(candidate: Candidate, config: TrainingConfig) -> Pipeline:
    """One shared preprocessor plus the family's calibrated estimator."""
    if candidate.family is ModelFamily.LOGISTIC_REGRESSION:
        estimator: Any = LogisticRegression(
            **candidate.params,
            random_state=config.seed,
        )
    elif candidate.family is ModelFamily.RANDOM_FOREST:
        estimator = RandomForestClassifier(
            **candidate.params,
            n_jobs=config.n_jobs,
            random_state=config.seed,
        )
    else:
        # ModelFamily is closed and TrainingConfig validates membership, so this is XGBoost.
        estimator = XGBClassifier(
            **candidate.params,
            n_jobs=config.n_jobs,
            random_state=config.seed,
        )

    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(candidate.scaling)),
            (
                "classifier",
                CalibratedClassifierCV(
                    estimator,
                    method="sigmoid",
                    cv=config.calibration_folds,
                ),
            ),
        ]
    )


def _positive_scores(model: Any, features: pd.DataFrame) -> NDArray[np.float64]:
    probabilities = np.asarray(model.predict_proba(features), dtype="float64")
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        msg = f"expected a two-column probability matrix, got shape {probabilities.shape}"
        raise TrainingError(msg)
    return probabilities[:, 1]


def _require_both_classes(split: DatasetSplit, name: str) -> None:
    if int(split.y.nunique()) < 2:
        msg = (
            f"the {name} split contains only one class; metrics and model selection are "
            "undefined. Use a larger split or a different split strategy."
        )
        raise TrainingError(msg)


def _require_minority_rows(split: DatasetSplit, name: str, required: int) -> None:
    """A stratified calibration split needs at least one row per class per fold."""
    minority = int(min(split.y.eq(0).sum(), split.y.eq(1).sum()))
    if minority < required:
        msg = (
            f"the {name} split has {minority} rows of its minority class; {required} are "
            "needed for the calibration folds"
        )
        raise TrainingError(msg)


# ---------------------------------------------------------------------------
# Tracking metadata
# ---------------------------------------------------------------------------
def _tracking_tags(
    *,
    splits: DatasetSplits,
    split_record: dict[str, Any],
    config: TrainingConfig,
    git_revision: str | None,
    run_fingerprint: str,
) -> dict[str, object]:
    """Tags that identify a run and its schema without recording any row values."""
    return {
        "run.fingerprint": run_fingerprint,
        "split.strategy": split_record.get("strategy", splits.config.strategy.value),
        "split.seed": split_record.get("seed", splits.config.seed),
        "feature.count": len(splits.feature_columns),
        "feature.schema.sha256": _schema_digest(split_record.get("feature_schema", {})),
        "model.families": ",".join(family.value for family in config.families),
        "git.revision": git_revision or "unknown",
        "tracking.registry": REGISTRY_NOTE,
    }


def _tracking_params(
    *,
    splits: DatasetSplits,
    split_record: dict[str, Any],
    config: TrainingConfig,
    git_revision: str | None,
) -> dict[str, object]:
    """Configuration and split metadata recorded before a single model is fitted."""
    split_section = split_record.get("splits")
    train_record = split_section.get("train", {}) if isinstance(split_section, dict) else {}
    validation_record = (
        split_section.get("validation", {}) if isinstance(split_section, dict) else {}
    )
    test_record = split_section.get("test", {}) if isinstance(split_section, dict) else {}
    return {
        "seed": config.seed,
        "config.calibration_folds": config.calibration_folds,
        "config.max_iter": config.max_iter,
        "config.rf_n_estimators": config.rf_n_estimators,
        "config.rf_max_depth": config.rf_max_depth,
        "config.rf_min_samples_leaf": config.rf_min_samples_leaf,
        "config.xgb_n_estimators": config.xgb_n_estimators,
        "config.xgb_max_depth": config.xgb_max_depth,
        "config.xgb_learning_rate": config.xgb_learning_rate,
        "config.n_jobs": config.n_jobs,
        "families": [family.value for family in config.families],
        "c_values": list(config.c_values),
        "cost.false_negative": config.cost_policy.false_negative_cost,
        "cost.false_positive": config.cost_policy.false_positive_cost,
        "split.strategy": split_record.get("strategy"),
        "split.seed": split_record.get("seed"),
        "split.fractions": split_record.get("fractions"),
        "split.row_count": split_record.get("row_count"),
        "split.train.rows": train_record.get("rows"),
        "split.validation.rows": validation_record.get("rows"),
        "split.test.rows": test_record.get("rows"),
        "split.train.position_sha256": train_record.get("position_sha256"),
        "split.validation.position_sha256": validation_record.get("position_sha256"),
        "split.test.position_sha256": test_record.get("position_sha256"),
        "feature.count": len(splits.feature_columns),
        "feature.columns": list(splits.feature_columns),
        "feature.schema.sha256": _schema_digest(split_record.get("feature_schema", {})),
        "git.revision": git_revision or "unknown",
    }


def _selected_tracking_params(run: ComparisonRun) -> dict[str, object]:
    """The winner's identity, known only after validation selection finishes."""
    return {
        "selected.model.family": run.selected.candidate.family.value,
        "selected.candidate": run.selected.candidate.name,
        "selection.criterion": SELECTION_CRITERION,
    }


def _tracking_metrics(run: ComparisonRun) -> dict[str, float]:
    """Validation and test metrics of the frozen winner, never accuracy as a headline."""
    validation = run.validation.model
    test = run.test.model
    return {
        "validation.pr_auc": validation.pr_auc,
        "validation.roc_auc": validation.roc_auc,
        "validation.precision": validation.precision,
        "validation.recall": validation.recall,
        "validation.f1": validation.f1,
        "validation.false_positive_rate": validation.false_positive_rate,
        "validation.alerts": float(validation.alerts),
        "validation.false_negatives": float(validation.false_negatives),
        "validation.false_positives": float(validation.false_positives),
        "validation.expected_cost_units": run.selected.expected_cost,
        "validation.threshold": run.threshold,
        "validation.majority_class.pr_auc": run.validation.majority_class.pr_auc,
        "test.pr_auc": test.pr_auc,
        "test.roc_auc": test.roc_auc,
        "test.precision": test.precision,
        "test.recall": test.recall,
        "test.f1": test.f1,
        "test.false_positive_rate": test.false_positive_rate,
        "test.alerts": float(test.alerts),
        "test.false_negatives": float(test.false_negatives),
        "test.false_positives": float(test.false_positives),
        "test.expected_cost_units": expected_cost(test, run.cost_policy),
        "test.majority_class.pr_auc": run.test.majority_class.pr_auc,
        "candidate.count": float(len(run.candidates)),
        "fit.total_seconds": float(sum(score.fit_seconds for score in run.candidates)),
    }


def _schema_digest(feature_schema: object) -> str:
    """Stable digest of a feature schema; the schema itself stays in the artifact."""
    canonical = json.dumps(feature_schema, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _tracking_record(tracking: TrackingRecord | None) -> dict[str, Any]:
    """Serialise the MLflow identity, or state plainly that tracking was disabled."""
    if tracking is None:
        return {"enabled": False, "registry": REGISTRY_NOTE}
    return tracking.to_dict()


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------
def _write_artifacts(
    *,
    run: ComparisonRun,
    splits: DatasetSplits,
    config: TrainingConfig,
    created_at: datetime,
    git_revision: str | None,
) -> None:
    artifact_dir = run.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)

    bundle = {
        "schema_version": 2,
        "created_at": created_at.isoformat(),
        "seed": config.seed,
        "run_fingerprint": run.run_fingerprint,
        "tracking": _tracking_record(run.tracking),
        "feature_columns": list(splits.feature_columns),
        "threshold": run.threshold,
        "cost_policy": config.cost_policy.to_dict(),
        "model": {
            "family": run.selected.candidate.family.value,
            "params": dict(run.selected.candidate.params),
            "imbalance": dict(run.selected.candidate.imbalance),
            "scaling": run.selected.candidate.scaling.value,
            "calibration": {
                "method": "sigmoid",
                "folds": config.calibration_folds,
            },
        },
        "pipeline": run.model,
    }
    joblib.dump(bundle, artifact_dir / MODEL_FILENAME)

    (artifact_dir / COMPARISON_FILENAME).write_text(
        json.dumps(_comparison_record(run, config), indent=2) + "\n",
        encoding="utf-8",
    )
    (artifact_dir / METRICS_FILENAME).write_text(
        json.dumps(_metrics_record(run, config.cost_policy), indent=2) + "\n",
        encoding="utf-8",
    )
    (artifact_dir / RUN_FILENAME).write_text(
        json.dumps(
            _run_record(
                run=run,
                splits=splits,
                config=config,
                created_at=created_at,
                git_revision=git_revision,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _comparison_record(run: ComparisonRun, config: TrainingConfig) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "selection": {
            "data": "validation",
            "criterion": SELECTION_CRITERION,
            "accuracy_used": False,
            "cost_policy": config.cost_policy.to_dict(),
            "chosen": run.selected.to_dict(config.cost_policy),
        },
        "candidates": [candidate.to_dict(config.cost_policy) for candidate in run.candidates],
    }


def _metrics_record(run: ComparisonRun, policy: CostPolicy) -> dict[str, Any]:
    return {
        "cost_policy": policy.to_dict(),
        "threshold": {
            "value": run.threshold,
            "rule": (
                "minimum expected cost on the validation split; frozen before the test "
                "split is scored"
            ),
        },
        "validation": _evaluation_record(run.validation, policy),
        "test": _evaluation_record(run.test, policy),
    }


def _evaluation_record(comparison: MetricsComparison, policy: CostPolicy) -> dict[str, Any]:
    record = comparison.to_dict()
    record["model_cost_impact"] = cost_impact(comparison.model, policy)
    return record


def _run_record(
    *,
    run: ComparisonRun,
    splits: DatasetSplits,
    config: TrainingConfig,
    created_at: datetime,
    git_revision: str | None,
) -> dict[str, Any]:
    positives = int(splits.train.y.sum())
    negatives = int(len(splits.train.y) - positives)
    scale_pos_weight = negatives / positives
    fit_seconds = [candidate.fit_seconds for candidate in run.candidates]
    return {
        "schema_version": 3,
        "created_at": created_at.isoformat(),
        "seed": config.seed,
        "run_fingerprint": run.run_fingerprint,
        "tracking": _tracking_record(run.tracking),
        "dataset": {
            "rows": splits.row_count,
            "features": len(splits.feature_columns),
        },
        "feature_columns": list(splits.feature_columns),
        "feature_schema": {column: str(dtype) for column, dtype in splits.train.X.dtypes.items()},
        "split": splits.describe(),
        "grid": {
            "families": [family.value for family in config.families],
            "logistic_regression": {
                "c_values": list(config.c_values),
                "class_weights": [None, "balanced"],
                "max_iter": config.max_iter,
            },
            "random_forest": {
                "n_estimators": config.rf_n_estimators,
                "max_depth": config.rf_max_depth,
                "min_samples_leaf": config.rf_min_samples_leaf,
                "max_features": "sqrt",
                "class_weights": [None, "balanced_subsample"],
            },
            "xgboost": {
                "n_estimators": config.xgb_n_estimators,
                "max_depth": config.xgb_max_depth,
                "learning_rate": config.xgb_learning_rate,
                "subsample": XGB_SUBSAMPLE,
                "colsample_bytree": XGB_COLSAMPLE_BYTREE,
                "tree_method": "hist",
                "scale_pos_weights": [1.0, scale_pos_weight],
            },
            "calibration_folds": config.calibration_folds,
            "n_jobs": config.n_jobs,
        },
        "selection": {
            "criterion": SELECTION_CRITERION,
            "cost_policy": config.cost_policy.to_dict(),
            "chosen": run.selected.candidate.name,
            "threshold": round(run.threshold, 6),
        },
        "imbalance": {
            "approach": (
                "training-only reweighting: class_weight for the linear model and the "
                "forest, scale_pos_weight for XGBoost"
            ),
            "resampling": "none; no synthetic, duplicated or undersampled training rows",
            "xgboost_scale_pos_weight": round(scale_pos_weight, 6),
            "derivation": f"{negatives} training negatives / {positives} training positives",
        },
        "calibration": {
            "method": "sigmoid",
            "fitted_on": f"internal {config.calibration_folds}-fold splits of the training data",
        },
        "computational": {
            "candidate_count": len(run.candidates),
            "total_fit_seconds": round(sum(fit_seconds), 3),
            "max_fit_seconds": round(max(fit_seconds), 3),
            "n_jobs": config.n_jobs,
            "limits": (
                f"bounded grids: logistic max_iter {config.max_iter}, "
                f"{config.rf_n_estimators}-tree depth-{config.rf_max_depth} random forest, "
                f"{config.xgb_n_estimators}-round depth-{config.xgb_max_depth} XGBoost, "
                f"{config.calibration_folds}-fold calibration"
            ),
        },
        "library_versions": {
            "scikit-learn": str(sklearn.__version__),
            "xgboost": str(xgboost.__version__),
            "pandas": str(pd.__version__),
            "numpy": str(np.__version__),
            "joblib": str(joblib.__version__),
        },
        "python_version": platform.python_version(),
        "git_revision": git_revision,
        "artifacts": {
            "model": MODEL_FILENAME,
            "comparison": COMPARISON_FILENAME,
            "metrics": METRICS_FILENAME,
            "run": RUN_FILENAME,
        },
        "notes": {
            "test_set": (
                "scored once after the candidate and the threshold were frozen; never used "
                "to select either. The earlier baseline also scored this split, and the "
                "final-evaluation stage repeats the test scoring, so it is not a pristine "
                "holdout for future tuning."
            ),
            "reproducibility": (
                "fixed seeds and deterministic grids; tree parallelism changes "
                "floating-point summation order, so bit-exact replication on another "
                "machine or thread count requires --n-jobs 1"
            ),
            "accuracy": (
                "reported only inside accuracy_comparison to expose the majority-class trap; "
                "never a selection metric or a headline"
            ),
            "status": (
                "comparison implemented; the final binary evaluation and operating policies "
                "live in models.final_evaluation. MLflow tracking is integrated for this "
                "trainer; the calibration comparison and the three-way review/block policy "
                "remain later work."
            ),
        },
    }


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
# CLI: python -m fraud_risk.models.training
# ---------------------------------------------------------------------------
def _emit(text: str) -> None:
    """Write CLI output to stdout (``print`` is reserved for the dashboard)."""
    sys.stdout.write(text + "\n")


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the comparison trainer."""
    parser = argparse.ArgumentParser(
        prog="python -m fraud_risk.models.training",
        description=(
            "Fit the logistic-regression, random-forest and XGBoost grids on the training "
            "split, select on validation expected cost, freeze a threshold, evaluate the "
            "test split once and write a local artifact bundle."
        ),
        epilog=(
            "Exit codes: 0 ok, 1 invalid data or an unusable split, 2 missing dataset. "
            "Cost weights come from FALSE_NEGATIVE_COST / FALSE_POSITIVE_COST. Artifacts "
            "default to artifacts/comparison/ (git-ignored). MLflow tracking is enabled by "
            "default and records parameters, metrics and the artifact files; no model is "
            "registered. Use --no-mlflow for an offline run."
        ),
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"validated dataset file (default: {DEFAULT_DATASET_PATH})",
    )
    parser.add_argument(
        "--strategy",
        choices=[strategy.value for strategy in SplitStrategy],
        default=SplitStrategy.TEMPORAL.value,
        help="partition rule (default: temporal)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed (default: Settings.random_seed)",
    )
    parser.add_argument(
        "--families",
        default=",".join(family.value for family in ModelFamily),
        help="comma-separated subset of: " + ", ".join(family.value for family in ModelFamily),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="parallel jobs for the tree fits; -1 uses all cores (default: -1)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help=f"artifact directory (default: {DEFAULT_ARTIFACT_DIR})",
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="MLflow tracking URI (default: Settings.mlflow_tracking_uri)",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="MLflow experiment name (default: Settings.mlflow_experiment_name)",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="skip MLflow tracking for this run; the local artifact bundle is still written",
    )
    return parser


def _parse_families(value: str) -> tuple[ModelFamily, ...]:
    families: list[ModelFamily] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            families.append(ModelFamily(token))
        except ValueError as exc:
            choices = ", ".join(family.value for family in ModelFamily)
            msg = f"unknown model family {token!r}; choose from {choices}"
            raise TrainingError(msg) from exc
    if not families:
        msg = "at least one model family is required"
        raise TrainingError(msg)
    return tuple(families)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the comparison trainer and return a process exit code."""
    options = build_parser().parse_args(argv)
    path: Path = options.path

    if not path.is_file():
        _emit(f"Dataset not found: {path}")
        _emit("Place it at data/raw/creditcard.csv or pass --path (see data/README.md).")
        return EXIT_MISSING

    try:
        settings = get_settings()
        frame = load_csv(path)
        validate_dataset(frame, path=path)
        seed = settings.random_seed if options.seed is None else int(options.seed)
        splits = split_dataset(
            frame,
            SplitConfig(strategy=SplitStrategy(options.strategy), seed=seed),
        )
        tracking = None
        if not options.no_mlflow:
            tracking = TrackingConfig(
                tracking_uri=options.tracking_uri or settings.mlflow_tracking_uri,
                experiment_name=options.experiment_name or settings.mlflow_experiment_name,
            )
        run = train_and_compare(
            splits,
            TrainingConfig(
                seed=seed,
                cost_policy=CostPolicy(
                    false_negative_cost=settings.false_negative_cost,
                    false_positive_cost=settings.false_positive_cost,
                ),
                families=_parse_families(str(options.families)),
                n_jobs=int(options.n_jobs),
            ),
            output_dir=Path(options.output_dir),
            tracking=tracking,
        )
    except (
        DataError,
        FeatureSchemaError,
        SplitError,
        TrackingError,
        TrainingError,
        MetricError,
    ) as exc:
        _emit(f"Training: FAILED\n  {type(exc).__name__}: {exc}")
        return EXIT_INVALID

    _print_run(run, splits)
    return EXIT_OK


def _print_run(run: ComparisonRun, splits: DatasetSplits) -> None:
    _emit(f"Split      : {splits.config.strategy.value}  seed {splits.config.seed}")
    _emit(
        f"Cost policy: FN {run.cost_policy.false_negative_cost:g} units, "
        f"FP {run.cost_policy.false_positive_cost:g} units (provisional, ARCHITECTURE 13.2)"
    )
    _emit(f"Fingerprint: {run.run_fingerprint}")
    if run.tracking is None:
        _emit("MLflow     : disabled (--no-mlflow)")
    else:
        _emit(
            f"MLflow     : run {run.tracking.run_id} "
            f"experiment {run.tracking.experiment_name!r} "
            f"at {run.tracking.tracking_uri}"
        )
    _emit(f"Candidates : {len(run.candidates)}")
    _emit("Validation ranking (selection: lowest expected cost):")
    for candidate in sorted(
        run.candidates,
        key=lambda score: (score.expected_cost, -score.average_precision),
    ):
        metrics = candidate.metrics
        _emit(f"  {candidate.candidate.name}")
        _emit(
            f"      AP {candidate.average_precision:.4f}  "
            f"cost {candidate.expected_cost:.0f}  "
            f"thr {candidate.threshold:.6f}  "
            f"precision {metrics.precision:.4f}  recall {metrics.recall:.4f}  "
            f"f1 {metrics.f1:.4f}  fpr {metrics.false_positive_rate:.6f}  "
            f"fit {candidate.fit_seconds:.1f} s"
        )
    _emit(f"Selected   : {run.selected.candidate.name}")
    _emit(
        f"Threshold  : {run.threshold:.6f} "
        "(minimum expected cost on validation; frozen before test)"
    )
    _emit("Validation (used for selection):")
    _emit(_format_metrics("winner", run.validation.model))
    _emit(_format_metrics("majority", run.validation.majority_class))
    _emit(
        f"  accuracy: model {run.validation.model.accuracy:.6f} vs majority "
        f"{run.validation.majority_class.accuracy:.6f} - shown only to expose the trap"
    )
    _emit("Test (reported once, threshold frozen):")
    _emit(_format_metrics("winner", run.test.model))
    _emit(_format_metrics("majority", run.test.majority_class))
    _emit(
        f"  accuracy: model {run.test.model.accuracy:.6f} vs majority "
        f"{run.test.majority_class.accuracy:.6f} - shown only to expose the trap"
    )
    _emit(
        f"Artifacts  : {run.artifact_dir} "
        f"({MODEL_FILENAME}, {COMPARISON_FILENAME}, {METRICS_FILENAME}, {RUN_FILENAME})"
    )


def _format_metrics(label: str, metrics: ClassificationMetrics) -> str:
    matrix = metrics.to_dict()["confusion_matrix"]
    return (
        f"  {label:<9} precision {metrics.precision:.4f}  recall {metrics.recall:.4f}  "
        f"f1 {metrics.f1:.4f}  pr-auc {metrics.pr_auc:.4f}  roc-auc {metrics.roc_auc:.4f}  "
        f"fpr {metrics.false_positive_rate:.6f}  "
        f"cm [[{matrix['true_negative']}, {matrix['false_positive']}], "
        f"[{matrix['false_negative']}, {matrix['true_positive']}]]"
    )


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())
