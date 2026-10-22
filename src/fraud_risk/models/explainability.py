"""SHAP explanations for the selected calibrated XGBoost model.

Responsibility
    Load the frozen model bundle, explain the fitted tree ensemble with
    :class:`shap.TreeExplainer`, and expose two deliberately separate outputs: global
    feature importance for a sample of transactions and top-k contributions for one
    transaction. Reports are written as data only; the plotting script lives in
    ``notebooks/explainability.py`` because plotting imports are forbidden in ``models``.

What is explained
    The bundle pipeline is ``preprocess -> CalibratedClassifierCV(XGBClassifier)``. The
    calibrated wrapper averages three sigmoid-calibrated tree models fitted on internal
    training folds. SHAP explains each tree model's raw margin (the XGBoost score before
    the sigmoid that produces a probability and before the outer calibration). The reported
    contribution is the mean of those per-fold tree-margin contributions.

    This is an honest but limited decomposition. It is additive in
    ``base_value + sum(contributions) = tree_margin``; it is **not** additive in the final
    calibrated probability, because probability calibration and fold averaging happen after
    the explained output. The calibrated probability and the decision threshold remain
    separate outputs and are never derived from, or mixed into, an explanation.

Feature values
    ``raw_value`` is the value supplied by the caller. ``model_value`` is the value after
    the bundle's fitted preprocessor, which is the exact array the estimator receives.
    For the selected tree model the preprocessor is a passthrough, so the two are equal;
    the distinction is kept because the bundle format allows a scaled preprocessor and a
    future model must not display a re-derived value.

Explainer choice
    :class:`shap.TreeExplainer` with ``feature_perturbation="tree_path_dependent"`` is used
    because the selected estimator is an XGBoost tree ensemble. No background transaction
    sample is required or copied into the artifact. This version deliberately supports the
    XGBoost family only: a random-forest or logistic bundle raises :class:`ExplanationError`
    rather than silently substituting a different explainer or output space.

Limitations
    - ``V1..V28`` in the reference dataset are anonymous PCA components. A contribution on
      ``V14`` is not a human-interpretable business factor.
    - Correlated features share credit. A SHAP value is an attribution to this fitted
      model, not a causal effect on fraud, and not evidence that changing a feature would
      change the outcome.
    - Tree-path-dependent SHAP uses the fitted trees' cover, not an interventional
      background distribution.
    - Global importance is mean absolute contribution on a sample; it is not model quality,
      a promotion gate, or proof of production readiness.

Status
    Implemented for the selected XGBoost bundle. See ``docs/ROADMAP.md`` (M5) and
    ``docs/ARCHITECTURE.md`` section 7.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from fraud_risk.features.engineering import validate_feature_frame
from fraud_risk.models.bundle import (
    DEFAULT_FROZEN_MODEL_PATH,
    ModelBundleError,
    load_model_bundle,
)

#: Default bundle: the model and threshold frozen by ``models.final_evaluation``.
DEFAULT_MODEL_PATH = DEFAULT_FROZEN_MODEL_PATH
#: Generated reports live under the git-ignored ``reports/`` directory.
DEFAULT_REPORT_DIR = Path("reports") / "explainability"
DEFAULT_TOP_K = 10
DEFAULT_MAX_ROWS = 2000

EXPLAINABILITY_SCHEMA_VERSION = 1
EXPLAINER_NAME = "shap.TreeExplainer"
FEATURE_PERTURBATION = "tree_path_dependent"
OUTPUT_SPACE = "pre_calibration_tree_margin"
AGGREGATION = "mean across calibration folds"
SUPPORTED_FAMILY = "xgboost"

#: SHAP additivity is checked against the estimator's own output. Floating-point
#: accumulation across hundreds of trees leaves a small residual; a large residual is a bug.
ADDITIVITY_TOLERANCE = 1e-4

EXPLANATION_LIMITATIONS: tuple[str, ...] = (
    (
        "The explained output is the XGBoost tree margin before the sigmoid probability "
        "and the outer calibration. A contribution is not a probability-point change."
    ),
    (
        "The final calibrated probability and the decision threshold are separate model "
        "and policy outputs. Neither is derived from SHAP values."
    ),
    (
        "The selected bundle is an average of calibrated tree models fitted on internal "
        "folds; the explanation averages their tree-margin contributions and is not an "
        "exact decomposition of predict_proba."
    ),
    (
        "V1..V28 are anonymous PCA components in the reference dataset. Their SHAP values "
        "are not human-interpretable business factors."
    ),
    (
        "Correlated features share credit. A SHAP value is an attribution to this fitted "
        "model, not proof of causation or of what would happen if a feature were changed."
    ),
    (
        "Tree-path-dependent SHAP uses the fitted trees' cover, not an interventional "
        "background sample."
    ),
    (
        "Global importance is a mean absolute contribution on sampled rows. It is not a "
        "measure of model quality, a promotion gate, or evidence of production readiness."
    ),
)


class ExplanationError(ValueError):
    """The bundle cannot produce an honest SHAP explanation."""


class ContributionDirection(StrEnum):
    """Which way a contribution pushes the explained tree output."""

    INCREASES = "increases_tree_margin"
    DECREASES = "decreases_tree_margin"
    NO_EFFECT = "no_effect"


@dataclass(frozen=True)
class ExplanationModel:
    """A validated XGBoost bundle plus the cached explainers for its calibrated folds."""

    path: Path
    sha256: str
    feature_columns: tuple[str, ...]
    pipeline: Any
    preprocessor: Any
    estimators: tuple[Any, ...]
    explainers: tuple[Any, ...]
    base_values: tuple[float, ...]
    calibration_slopes: tuple[float, ...]

    @property
    def family(self) -> str:
        """The estimator family this module can explain."""
        return SUPPORTED_FAMILY

    @property
    def n_folds(self) -> int:
        """Number of calibrated tree models whose margins are averaged."""
        return len(self.estimators)

    def to_dict(self) -> dict[str, Any]:
        """Model provenance without estimator objects or transaction values."""
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "family": self.family,
            "feature_columns": list(self.feature_columns),
            "feature_count": len(self.feature_columns),
            "n_folds": self.n_folds,
            "explainer": EXPLAINER_NAME,
            "feature_perturbation": FEATURE_PERTURBATION,
            "output_space": OUTPUT_SPACE,
            "aggregation": AGGREGATION,
            "base_values": [float(value) for value in self.base_values],
            "calibration_slopes": [float(value) for value in self.calibration_slopes],
        }


@dataclass(frozen=True)
class FeatureContribution:
    """One feature's value and contribution to the explained tree output."""

    feature: str
    rank: int
    raw_value: float
    model_value: float
    contribution: float
    abs_contribution: float
    direction: ContributionDirection

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "rank": self.rank,
            "raw_value": float(self.raw_value),
            "model_value": float(self.model_value),
            "contribution": float(self.contribution),
            "abs_contribution": float(self.abs_contribution),
            "direction": self.direction.value,
        }


@dataclass(frozen=True)
class TransactionExplanation:
    """Top-k SHAP contributions for one transaction, separate from score and threshold."""

    model_path: str
    model_sha256: str
    model_family: str
    output_space: str
    explainer: str
    feature_perturbation: str
    aggregation: str
    n_folds: int
    feature_count: int
    feature_columns: tuple[str, ...]
    top_k: int
    base_value: float
    output_value: float
    other_features_contribution: float
    additivity_residual: float
    contributions: tuple[FeatureContribution, ...]
    limitations: tuple[str, ...] = EXPLANATION_LIMITATIONS

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": EXPLAINABILITY_SCHEMA_VERSION,
            "model": {
                "path": self.model_path,
                "sha256": self.model_sha256,
                "family": self.model_family,
            },
            "explanation": {
                "output_space": self.output_space,
                "explainer": self.explainer,
                "feature_perturbation": self.feature_perturbation,
                "aggregation": self.aggregation,
                "n_folds": self.n_folds,
                "feature_count": self.feature_count,
                "feature_columns": list(self.feature_columns),
                "top_k": self.top_k,
                "base_value": float(self.base_value),
                "output_value": float(self.output_value),
                "other_features_contribution": float(self.other_features_contribution),
                "additivity_residual": float(self.additivity_residual),
                "contributions": [item.to_dict() for item in self.contributions],
            },
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True)
class GlobalFeatureImportance:
    """One feature's average contribution over a sample of transactions."""

    feature: str
    rank: int
    mean_abs_contribution: float
    mean_contribution: float
    share_of_total_abs: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "rank": self.rank,
            "mean_abs_contribution": float(self.mean_abs_contribution),
            "mean_contribution": float(self.mean_contribution),
            "share_of_total_abs": float(self.share_of_total_abs),
        }


@dataclass(frozen=True)
class GlobalExplanation:
    """Global SHAP importance for a sampled frame. Carries no row-level values."""

    model_path: str
    model_sha256: str
    model_family: str
    output_space: str
    explainer: str
    feature_perturbation: str
    aggregation: str
    n_folds: int
    feature_columns: tuple[str, ...]
    rows_available: int
    rows_explained: int
    sampling: str
    base_value: float
    mean_output_value: float
    importance: tuple[GlobalFeatureImportance, ...]
    limitations: tuple[str, ...] = EXPLANATION_LIMITATIONS

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": EXPLAINABILITY_SCHEMA_VERSION,
            "model": {
                "path": self.model_path,
                "sha256": self.model_sha256,
                "family": self.model_family,
            },
            "explanation": {
                "output_space": self.output_space,
                "explainer": self.explainer,
                "feature_perturbation": self.feature_perturbation,
                "aggregation": self.aggregation,
                "n_folds": self.n_folds,
                "feature_columns": list(self.feature_columns),
                "rows_available": self.rows_available,
                "rows_explained": self.rows_explained,
                "sampling": self.sampling,
                "base_value": float(self.base_value),
                "mean_output_value": float(self.mean_output_value),
                "importance": [item.to_dict() for item in self.importance],
            },
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True)
class ExplanationReport:
    """Paths written by :func:`write_explanation_report`."""

    output_dir: Path
    summary_markdown: Path
    summary_json: Path
    global_csv: Path
    global_json: Path
    transaction_json: Path | None

    def written_paths(self) -> tuple[Path, ...]:
        """Every file the report writer produced, in a stable order."""
        paths = [
            self.summary_markdown,
            self.summary_json,
            self.global_csv,
            self.global_json,
        ]
        if self.transaction_json is not None:
            paths.append(self.transaction_json)
        return tuple(paths)


def load_explanation_model(path: Path = DEFAULT_MODEL_PATH) -> ExplanationModel:
    """Load a frozen comparison bundle and build one cached TreeExplainer per fold.

    Raises :class:`ExplanationError` when the bundle is missing or invalid, when it does not
    contain the selected XGBoost family, when the pipeline is not the expected
    ``preprocess -> CalibratedClassifierCV`` shape, or when the fitted preprocessor's output
    names do not match the bundle's feature schema.
    """
    path = Path(path)
    try:
        frozen = load_model_bundle(path)
    except ModelBundleError as exc:
        msg = f"cannot load explanation model: {exc}"
        raise ExplanationError(msg) from exc

    family = frozen.metadata.get("family")
    if family != SUPPORTED_FAMILY:
        msg = (
            "SHAP explanations are implemented for the selected xgboost family only; "
            f"the bundle reports {family!r}. No explainer is substituted silently."
        )
        raise ExplanationError(msg)

    preprocessor, estimators, slopes = _extract_calibrated_estimators(
        frozen.pipeline,
        expected_features=len(frozen.feature_columns),
    )
    _validate_preprocessor_schema(preprocessor, frozen.feature_columns)

    explainers: list[Any] = []
    base_values: list[float] = []
    for estimator in estimators:
        try:
            explainer = _tree_explainer(estimator)
            # TreeExplainer populates its path-dependent expected value on the first call.
            # Reading it before a call returns the model's base score, which is not the SHAP
            # reference value and would make every additivity check fail.
            explainer.shap_values(
                np.zeros((1, len(frozen.feature_columns)), dtype="float64"),
                check_additivity=False,
            )
            expected = np.asarray(explainer.expected_value, dtype="float64")
        except ExplanationError:
            raise
        except Exception as exc:  # shap/estimator-specific failures must not be hidden
            msg = f"could not build a TreeExplainer for the selected XGBoost model: {exc}"
            raise ExplanationError(msg) from exc
        if expected.size != 1:
            msg = (
                "the selected classifier is not a single-output binary tree model; "
                f"TreeExplainer returned {expected.size} base values"
            )
            raise ExplanationError(msg)
        explainers.append(explainer)
        base_values.append(float(expected.reshape(-1)[0]))

    return ExplanationModel(
        path=path,
        sha256=frozen.sha256,
        feature_columns=frozen.feature_columns,
        pipeline=frozen.pipeline,
        preprocessor=preprocessor,
        estimators=tuple(estimators),
        explainers=tuple(explainers),
        base_values=tuple(base_values),
        calibration_slopes=tuple(slopes),
    )


def explain_transaction(
    model: ExplanationModel,
    features: pd.DataFrame,
    *,
    top_k: int = DEFAULT_TOP_K,
) -> TransactionExplanation:
    """Explain exactly one transaction and return its top-k contributions."""
    if len(features) != 1:
        msg = f"explain_transaction needs exactly one row (got {len(features)})"
        raise ExplanationError(msg)
    return explain_transactions(model, features, top_k=top_k)[0]


def explain_transactions(
    model: ExplanationModel,
    features: pd.DataFrame,
    *,
    top_k: int = DEFAULT_TOP_K,
) -> tuple[TransactionExplanation, ...]:
    """Explain every row in ``features`` and return one explanation per row.

    The input frame may contain extra columns; only the bundle's declared feature columns are
    used. Each returned contribution carries both the caller's raw value and the exact value
    produced by the fitted preprocessor. No probability or decision threshold is included:
    those remain separate outputs.
    """
    top_k = _validated_top_k(top_k, len(model.feature_columns))
    raw_values, model_values = _prepare_inputs(model, features)
    contributions, base_value, output_value, residual = _explain_matrix(
        model,
        model_values,
    )

    explanations: list[TransactionExplanation] = []
    for row in range(raw_values.shape[0]):
        order = sorted(
            range(len(model.feature_columns)),
            key=lambda column: (-abs(float(contributions[row, column])), column),
        )
        selected = order[:top_k]
        selected_set = set(selected)
        other = float(
            sum(
                contributions[row, column]
                for column in range(len(model.feature_columns))
                if column not in selected_set
            )
        )
        items: list[FeatureContribution] = []
        for rank, column in enumerate(selected, start=1):
            value = float(contributions[row, column])
            items.append(
                FeatureContribution(
                    feature=model.feature_columns[column],
                    rank=rank,
                    raw_value=float(raw_values[row, column]),
                    model_value=float(model_values[row, column]),
                    contribution=value,
                    abs_contribution=abs(value),
                    direction=_direction(value),
                )
            )
        explanations.append(
            TransactionExplanation(
                model_path=str(model.path),
                model_sha256=model.sha256,
                model_family=model.family,
                output_space=OUTPUT_SPACE,
                explainer=EXPLAINER_NAME,
                feature_perturbation=FEATURE_PERTURBATION,
                aggregation=AGGREGATION,
                n_folds=model.n_folds,
                feature_count=len(model.feature_columns),
                feature_columns=model.feature_columns,
                top_k=len(items),
                base_value=float(base_value[row]),
                output_value=float(output_value[row]),
                other_features_contribution=other,
                additivity_residual=float(residual[row]),
                contributions=tuple(items),
            )
        )
    return tuple(explanations)


def global_feature_importance(
    model: ExplanationModel,
    features: pd.DataFrame,
    *,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> GlobalExplanation:
    """Rank features by mean absolute SHAP contribution on a deterministic sample.

    The sample is uniformly spaced over the supplied frame (first and last rows included)
    rather than randomly drawn, so a report is reproducible without a second seed. No
    row-level feature values are retained in the returned object.
    """
    max_rows = _validated_max_rows(max_rows)
    raw_values, model_values = _prepare_inputs(model, features)
    indices = _sample_indices(raw_values.shape[0], max_rows)
    sampled = model_values[indices]

    mean_abs_total = np.zeros(len(model.feature_columns), dtype="float64")
    mean_total = np.zeros(len(model.feature_columns), dtype="float64")
    base_total = 0.0
    output_total = 0.0
    for explainer, estimator, base_value in zip(
        model.explainers,
        model.estimators,
        model.base_values,
        strict=True,
    ):
        shap_values = _shap_values(explainer, sampled)
        margins = _margins(estimator, sampled)
        _check_additivity(base_value, shap_values, margins)
        mean_abs_total += np.mean(np.abs(shap_values), axis=0)
        mean_total += np.mean(shap_values, axis=0)
        base_total += float(base_value)
        output_total += float(np.mean(margins))

    folds = float(model.n_folds)
    mean_abs = mean_abs_total / folds
    mean_contribution = mean_total / folds
    total_abs = float(mean_abs.sum())
    order = sorted(range(len(mean_abs)), key=lambda column: (-float(mean_abs[column]), column))
    importance = tuple(
        GlobalFeatureImportance(
            feature=model.feature_columns[column],
            rank=rank,
            mean_abs_contribution=float(mean_abs[column]),
            mean_contribution=float(mean_contribution[column]),
            share_of_total_abs=(float(mean_abs[column]) / total_abs if total_abs > 0.0 else 0.0),
        )
        for rank, column in enumerate(order, start=1)
    )

    sampling = (
        "all supplied rows"
        if len(indices) == raw_values.shape[0]
        else "uniformly spaced; first and last rows included"
    )
    return GlobalExplanation(
        model_path=str(model.path),
        model_sha256=model.sha256,
        model_family=model.family,
        output_space=OUTPUT_SPACE,
        explainer=EXPLAINER_NAME,
        feature_perturbation=FEATURE_PERTURBATION,
        aggregation=AGGREGATION,
        n_folds=model.n_folds,
        feature_columns=model.feature_columns,
        rows_available=raw_values.shape[0],
        rows_explained=len(indices),
        sampling=sampling,
        base_value=base_total / folds,
        mean_output_value=output_total / folds,
        importance=importance,
    )


def write_explanation_report(
    global_explanation: GlobalExplanation,
    transaction_explanation: TransactionExplanation | None = None,
    *,
    output_dir: Path = DEFAULT_REPORT_DIR,
) -> ExplanationReport:
    """Write strict JSON, a CSV table and a Markdown report under ``output_dir``.

    The report intentionally contains no global row-level data. An individual explanation
    contains the top-k raw and model-input values for that one transaction because those
    values are required to interpret the contribution; the directory is git-ignored.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(tz=UTC).isoformat()

    global_json = output_dir / "global_importance.json"
    global_csv = output_dir / "global_importance.csv"
    summary_json = output_dir / "summary.json"
    summary_markdown = output_dir / "summary.md"
    transaction_json = (
        output_dir / "transaction_explanation.json" if transaction_explanation is not None else None
    )

    _write_json(global_json, global_explanation.to_dict())
    if transaction_explanation is not None and transaction_json is not None:
        _write_json(transaction_json, transaction_explanation.to_dict())
    _write_global_csv(global_csv, global_explanation)

    summary: dict[str, Any] = {
        "schema_version": EXPLAINABILITY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "global": global_explanation.to_dict(),
        "transaction": (
            transaction_explanation.to_dict() if transaction_explanation is not None else None
        ),
        "artifacts": {
            "global_csv": global_csv.name,
            "global_json": global_json.name,
            "summary_markdown": summary_markdown.name,
            "summary_json": summary_json.name,
            "transaction_json": transaction_json.name if transaction_json else None,
        },
        "notes": [
            "SHAP values are not causal effects.",
            "The explained output is the pre-calibration tree margin, not the probability.",
            "The individual report contains raw and model-input values for one row; reports "
            "are git-ignored and must not be treated as a dataset export.",
        ],
    }
    _write_json(summary_json, summary)
    summary_markdown.write_text(
        render_markdown(global_explanation, transaction_explanation),
        encoding="utf-8",
    )

    return ExplanationReport(
        output_dir=output_dir,
        summary_markdown=summary_markdown,
        summary_json=summary_json,
        global_csv=global_csv,
        global_json=global_json,
        transaction_json=transaction_json,
    )


def render_markdown(
    global_explanation: GlobalExplanation,
    transaction_explanation: TransactionExplanation | None = None,
    *,
    top_features: int = 15,
) -> str:
    """Render the report as Markdown, stating the limits before the numbers."""
    lines = [
        "# SHAP explanation report",
        "",
        f"- Model: `{global_explanation.model_path}`",
        f"- Model SHA-256: `{global_explanation.model_sha256}`",
        f"- Family: `{global_explanation.model_family}`",
        f"- Explainer: `{global_explanation.explainer}` "
        f"(`{global_explanation.feature_perturbation}`)",
        f"- Explained output: **{global_explanation.output_space}** "
        "(before the probability and calibration)",
        f"- Aggregation: {global_explanation.aggregation} ({global_explanation.n_folds} folds)",
        "",
        "## What this report does and does not say",
        "",
        "A positive contribution pushes the tree margin toward fraud; a negative contribution "
        "pushes it away. The numbers are not probability points, not the decision threshold, "
        "and not causal effects. Anonymous PCA components have no human-readable meaning.",
        "",
        "## Global importance",
        "",
        f"Rows explained: {global_explanation.rows_explained} of "
        f"{global_explanation.rows_available} supplied ({global_explanation.sampling}).",
        "",
        "| Rank | Feature | Mean absolute SHAP | Mean SHAP | Share of absolute total |",
        "|-----:|---------|-------------------:|----------:|------------------------:|",
    ]
    for global_item in global_explanation.importance[:top_features]:
        lines.append(
            f"| {global_item.rank} | `{global_item.feature}` | "
            f"{global_item.mean_abs_contribution:.6g} | "
            f"{global_item.mean_contribution:.6g} | {global_item.share_of_total_abs:.4%} |"
        )
    if len(global_explanation.importance) > top_features:
        lines.append(
            f"| ... | {len(global_explanation.importance) - top_features} more features | "
            "see `global_importance.csv` | | |"
        )

    if transaction_explanation is not None:
        transaction = transaction_explanation
        lines.extend(
            [
                "",
                "## Individual transaction",
                "",
                f"Base value: {transaction.base_value:.6g}. "
                f"Explained tree margin: {transaction.output_value:.6g}. "
                f"Other features: {transaction.other_features_contribution:.6g}. "
                f"Additivity residual: {transaction.additivity_residual:.3g}.",
                "",
                "`raw_value` is the caller's input; `model_value` is the exact value after "
                "the bundle's fitted preprocessor.",
                "",
                "| Rank | Feature | Raw value | Model value | Contribution | Direction |",
                "|-----:|---------|----------:|------------:|-------------:|-----------|",
            ]
        )
        for transaction_item in transaction.contributions:
            lines.append(
                f"| {transaction_item.rank} | `{transaction_item.feature}` | "
                f"{transaction_item.raw_value:.6g} | "
                f"{transaction_item.model_value:.6g} | "
                f"{transaction_item.contribution:.6g} | "
                f"{transaction_item.direction.value} |"
            )
        lines.extend(
            [
                "",
                "The individual section contains values for one transaction. The report is "
                "git-ignored and is not a data export.",
            ]
        )

    lines.extend(["", "## Limitations", ""])
    for limitation in global_explanation.limitations:
        lines.append(f"- {limitation}")
    lines.append("")
    return "\n".join(lines)


def _extract_calibrated_estimators(
    pipeline: Any,
    *,
    expected_features: int,
) -> tuple[Any, tuple[Any, ...], tuple[float, ...]]:
    """Validate the bundle pipeline and return its preprocessor, trees and calibrator slopes."""
    if not isinstance(pipeline, Pipeline):
        msg = "the bundle pipeline must be a sklearn Pipeline"
        raise ExplanationError(msg)
    steps = dict(pipeline.named_steps)
    if "preprocess" not in steps or "classifier" not in steps:
        msg = "the bundle pipeline must contain 'preprocess' and 'classifier' steps"
        raise ExplanationError(msg)
    preprocessor = steps["preprocess"]
    classifier = steps["classifier"]
    if not isinstance(classifier, CalibratedClassifierCV):
        msg = (
            "the selected model must be a calibrated classifier; found "
            f"{type(classifier).__name__}. The uncalibrated tree is not explained silently."
        )
        raise ExplanationError(msg)

    calibrated = tuple(getattr(classifier, "calibrated_classifiers_", ()))
    if not calibrated:
        msg = "the calibrated classifier has no fitted calibration folds"
        raise ExplanationError(msg)

    estimators: list[Any] = []
    slopes: list[float] = []
    for fold, item in enumerate(calibrated):
        estimator = getattr(item, "estimator", None)
        if not isinstance(estimator, XGBClassifier):
            msg = (
                f"calibration fold {fold} contains {type(estimator).__name__}; this module "
                f"supports the selected {SUPPORTED_FAMILY} family only"
            )
            raise ExplanationError(msg)
        if getattr(estimator, "n_features_in_", None) != expected_features:
            msg = (
                f"calibration fold {fold} expects {getattr(estimator, 'n_features_in_', None)} "
                f"features, but the bundle declares {expected_features}"
            )
            raise ExplanationError(msg)
        if list(np.asarray(getattr(estimator, "classes_", [])).tolist()) != [0, 1]:
            msg = "the selected XGBoost estimator must have binary classes [0, 1]"
            raise ExplanationError(msg)

        calibrators = getattr(item, "calibrators", None)
        if not isinstance(calibrators, (list, tuple)) or len(calibrators) != 1:
            msg = "the selected calibrated binary model must contain exactly one calibrator"
            raise ExplanationError(msg)
        slope = getattr(calibrators[0], "a_", None)
        if isinstance(slope, bool) or not isinstance(slope, (int, float, np.floating)):
            msg = "the fitted calibrator has no numeric slope"
            raise ExplanationError(msg)
        if not np.isfinite(float(slope)):
            msg = "the fitted calibrator slope is not finite"
            raise ExplanationError(msg)

        estimators.append(estimator)
        slopes.append(float(slope))

    return preprocessor, tuple(estimators), tuple(slopes)


def _validate_preprocessor_schema(
    preprocessor: Any,
    feature_columns: Sequence[str],
) -> None:
    """The SHAP feature axis must be the bundle's feature schema in the same order."""
    try:
        names = tuple(str(name) for name in preprocessor.get_feature_names_out())
    except Exception as exc:
        msg = f"the fitted preprocessor does not expose feature names: {exc}"
        raise ExplanationError(msg) from exc
    if names != tuple(feature_columns):
        msg = (
            "the fitted preprocessor output names do not match the bundle feature schema "
            f"({len(names)} vs {len(feature_columns)} columns)"
        )
        raise ExplanationError(msg)


def _tree_explainer(estimator: XGBClassifier) -> Any:
    """Build the appropriate explainer for the selected tree estimator."""
    import shap

    return shap.TreeExplainer(
        estimator,
        model_output="raw",
        feature_perturbation=FEATURE_PERTURBATION,
    )


def _prepare_inputs(
    model: ExplanationModel,
    features: pd.DataFrame,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate a frame and return its raw values plus exact preprocessor output."""
    if not isinstance(features, pd.DataFrame) or features.empty:
        msg = "features must be a non-empty pandas DataFrame"
        raise ExplanationError(msg)
    try:
        validate_feature_frame(features, feature_columns=model.feature_columns)
    except Exception as exc:
        msg = f"features do not satisfy the bundle feature schema: {exc}"
        raise ExplanationError(msg) from exc

    raw = features.loc[:, list(model.feature_columns)].copy()
    raw_values = np.asarray(raw.to_numpy(dtype="float64", copy=True), dtype="float64")
    if not np.isfinite(raw_values).all():
        msg = "model feature columns contain non-finite values"
        raise ExplanationError(msg)

    try:
        transformed = model.preprocessor.transform(raw)
    except Exception as exc:
        msg = f"the bundle preprocessor failed on the supplied features: {exc}"
        raise ExplanationError(msg) from exc
    model_values = np.asarray(transformed, dtype="float64")
    expected = raw_values.shape
    if model_values.ndim != 2 or model_values.shape != expected:
        msg = f"the fitted preprocessor returned shape {model_values.shape}; expected {expected}"
        raise ExplanationError(msg)
    if not np.isfinite(model_values).all():
        msg = "the fitted preprocessor produced non-finite model inputs"
        raise ExplanationError(msg)
    return raw_values, model_values


def _explain_matrix(
    model: ExplanationModel,
    model_values: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Average per-fold SHAP values and margins; return contributions and additivity."""
    contributions = np.zeros_like(model_values, dtype="float64")
    base = np.zeros(model_values.shape[0], dtype="float64")
    output = np.zeros(model_values.shape[0], dtype="float64")

    for explainer, estimator, base_value in zip(
        model.explainers,
        model.estimators,
        model.base_values,
        strict=True,
    ):
        shap_values = _shap_values(explainer, model_values)
        margins = _margins(estimator, model_values)
        _check_additivity(base_value, shap_values, margins)
        contributions += shap_values
        base += base_value
        output += margins

    folds = float(model.n_folds)
    contributions /= folds
    base /= folds
    output /= folds
    residual = output - (base + contributions.sum(axis=1))
    return contributions, base, output, residual


def _shap_values(explainer: Any, model_values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Call the explainer and normalise its output to ``(rows, features)``."""
    try:
        raw = explainer.shap_values(model_values, check_additivity=False)
    except Exception as exc:
        msg = f"SHAP explanation failed on the supplied features: {exc}"
        raise ExplanationError(msg) from exc
    values = np.asarray(raw, dtype="float64")
    if values.ndim == 3 and values.shape[-1] == 2:
        values = values[:, :, 1]
    if values.shape != model_values.shape:
        msg = (
            f"SHAP returned shape {values.shape}; expected {model_values.shape}. "
            "The explainer output is not in the bundle's feature space."
        )
        raise ExplanationError(msg)
    if not np.isfinite(values).all():
        msg = "SHAP returned non-finite contributions"
        raise ExplanationError(msg)
    return values


def _margins(
    estimator: XGBClassifier,
    model_values: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return the XGBoost raw margin, the output TreeExplainer explains."""
    try:
        raw = estimator.predict(model_values, output_margin=True)
    except Exception as exc:
        msg = f"the selected XGBoost estimator could not produce raw margins: {exc}"
        raise ExplanationError(msg) from exc
    margins = np.asarray(raw, dtype="float64").reshape(-1)
    expected = (model_values.shape[0],)
    if margins.shape != expected:
        msg = f"raw margins have shape {margins.shape}; expected {expected}"
        raise ExplanationError(msg)
    if not np.isfinite(margins).all():
        msg = "the selected XGBoost estimator produced non-finite raw margins"
        raise ExplanationError(msg)
    return margins


def _check_additivity(
    base_value: float,
    shap_values: NDArray[np.float64],
    margins: NDArray[np.float64],
) -> None:
    """Fail loudly if SHAP values do not reconstruct the explained tree output."""
    reconstructed = float(base_value) + shap_values.sum(axis=1)
    residual = np.abs(reconstructed - margins)
    if residual.size == 0 or not np.isfinite(residual).all():
        msg = "could not verify SHAP additivity on an empty or non-finite output"
        raise ExplanationError(msg)
    if float(residual.max()) > ADDITIVITY_TOLERANCE:
        msg = (
            "SHAP values do not reconstruct the tree margin within "
            f"{ADDITIVITY_TOLERANCE:g}; maximum residual {float(residual.max()):.6g}"
        )
        raise ExplanationError(msg)


def _direction(contribution: float) -> ContributionDirection:
    if contribution > 0.0:
        return ContributionDirection.INCREASES
    if contribution < 0.0:
        return ContributionDirection.DECREASES
    return ContributionDirection.NO_EFFECT


def _sample_indices(rows: int, max_rows: int) -> NDArray[np.int64]:
    if rows <= max_rows:
        return np.arange(rows, dtype="int64")
    indices = np.linspace(0, rows - 1, max_rows, dtype="int64")
    return np.unique(indices)


def _validated_top_k(top_k: int, feature_count: int) -> int:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        msg = f"top_k must be a positive int (got {top_k!r})"
        raise ExplanationError(msg)
    return min(top_k, feature_count)


def _validated_max_rows(max_rows: int) -> int:
    if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < 1:
        msg = f"max_rows must be a positive int (got {max_rows!r})"
        raise ExplanationError(msg)
    return max_rows


def _write_global_csv(path: Path, explanation: GlobalExplanation) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
                "feature",
                "mean_abs_contribution",
                "mean_contribution",
                "share_of_total_abs",
            ]
        )
        for item in explanation.importance:
            writer.writerow(
                [
                    item.rank,
                    item.feature,
                    item.mean_abs_contribution,
                    item.mean_contribution,
                    item.share_of_total_abs,
                ]
            )


def _write_json(path: Path, record: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "ADDITIVITY_TOLERANCE",
    "AGGREGATION",
    "DEFAULT_MAX_ROWS",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_REPORT_DIR",
    "DEFAULT_TOP_K",
    "EXPLAINABILITY_SCHEMA_VERSION",
    "EXPLAINER_NAME",
    "EXPLANATION_LIMITATIONS",
    "FEATURE_PERTURBATION",
    "OUTPUT_SPACE",
    "SUPPORTED_FAMILY",
    "ContributionDirection",
    "ExplanationError",
    "ExplanationModel",
    "ExplanationReport",
    "FeatureContribution",
    "GlobalExplanation",
    "GlobalFeatureImportance",
    "TransactionExplanation",
    "explain_transaction",
    "explain_transactions",
    "global_feature_importance",
    "load_explanation_model",
    "render_markdown",
    "write_explanation_report",
]
