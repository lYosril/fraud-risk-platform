#!/usr/bin/env python
"""Render SHAP explanations for the selected frozen model.

Run from the repository root:

    python notebooks/explainability.py
    python notebooks/explainability.py --model artifacts/comparison/model.joblib \\
        --path data/sample/creditcard.synthetic.csv --max-rows 100 --row 0

The default model is the bundle frozen by ``models.final_evaluation`` and the default data
source is ``data/raw/creditcard.csv``. Nothing is substituted when either file is missing.

Outputs (git-ignored):

    reports/explainability/summary.md
    reports/explainability/summary.json
    reports/explainability/global_importance.csv
    reports/explainability/global_importance.json
    reports/explainability/transaction_explanation.json
    reports/explainability/provenance.json
    reports/explainability/figures/global_importance.png
    reports/explainability/figures/transaction_contributions.png

Why this lives outside ``src/fraud_risk``: plotting imports are forbidden in the package
(AGENTS.md section 2). ``models.explainability`` computes the explanations; this script only
selects a reference frame, renders figures and writes files.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")  # Headless: write PNG files, never open a window.

import matplotlib.pyplot as plt

from fraud_risk.data.dataset import DEFAULT_DATASET_PATH
from fraud_risk.data.errors import DataError
from fraud_risk.data.ingestion import load_csv
from fraud_risk.data.validation import validate_dataset
from fraud_risk.features.engineering import FeatureSchemaError
from fraud_risk.models.explainability import (
    DEFAULT_MODEL_PATH,
    DEFAULT_REPORT_DIR,
    ExplanationError,
    GlobalExplanation,
    TransactionExplanation,
    explain_transaction,
    global_feature_importance,
    load_explanation_model,
    write_explanation_report,
)
from fraud_risk.models.splitting import (
    EXIT_INVALID,
    EXIT_MISSING,
    EXIT_OK,
    DatasetSplit,
    SplitConfig,
    SplitError,
    SplitStrategy,
    split_dataset,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIGURES_DIR = DEFAULT_REPORT_DIR / "figures"
INCREASE_COLOR = "#C44E52"
DECREASE_COLOR = "#4C72B0"
NEUTRAL_COLOR = "#7F7F7F"


def _save(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)


def plot_global_importance(
    explanation: GlobalExplanation,
    path: Path,
    *,
    top_features: int = 15,
) -> None:
    """Horizontal bars of mean absolute SHAP, with the output-space caveat."""
    items = list(explanation.importance[:top_features])[::-1]
    labels = [item.feature for item in items]
    values = [item.mean_abs_contribution for item in items]
    figure, axes = plt.subplots(figsize=(9.0, 6.2))
    axes.barh(labels, values, color=INCREASE_COLOR, alpha=0.85)
    axes.set_xlabel("mean |SHAP| contribution to the pre-calibration tree margin")
    axes.set_title(
        f"Global SHAP importance - {explanation.rows_explained:,} sampled rows, "
        f"{explanation.n_folds} calibration folds"
    )
    axes.grid(axis="x", alpha=0.25)
    figure.text(
        0.01,
        0.01,
        "Attribution to this fitted model, not a causal effect. "
        "V1..V28 are anonymous PCA components; tree-margin units are not probability points.",
        fontsize=8,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    _save(figure, path)


def plot_transaction_contributions(
    explanation: TransactionExplanation,
    path: Path,
) -> None:
    """Signed top-k contributions for one transaction."""
    items = list(explanation.contributions)[::-1]
    labels = [item.feature for item in items]
    values = [item.contribution for item in items]
    colors = [
        INCREASE_COLOR if value > 0 else DECREASE_COLOR if value < 0 else NEUTRAL_COLOR
        for value in values
    ]
    figure, axes = plt.subplots(figsize=(9.0, max(4.2, 0.38 * len(items) + 1.6)))
    axes.barh(labels, values, color=colors, alpha=0.85)
    axes.axvline(0, color="black", linewidth=0.8)
    axes.set_xlabel("contribution to the pre-calibration tree margin")
    axes.set_title(
        "Transaction SHAP contributions "
        f"(base {explanation.base_value:.3f}, tree margin {explanation.output_value:.3f})"
    )
    axes.grid(axis="x", alpha=0.25)
    figure.text(
        0.01,
        0.01,
        "Red pushes the tree margin toward fraud; blue pushes it away. "
        "These are not probability points, not the decision threshold, and not causal effects.",
        fontsize=8,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    _save(figure, path)


def _resolve_split_plan(
    model_path: Path,
    *,
    feature_columns: tuple[str, ...],
    strategy: str | None,
    seed: int | None,
) -> tuple[SplitConfig, tuple[str, ...]]:
    """Read the split record next to the model, with CLI overrides for a fixture bundle."""
    run_path = model_path.parent / "run.json"
    split_record: dict[str, Any] | None = None
    if run_path.is_file():
        try:
            loaded = json.loads(run_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            msg = f"could not read split provenance {run_path}: {exc}"
            raise ExplanationError(msg) from exc
        if not isinstance(loaded, dict) or not isinstance(loaded.get("split"), dict):
            msg = f"{run_path} does not contain a split record"
            raise ExplanationError(msg)
        split_record = cast(dict[str, Any], loaded["split"])

    if split_record is None:
        if strategy is None or seed is None:
            msg = (
                "no run.json split record next to the model; pass both --strategy and --seed "
                "to recreate the split"
            )
            raise ExplanationError(msg)
        return SplitConfig(strategy=SplitStrategy(strategy), seed=seed), feature_columns

    split_strategy = strategy if strategy is not None else split_record.get("strategy")
    if not isinstance(split_strategy, str):
        msg = f"{run_path} has no valid split strategy"
        raise ExplanationError(msg)
    split_seed = seed if seed is not None else split_record.get("seed")
    if isinstance(split_seed, bool) or not isinstance(split_seed, int):
        msg = f"{run_path} has no valid split seed"
        raise ExplanationError(msg)

    raw_fractions = split_record.get("fractions")
    if not isinstance(raw_fractions, dict):
        msg = f"{run_path} has no split fractions"
        raise ExplanationError(msg)
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
        raise ExplanationError(msg) from exc

    raw_features = split_record.get("feature_columns")
    if raw_features is None:
        features = tuple(feature_columns)
    elif (
        isinstance(raw_features, list)
        and raw_features
        and all(isinstance(column, str) and column for column in raw_features)
    ):
        features = tuple(str(column) for column in raw_features)
    else:
        msg = f"{run_path} has invalid feature columns"
        raise ExplanationError(msg)

    if features != tuple(feature_columns):
        msg = (
            "the split record's feature columns do not match the model bundle "
            f"({len(features)} vs {len(feature_columns)} columns)"
        )
        raise ExplanationError(msg)
    return config, features


def _select_split(splits: Any, name: str) -> DatasetSplit:
    if name == "train":
        return cast(DatasetSplit, splits.train)
    if name == "validation":
        return cast(DatasetSplit, splits.validation)
    if name == "test":
        return cast(DatasetSplit, splits.test)
    msg = f"unknown split {name!r}"  # pragma: no cover - argparse constrains this
    raise ExplanationError(msg)


def _write_provenance(
    path: Path,
    *,
    dataset_path: Path,
    model_path: Path,
    split_name: str,
    split: DatasetSplit,
    row_index: int,
    max_rows: int,
    top_k: int,
) -> None:
    record = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "dataset": str(dataset_path),
        "model": str(model_path),
        "split": {
            "name": split_name,
            "rows": split.row_count,
            "positive_rate": round(split.positive_rate, 8),
            "time_range_seconds": list(split.time_range) if split.time_range is not None else None,
        },
        "transaction_row_index": row_index,
        "global_max_rows": max_rows,
        "top_k": top_k,
        "notes": [
            "The individual report contains raw and model-input values for one transaction.",
            "Reports are git-ignored and are not a dataset export.",
            "SHAP contributions describe the fitted model, not a causal process.",
        ],
    }
    path.write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute global and individual SHAP explanations for the selected frozen model "
            "and write them under the git-ignored reports directory."
        )
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"validated dataset CSV (default: {DEFAULT_DATASET_PATH})",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"frozen model bundle (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help=f"report directory (default: {DEFAULT_REPORT_DIR})",
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="validation",
        help="reference split for the explanation sample (default: validation)",
    )
    parser.add_argument(
        "--row",
        type=int,
        default=0,
        help="zero-based row in the selected split to explain (default: 0)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=2000,
        help="maximum rows used for global importance (default: 2000)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="individual contributions to report (default: 10)",
    )
    parser.add_argument(
        "--strategy",
        choices=[strategy.value for strategy in SplitStrategy],
        default=None,
        help="override the split strategy when no run.json is next to the model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="override the split seed when no run.json is next to the model",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    options = build_parser().parse_args(argv)
    dataset_path = Path(options.path)
    model_path = Path(options.model)
    output_dir = Path(options.output_dir)

    if not dataset_path.is_file():
        print(
            f"Explainability aborted: no dataset at {dataset_path}. Place creditcard.csv "
            "there (see data/README.md) or pass --path.",
            file=sys.stderr,
        )
        return EXIT_MISSING
    if not model_path.is_file():
        print(
            f"Explainability aborted: no model bundle at {model_path}. Run the comparison "
            "and final-evaluation stages first, or pass --model.",
            file=sys.stderr,
        )
        return EXIT_MISSING

    try:
        model = load_explanation_model(model_path)
        config, feature_columns = _resolve_split_plan(
            model_path,
            feature_columns=model.feature_columns,
            strategy=options.strategy,
            seed=options.seed,
        )
        frame = load_csv(dataset_path)
        validate_dataset(frame, path=dataset_path)
        splits = split_dataset(frame, config, feature_columns=feature_columns)
        split = _select_split(splits, str(options.split))
        if options.row < 0 or options.row >= split.row_count:
            msg = (
                f"--row must be in [0, {split.row_count - 1}] for the "
                f"{options.split} split (got {options.row})"
            )
            raise ExplanationError(msg)

        global_explanation = global_feature_importance(
            model,
            split.X,
            max_rows=int(options.max_rows),
        )
        transaction = explain_transaction(
            model,
            split.X.iloc[[int(options.row)]],
            top_k=int(options.top_k),
        )
        report = write_explanation_report(
            global_explanation,
            transaction,
            output_dir=output_dir,
        )
        figures = output_dir / "figures"
        figures.mkdir(parents=True, exist_ok=True)
        plot_global_importance(global_explanation, figures / "global_importance.png")
        plot_transaction_contributions(
            transaction,
            figures / "transaction_contributions.png",
        )
        _write_provenance(
            output_dir / "provenance.json",
            dataset_path=dataset_path,
            model_path=model_path,
            split_name=str(options.split),
            split=split,
            row_index=int(options.row),
            max_rows=int(options.max_rows),
            top_k=int(options.top_k),
        )
    except (
        DataError,
        FeatureSchemaError,
        SplitError,
        ExplanationError,
        OSError,
    ) as exc:
        print(f"Explainability: FAILED\n  {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_INVALID

    print(
        f"SHAP explanation complete: {global_explanation.rows_explained:,} sampled rows, "
        f"one {options.split} transaction, {model.n_folds} calibration folds."
    )
    print(f"Output explained: {global_explanation.output_space} (not the probability).")
    print("Written under " + str(output_dir) + ":")
    for path in sorted(report.written_paths()):
        print(f"  {path}")
    for figure_path in sorted(figures.glob("*.png")):
        print(f"  {figure_path}")
    print(f"  {output_dir / 'provenance.json'}")
    print(
        "Reminder: contributions are attributions to this fitted model, not causal effects; "
        "the individual report contains values for one row and is git-ignored."
    )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
