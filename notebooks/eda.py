#!/usr/bin/env python
"""Reproducible exploratory data analysis.

Run from the repository root:

    python notebooks/eda.py
    python notebooks/eda.py --path data/sample/creditcard.synthetic.csv \\
        --output-dir reports/eda-synthetic

The default input is ``data/raw/creditcard.csv``; the run aborts with a clear message when
the file is absent, and never substitutes or fabricates data. Every analysis number comes
from :mod:`fraud_risk.features.eda`; this script only renders it.

Outputs (git-ignored):

    reports/eda/summary.md          human-readable report, with the origin banner
    reports/eda/summary.json        the same numbers, machine-readable
    reports/eda/feature_statistics.csv
    reports/eda/correlations.csv
    reports/eda/amount_by_class.csv
    reports/eda/hourly.csv
    reports/eda/figures/*.png       six figures

Why a script rather than a notebook: it is diffable, runs unattended, and the tests execute
it end to end. This file deliberately lives outside ``src/fraud_risk`` — plotting imports are
forbidden inside the package (AGENTS.md §2).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Headless: write PNG files, never open a window.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fraud_risk.data.dataset import (
    AMOUNT_COLUMN,
    DEFAULT_DATASET_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    TIME_COLUMN,
)
from fraud_risk.data.errors import DataError
from fraud_risk.data.ingestion import load_csv
from fraud_risk.features.eda import (
    TOP_CORRELATIONS,
    DatasetOrigin,
    EdaReport,
    analyse,
    write_report,
)

LEGIT_COLOR = "#4C72B0"
FRAUD_COLOR = "#C44E52"
CLASS_STYLES = ((0, LEGIT_COLOR, "legitimate"), (1, FRAUD_COLOR, "fraud"))
DEFAULT_OUTPUT_DIR = Path("reports") / "eda"


def _save(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)


def plot_class_balance(report: EdaReport, path: Path) -> None:
    """Two bars, log scale, with counts and shares annotated."""
    counts = [report.validation.class_counts.get(0, 0), report.validation.class_counts.get(1, 0)]
    shares = [
        report.validation.class_proportions.get(0, 0.0),
        report.validation.class_proportions.get(1, 0.0),
    ]
    figure, axes = plt.subplots(figsize=(7.5, 4.8))
    bars = axes.bar(
        ["legitimate (0)", "fraud (1)"], counts, color=[LEGIT_COLOR, FRAUD_COLOR], width=0.55
    )
    axes.set_title(f"Class balance - {report.validation.row_count:,} transactions")
    axes.set_ylabel("transactions (log scale)")
    if all(count > 0 for count in counts):
        axes.set_yscale("log")
        axes.set_ylim(bottom=0.7)
    for bar, count, share in zip(bars, counts, shares, strict=True):
        axes.annotate(
            f"{count:,}\n{share:.4%}",
            (bar.get_x() + bar.get_width() / 2, count),
            ha="center",
            va="bottom",
            fontsize=10,
        )
    axes.margins(y=0.2)
    _save(figure, path)


def plot_amount(frame: pd.DataFrame, path: Path) -> None:
    """Positive-amount density per class (log x) and quartiles per class."""
    positive = frame[frame[AMOUNT_COLUMN] > 0]
    zero_count = int((frame[AMOUNT_COLUMN] == 0).sum())
    figure, (left, right) = plt.subplots(1, 2, figsize=(13.5, 4.8))

    if not positive.empty:
        low = float(max(positive[AMOUNT_COLUMN].min(), 0.01))
        high = float(max(positive[AMOUNT_COLUMN].max(), low * 1.01))
        bins = np.logspace(np.log10(low), np.log10(high), 60)
        for label, color, name in CLASS_STYLES:
            values = positive.loc[positive[TARGET_COLUMN] == label, AMOUNT_COLUMN]
            if values.empty:
                continue
            left.hist(
                values,
                bins=bins,
                density=True,
                alpha=0.55,
                color=color,
                label=f"{name} (n={len(values):,})",
            )
        left.set_xscale("log")
        left.legend()
    left.set_xlabel("Amount (log scale; positive values only)")
    left.set_ylabel("density within class")
    left.set_title("Amount distribution by class")

    groups = [
        (
            name,
            frame.loc[frame[TARGET_COLUMN] == label, AMOUNT_COLUMN],
        )
        for label, _color, name in CLASS_STYLES
    ]
    non_empty = [(name, values[values > 0]) for name, values in groups if not values.empty]
    if non_empty:
        right.boxplot(
            [values for _, values in non_empty],
            tick_labels=[name for name, _ in non_empty],
            showfliers=False,
        )
        right.set_yscale("log")
    right.set_ylabel("Amount (log scale)")
    right.set_title("Amount quartiles by class (outliers hidden)")
    figure.suptitle(
        f"Amount - {zero_count:,} zero-amount rows are excluded from the log axis",
        fontsize=11,
    )
    figure.tight_layout()
    _save(figure, path)


def plot_feature_distributions(frame: pd.DataFrame, path: Path) -> None:
    """A 4x7 grid: every PCA component gets its own histogram."""
    figure, axes = plt.subplots(4, 7, figsize=(18, 10))
    for axes_item, column in zip(axes.ravel(), FEATURE_COLUMNS, strict=True):
        axes_item.hist(frame[column], bins=50, color=LEGIT_COLOR, alpha=0.85)
        axes_item.set_title(column, fontsize=9)
        axes_item.tick_params(labelsize=7)
    figure.suptitle(
        "V1..V28 - anonymised PCA components, all rows (50 bins each; units are not interpretable)",
        fontsize=12,
    )
    figure.tight_layout()
    _save(figure, path)


def plot_correlations(report: EdaReport, frame: pd.DataFrame, path: Path) -> None:
    """Top features by |r| with the label, plus a selected correlation heatmap."""
    top = report.correlations.head(TOP_CORRELATIONS)
    ordered = top.iloc[::-1]
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1, 1.15]}
    )

    colors = [FRAUD_COLOR if value > 0 else LEGIT_COLOR for value in ordered["pearson"]]
    left.barh(ordered.index, ordered["pearson"], color=colors)
    left.axvline(0, color="black", linewidth=0.8)
    left.set_xlabel("Pearson r with the label (association, not causation)")
    left.set_title(f"Top {len(top)} features by |r|")

    heat_columns = list(top.index[:8])
    for column in (AMOUNT_COLUMN, TIME_COLUMN, TARGET_COLUMN):
        if column not in heat_columns:
            heat_columns.append(column)
    matrix = frame[heat_columns].corr()
    image = right.imshow(matrix.to_numpy(), vmin=-1, vmax=1, cmap="RdBu_r")
    right.set_xticks(range(len(heat_columns)), heat_columns, rotation=45, ha="right")
    right.set_yticks(range(len(heat_columns)), heat_columns)
    for row in range(len(heat_columns)):
        for column in range(len(heat_columns)):
            value = float(matrix.iloc[row, column])
            right.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="black" if abs(value) < 0.6 else "white",
            )
    figure.colorbar(image, ax=right, fraction=0.046, pad=0.04)
    right.set_title("Pearson correlation matrix, selected columns")
    figure.tight_layout()
    _save(figure, path)


def plot_top_feature_distributions(report: EdaReport, frame: pd.DataFrame, path: Path) -> None:
    """Class-conditional histograms for the six most associated features."""
    features = list(report.correlations.head(6).index)
    figure, axes = plt.subplots(2, 3, figsize=(15, 8))
    for axes_item, column in zip(axes.ravel(), features, strict=False):
        low = float(frame[column].min())
        high = float(frame[column].max())
        bins = np.linspace(low, high if high > low else low + 1.0, 61)
        for label, color, name in CLASS_STYLES:
            values = frame.loc[frame[TARGET_COLUMN] == label, column]
            if values.empty:
                continue
            axes_item.hist(
                values,
                bins=bins,
                density=True,
                alpha=0.55,
                color=color,
                label=f"{name} (n={len(values):,})",
            )
        correlation = float(report.correlations.loc[column, "pearson"])
        axes_item.set_title(f"{column}  (Pearson r = {correlation:.3f})")
        axes_item.legend(fontsize=8)
    figure.suptitle(
        "Class-conditional distributions of the six most correlated features (density)",
        fontsize=12,
    )
    figure.tight_layout()
    _save(figure, path)


def plot_time(report: EdaReport, path: Path) -> None:
    """Hourly volume and hourly fraud rate over the file's relative time axis."""
    hourly = report.hourly
    hours = hourly.index.to_numpy()

    figure, (top, bottom) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    top.bar(hours, hourly["transactions"], color=LEGIT_COLOR, width=0.9)
    top.set_ylabel("transactions")
    top.set_title(
        f"Hourly volume - {report.time_summary.span_hours:.2f} hours since the first "
        "transaction in the file"
    )

    bottom.plot(hours, hourly["fraud_rate"], color=FRAUD_COLOR, marker="o", markersize=3)
    bottom.set_xlabel("hour index (elapsed hours; not wall-clock)")
    bottom.set_ylabel("fraud rate")
    bottom.set_title(
        f"Hourly fraud rate - {report.validation.class_counts.get(1, 0):,} fraudulent rows "
        "in total, so each hour is based on very few events"
    )
    bottom.grid(alpha=0.3)
    figure.tight_layout()
    _save(figure, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the exploratory analysis and write figures plus summary statistics."
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"dataset CSV to analyse (default: {DEFAULT_DATASET_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"where to write reports and figures (default: {DEFAULT_OUTPUT_DIR})",
    )
    arguments = parser.parse_args(argv)

    if not arguments.path.is_file():
        print(
            f"EDA aborted: no dataset at {arguments.path}. Place creditcard.csv there "
            "(see data/README.md) or pass --path to another file.",
            file=sys.stderr,
        )
        return 2

    try:
        frame = load_csv(arguments.path)
        report = analyse(frame, path=arguments.path)
    except DataError as exc:
        print(f"EDA aborted: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    figures = arguments.output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    written = write_report(report, arguments.output_dir)
    plot_class_balance(report, figures / "01_class_balance.png")
    plot_amount(frame, figures / "02_amount_distribution.png")
    plot_feature_distributions(frame, figures / "03_feature_distributions.png")
    plot_correlations(report, frame, figures / "04_correlations.png")
    plot_top_feature_distributions(report, frame, figures / "05_top_feature_distributions.png")
    plot_time(report, figures / "06_time_patterns.png")

    origin = (
        "SYNTHETIC FIXTURE"
        if report.origin is DatasetOrigin.SYNTHETIC
        else report.origin.value.upper()
    )
    print(
        f"EDA complete ({origin}): {report.validation.row_count:,} rows, "
        f"{report.validation.class_counts.get(1, 0):,} fraudulent."
    )
    print(f"Written under {arguments.output_dir}:")
    for written_path in sorted(written):
        print(f"  {written_path}")
    for figure_path in sorted(figures.glob("*.png")):
        print(f"  {figure_path}")
    if report.origin is DatasetOrigin.SYNTHETIC:
        print(
            "REMINDER: these outputs describe a synthetic fixture and must not be quoted as "
            "dataset results.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
