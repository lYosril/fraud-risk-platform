"""Exploratory data analysis: summaries and diagnostics.

Responsibility
    Turn a validated dataset into the numbers an analyst needs before any modelling:
    dimensions, schema, duplicates, class balance, amount/time behaviour, per-feature
    statistics and feature/target associations.

Rules
    - **Nothing is fitted here.** No scaler, encoder, bin edge or quantile computed in this
      module is stored or reused later. Preprocessing is fitted inside the shared transformer
      on the training split only (AGENTS.md §4), so EDA cannot leak evaluation statistics
      into the pipeline.
    - Importable without a plotting library. This module returns data; the EDA script under
      ``notebooks/`` renders it. Plotting imports are forbidden in ``features`` (AGENTS.md §2).
    - Associations are reported as associations in *this sample*. A correlation is never
      presented as a causal effect.
    - The input must already satisfy the data contract (``validate_dataset``). A violation
      stops the analysis instead of producing confident numbers about broken data.

Time and currency
    ``Time`` is the number of seconds elapsed since the first transaction *in the file*, not
    a wall-clock timestamp: the file carries no dates. ``Amount`` has no documented currency.

Status
    Implemented in M3. See ``docs/ROADMAP.md``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from fraud_risk import __version__
from fraud_risk.data.dataset import (
    AMOUNT_COLUMN,
    DEFAULT_DATASET_PATH,
    FEATURE_COLUMNS,
    SAMPLE_DATASET_PATH,
    TARGET_COLUMN,
    TIME_COLUMN,
    describe_columns,
)
from fraud_risk.data.validation import INPUT_COLUMNS, DatasetReport, validate_dataset

#: Seconds per bucket used for the hourly view of ``Time``.
HOUR_SECONDS = 3600
#: How many features the report ranks by absolute Pearson correlation.
TOP_CORRELATIONS = 12


class DatasetOrigin(StrEnum):
    """What the analysed file is, used to label every report and figure.

    Decided from the path, not from the contents: a copy of the reference file placed
    elsewhere is honestly reported as a custom file rather than claimed to be the reference.
    """

    REFERENCE = "reference"
    SYNTHETIC = "synthetic"
    CUSTOM = "custom"


@dataclass(frozen=True)
class ClassBalance:
    """Class counts, their share of the rows, and the negative:positive ratio."""

    counts: dict[int, int]
    proportions: dict[int, float]
    negatives_per_positive: float | None


@dataclass(frozen=True)
class NonPcaAssociation:
    """The strongest correlation between a PCA component and a non-transformed column."""

    component: str
    column: str
    correlation: float


@dataclass(frozen=True)
class TimeSummary:
    """What the ``Time`` column covers, in the file's own relative seconds."""

    span_hours: float
    is_sorted: bool
    has_fractional_seconds: bool
    busiest_hour: int
    busiest_hour_transactions: int
    busiest_hour_fraud_rate: float
    quietest_hour: int
    quietest_hour_transactions: int


@dataclass(frozen=True)
class EdaReport:
    """Everything the EDA stage computes. Carries no row-level transaction values."""

    path: Path
    origin: DatasetOrigin
    generated_at: datetime
    size_bytes: int
    package_version: str
    validation: DatasetReport
    dtypes: dict[str, str]
    memory_bytes: int
    class_balance: ClassBalance
    feature_statistics: pd.DataFrame = field(repr=False)
    amount_by_class: pd.DataFrame = field(repr=False)
    hourly: pd.DataFrame = field(repr=False)
    correlations: pd.DataFrame = field(repr=False)
    time_summary: TimeSummary = field(repr=False)
    max_inter_feature_correlation: float
    max_component_extra: NonPcaAssociation | None


def classify_origin(path: Path) -> DatasetOrigin:
    """Label ``path`` as the reference file, the synthetic fixture, or a custom file.

    Comparison is by resolved path, so a renamed copy is *not* silently promoted to
    "reference". Paths are resolved against the current directory; run from the repository
    root, as documented.
    """
    resolved = path.resolve()
    if resolved == DEFAULT_DATASET_PATH.resolve():
        return DatasetOrigin.REFERENCE
    if resolved == SAMPLE_DATASET_PATH.resolve():
        return DatasetOrigin.SYNTHETIC
    return DatasetOrigin.CUSTOM


def analyse(frame: pd.DataFrame, *, path: Path) -> EdaReport:
    """Validate ``frame`` and compute the exploratory summaries.

    The frame is never modified. Raises the same typed errors as
    :func:`fraud_risk.data.validation.validate_dataset` when the data contract is violated:
    the analysis refuses to describe data that has already failed validation.
    """
    validation = validate_dataset(frame, path=path)
    origin = classify_origin(path)
    generated_at = datetime.now(UTC)

    class_counts = validation.class_counts
    negatives = class_counts.get(0, 0)
    positives = class_counts.get(1, 0)
    class_balance = ClassBalance(
        counts=class_counts,
        proportions=validation.class_proportions,
        negatives_per_positive=(negatives / positives) if positives else None,
    )

    hourly = _hourly_table(frame)
    time_summary = _time_summary(frame, hourly)

    return EdaReport(
        path=path,
        origin=origin,
        generated_at=generated_at,
        size_bytes=path.stat().st_size if path.exists() else 0,
        package_version=__version__,
        validation=validation,
        dtypes={str(column): str(dtype) for column, dtype in frame.dtypes.items()},
        memory_bytes=int(frame.memory_usage(deep=True).sum()),
        class_balance=class_balance,
        feature_statistics=_feature_statistics(frame),
        amount_by_class=_amount_by_class(frame),
        hourly=hourly,
        correlations=_correlations(frame),
        time_summary=time_summary,
        max_inter_feature_correlation=_max_inter_feature_correlation(frame),
        max_component_extra=_max_component_extra_association(frame),
    )


def render_markdown(report: EdaReport) -> str:
    """Render the report as GitHub-flavoured Markdown, with every number computed."""
    validation = report.validation
    balance = report.class_balance
    total = validation.row_count
    missing = validation.missing_value_count
    duplicates = validation.duplicate_row_count

    lines: list[str] = [
        "# Exploratory data analysis",
        "",
        _origin_banner(report),
        "",
        f"- Generated: {report.generated_at.isoformat(timespec='seconds')}",
        f"- Input: `{report.path}` ({_fmt_size(report.size_bytes)})",
        f"- Rows: {total:,} · Columns: {validation.feature_count + 1} "
        f"({validation.feature_count} features + `{TARGET_COLUMN}`)",
        f"- Package: `fraud_risk {report.package_version}`",
        "",
        "Every count and statistic below is computed from the input file by "
        "`fraud_risk.features.eda`; none of them is hardcoded.",
        "",
        "## 1. Shape and schema",
        "",
        _schema_table(report.dtypes),
        "",
        "Column meanings in this dataset:",
        "",
        f"- `{TIME_COLUMN}` — seconds elapsed since the first transaction in the file. It is "
        "**not** a wall-clock timestamp; the file carries no calendar dates.",
        "- `V1`…`V28` — **anonymised PCA components**. The publisher transformed the original "
        "features with PCA to protect cardholder confidentiality, so these columns are "
        "linear combinations of unknown inputs. Their units, direction and business meaning "
        "are not recoverable. This is why they are never interpreted individually.",
        f"- `{AMOUNT_COLUMN}` — transaction amount, in an undisclosed currency, not "
        "inflation-adjusted.",
        f"- `{TARGET_COLUMN}` — the label (`1` = fraud, `0` = legitimate). It is the target, "
        "never a feature.",
        "",
        "## 2. Missing values and duplicates",
        "",
        f"- Missing values: **{missing:,}** across the "
        f"{validation.feature_count + 1} documented columns.",
        f"- Duplicate rows: **{duplicates:,}** (exact matches across all columns). They are "
        "counted, not removed — see the leakage notes in §10.",
    ]
    if missing:
        lines.append(
            "- The data contract rejects missing values, so this run should not have "
            "produced a report at all; treat the figures above with suspicion."
        )
    lines += [
        "",
        "## 3. Class imbalance",
        "",
        _markdown_table(
            ("class", "meaning", "count", "share"),
            [
                (
                    "0",
                    "legitimate",
                    f"{balance.counts.get(0, 0):,}",
                    f"{balance.proportions.get(0, 0.0):.4%}",
                ),
                (
                    "1",
                    "fraud",
                    f"{balance.counts.get(1, 0):,}",
                    f"{balance.proportions.get(1, 0.0):.4%}",
                ),
            ],
        ),
        "",
        f"One fraudulent transaction for every "
        f"**{_fmt_optional(balance.negatives_per_positive, 1)}** legitimate ones.",
        "",
        "Consequences: accuracy is meaningless here — predicting *legitimate* for every row "
        "would already score the majority share, which is why accuracy is not a reported "
        "metric. PR-AUC is primary and ROC-AUC is reported, never headlined. Per-class "
        f"statistics for the fraud class rest on {balance.counts.get(1, 0):,} events and are "
        "correspondingly noisy.",
        "",
        "## 4. Amount",
        "",
        _amount_table(report.amount_by_class),
        "",
        _amount_observations(report),
        "",
        "## 5. Time",
        "",
        f"- Span: **{report.time_summary.span_hours:,.2f} hours** "
        f"({report.time_summary.span_hours / 24:,.2f} days) from the first transaction.",
        f"- Stored as `{report.dtypes.get(TIME_COLUMN, 'unknown')}`; "
        + (
            "every value is a whole number of seconds."
            if not report.time_summary.has_fractional_seconds
            else "some values carry fractional seconds."
        ),
        f"- Chronologically sorted in the file: "
        f"**{'yes' if report.time_summary.is_sorted else 'no'}**.",
        f"- Busiest hour: hour {report.time_summary.busiest_hour} with "
        f"{_plural(report.time_summary.busiest_hour_transactions, 'transaction')} "
        f"({report.time_summary.busiest_hour_fraud_rate:.4%} fraud).",
        f"- Quietest hour: hour {report.time_summary.quietest_hour} with "
        f"{_plural(report.time_summary.quietest_hour_transactions, 'transaction')}.",
        "",
        "The hourly table (`hourly.csv`) has one row per elapsed hour: volume, fraud count, "
        "fraud rate and mean amount. Because `Time` is relative and covers a short window, "
        "it carries no seasonality, weekday or holiday information.",
        "",
        "## 6. Feature distributions",
        "",
        "`V1`…`V28` are the anonymised PCA components described in §1. Their histograms are "
        "shown in `figures/03_feature_distributions.png`; per-feature mean, standard "
        "deviation, quartiles, skew and kurtosis are in `feature_statistics.csv`.",
        "",
        _feature_observations(report),
        "",
        "## 7. Feature/target associations",
        "",
        "Ranked by absolute Pearson correlation with the label:",
        "",
        _correlation_table(report.correlations),
        "",
        f"The largest absolute Pearson correlation in this file is "
        f"**{_fmt_float(report.correlations['abs_pearson'].max(), 4)}**. {_pca_note(report)}",
        "",
        "**These are associations in one sample, not causal effects.** A correlation of "
        "either sign says nothing about what would happen if a feature were changed, and a "
        "high correlation does not by itself make a feature a good predictor of a "
        f"{balance.proportions.get(1, 0.0):.2%}-positive event. Multivariate models can rank "
        "features very differently from univariate correlations.",
        "",
        "## 8. Method notes",
        "",
        "- No transformation is fitted or saved in this stage. No scaler, encoder, imputer or "
        "discretiser statistics are computed for later reuse, so nothing here can leak "
        "evaluation-period information into preprocessing.",
        "- Histogram bins and log axes are **display choices**, recomputed by the plotting "
        "script for each figure. They are not part of the model pipeline.",
        "- The analysis inspects the whole file, including any period that will later be held "
        "out for evaluation. Any modelling decision inspired by this report must be "
        "re-justified on the training split alone.",
        "",
        "## 9. Limitations",
        "",
        "- `V1`…`V28` are PCA components of undisclosed original features: their business "
        "meaning cannot be audited, and fairness or reason-code analysis on them is limited.",
        "- The published dataset is a short snapshot from September 2013 for European "
        "cardholders (see `docs/DATA_CARD.md`). Neither the fraud rate nor the fraud "
        "typologies should be assumed to transfer to another period, market or product.",
        "- Labels are chargeback-derived and may arrive late and be noisy; the dataset "
        "contains no label timestamps, so label delay cannot be measured here.",
        "- Amount has no documented currency and is not adjusted for inflation.",
        "- Duplicate rows cannot be classified as genuine repeated transactions or ingestion "
        "artefacts from the file alone.",
        "",
        "## 10. Leakage risks",
        "",
        f"- `{TARGET_COLUMN}` is the label. It must never reach a feature matrix; the shared "
        "transformer takes the feature columns explicitly, not the whole frame.",
        _duplicate_leak_note(duplicates),
        f"- `{TIME_COLUMN}` encodes position within the file. It is usable only with a "
        "temporal split; a random split would let the model interpolate across time and "
        "report a score it cannot achieve in production.",
        "- No preprocessing statistic is shared across splits by this stage. Every fitted "
        "transform belongs in the artifact bundle and is fitted on training rows only.",
        "- The file's fraud prevalence is a property of this curated sample, not of a live "
        "portfolio. Thresholds and calibration must be re-derived on production-like data.",
        "",
    ]
    return "\n".join(lines)


def report_to_dict(report: EdaReport) -> dict[str, Any]:
    """Convert the report to JSON-serialisable primitives (no numpy scalars)."""
    validation = report.validation
    balance = report.class_balance
    correlations = report.correlations.head(TOP_CORRELATIONS)
    return {
        "generated_at": report.generated_at.isoformat(timespec="seconds"),
        "package_version": report.package_version,
        "input": {
            "path": str(report.path),
            "origin": str(report.origin),
            "size_bytes": report.size_bytes,
            "rows": validation.row_count,
            "columns": validation.feature_count + 1,
            "features": validation.feature_count,
        },
        "schema": {"columns": describe_columns(), "dtypes": report.dtypes},
        "missing_values": validation.missing_value_count,
        "duplicate_rows": validation.duplicate_row_count,
        "class_balance": {
            "counts": {str(label): int(count) for label, count in balance.counts.items()},
            "proportions": {
                str(label): _json_number(float(share))
                for label, share in balance.proportions.items()
            },
            "negatives_per_positive": (
                _json_number(balance.negatives_per_positive)
                if balance.negatives_per_positive is not None
                else None
            ),
        },
        "time": {
            "span_hours": _json_number(report.time_summary.span_hours),
            "is_sorted": report.time_summary.is_sorted,
            "busiest_hour": report.time_summary.busiest_hour,
            "busiest_hour_transactions": report.time_summary.busiest_hour_transactions,
            "quietest_hour": report.time_summary.quietest_hour,
            "quietest_hour_transactions": report.time_summary.quietest_hour_transactions,
        },
        "amount_by_class": {
            str(index): {
                "count": int(row["count"]),
                "mean": _json_number(float(row["mean"])),
                "median": _json_number(float(row["median"])),
                "max": _json_number(float(row["max"])),
                "zero_share": _json_number(float(row["zero_share"])),
            }
            for index, row in report.amount_by_class.iterrows()
        },
        "correlations": {
            "top": {
                str(feature): {
                    "pearson": _json_number(float(row["pearson"])),
                    "spearman": _json_number(float(row["spearman"])),
                }
                for feature, row in correlations.iterrows()
            },
            "max_abs_pearson": _json_number(float(report.correlations["abs_pearson"].max())),
            "max_abs_inter_feature_correlation": _json_number(report.max_inter_feature_correlation),
            "max_component_extra": (
                {
                    "component": report.max_component_extra.component,
                    "column": report.max_component_extra.column,
                    "pearson": _json_number(report.max_component_extra.correlation),
                }
                if report.max_component_extra is not None
                else None
            ),
        },
    }


def write_report(report: EdaReport, output_dir: Path) -> list[Path]:
    """Write the Markdown/JSON summaries and the CSV tables; return the paths written.

    Figures are the caller's job: they need a plotting library, which this package does not
    import.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    tables = {
        "feature_statistics.csv": report.feature_statistics,
        "correlations.csv": report.correlations,
        "amount_by_class.csv": report.amount_by_class,
        "hourly.csv": report.hourly,
    }
    for filename, table in tables.items():
        target = output_dir / filename
        table.to_csv(target)
        written.append(target)

    markdown = output_dir / "summary.md"
    markdown.write_text(render_markdown(report), encoding="utf-8")
    written.append(markdown)

    summary = output_dir / "summary.json"
    payload = json.dumps(report_to_dict(report), indent=2, sort_keys=True)
    summary.write_text(payload + "\n", encoding="utf-8")
    written.append(summary)

    return written


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------
def _feature_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    columns = list(INPUT_COLUMNS)
    statistics = (
        frame[columns].describe().T.rename(columns={"25%": "q25", "50%": "median", "75%": "q75"})
    )
    statistics["skew"] = frame[columns].skew()
    statistics["kurtosis"] = frame[columns].kurtosis()
    # How far the most extreme value sits from the mean, in standard deviations: a compact
    # way to show heavy tails that a mean/std summary hides.
    statistics["max_abs_z"] = (
        pd.concat(
            [
                (statistics["max"] - statistics["mean"]).abs(),
                (statistics["mean"] - statistics["min"]).abs(),
            ],
            axis=1,
        ).max(axis=1)
        / statistics["std"]
    )
    statistics = statistics[
        [
            "count",
            "mean",
            "std",
            "min",
            "q25",
            "median",
            "q75",
            "max",
            "skew",
            "kurtosis",
            "max_abs_z",
        ]
    ]
    statistics.index.name = "feature"
    return statistics


def _amount_by_class(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby(TARGET_COLUMN)[AMOUNT_COLUMN]
    statistics = grouped.describe().rename(columns={"25%": "q25", "50%": "median", "75%": "q75"})

    zero_rows = (
        frame.assign(_zero=(frame[AMOUNT_COLUMN] == 0))
        .groupby(TARGET_COLUMN)["_zero"]
        .sum()
        .astype("int64")
    )
    statistics["zero_rows"] = zero_rows
    statistics["zero_share"] = zero_rows / statistics["count"]
    statistics["skew"] = grouped.skew()
    statistics = statistics[
        [
            "count",
            "mean",
            "std",
            "min",
            "q25",
            "median",
            "q75",
            "max",
            "zero_rows",
            "zero_share",
            "skew",
        ]
    ]
    statistics.index.name = "class"
    return statistics


def _hourly_table(frame: pd.DataFrame) -> pd.DataFrame:
    hour = (frame[TIME_COLUMN] // HOUR_SECONDS).astype("int64")
    grouped = frame.assign(_hour=hour).groupby("_hour")
    hourly = pd.DataFrame(
        {
            "transactions": grouped.size(),
            "frauds": grouped[TARGET_COLUMN].sum(),
            "mean_amount": grouped[AMOUNT_COLUMN].mean(),
        }
    )
    hourly["fraud_rate"] = hourly["frauds"] / hourly["transactions"]
    hourly.index.name = "hour"
    return hourly


def _time_summary(frame: pd.DataFrame, hourly: pd.DataFrame) -> TimeSummary:
    time_values = frame[TIME_COLUMN]
    span_hours = float((time_values.max() - time_values.min()) / HOUR_SECONDS)
    volumes = hourly["transactions"].to_numpy()
    busiest_position = int(volumes.argmax())
    quietest_position = int(volumes.argmin())
    return TimeSummary(
        span_hours=span_hours,
        is_sorted=bool(time_values.is_monotonic_increasing),
        has_fractional_seconds=bool((time_values % 1 != 0).any()),
        busiest_hour=int(hourly.index[busiest_position]),
        busiest_hour_transactions=int(volumes[busiest_position]),
        busiest_hour_fraud_rate=float(hourly["fraud_rate"].iloc[busiest_position]),
        quietest_hour=int(hourly.index[quietest_position]),
        quietest_hour_transactions=int(volumes[quietest_position]),
    )


def _correlations(frame: pd.DataFrame) -> pd.DataFrame:
    columns = list(INPUT_COLUMNS)
    target = frame[TARGET_COLUMN]
    if target.nunique() < 2:
        # A constant target makes every correlation undefined; computing them would emit
        # third-party warnings and produce NaNs anyway, so state the answer explicitly.
        correlations = pd.DataFrame(
            {"pearson": np.nan, "spearman": np.nan, "abs_pearson": np.nan}, index=columns
        )
        correlations.index.name = "feature"
        return correlations
    pearson = frame[columns].corrwith(target, method="pearson")
    # Spearman is Pearson on average ranks. Computing it this way keeps the analysis free of
    # scipy, which pandas only pulls in for method="spearman" as an undeclared dependency.
    spearman = frame[columns].rank().corrwith(target.rank())
    correlations = pd.DataFrame({"pearson": pearson, "spearman": spearman})
    correlations["abs_pearson"] = correlations["pearson"].abs()
    correlations = correlations.sort_values(by="abs_pearson", ascending=False, na_position="last")
    correlations.index.name = "feature"
    return correlations


def _max_inter_feature_correlation(frame: pd.DataFrame) -> float:
    matrix = frame[list(FEATURE_COLUMNS)].corr().to_numpy()
    upper = matrix[np.triu_indices_from(matrix, k=1)]
    if upper.size == 0:
        return float("nan")
    return float(np.nanmax(np.abs(upper)))


def _max_component_extra_association(frame: pd.DataFrame) -> NonPcaAssociation | None:
    """Find the strongest |r| between a PCA component and ``Amount``/``Time``.

    The publisher transformed only ``V1``..``V28``; ``Time`` and ``Amount`` stayed as they
    were, so a component may still be correlated with them even though the components are
    mutually orthogonal.
    """
    components = list(FEATURE_COLUMNS)
    best: NonPcaAssociation | None = None
    for column in (AMOUNT_COLUMN, TIME_COLUMN):
        series = frame[components].corrwith(frame[column]).abs()
        if series.empty or series.isna().all():
            continue
        component = str(series.idxmax())
        value = float(cast(float, series.loc[component]))
        if best is None or value > best.correlation:
            best = NonPcaAssociation(component=component, column=column, correlation=value)
    return best


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def _origin_banner(report: EdaReport) -> str:
    if report.origin is DatasetOrigin.SYNTHETIC:
        return (
            "> **SYNTHETIC FIXTURE — NOT THE REFERENCE DATASET.** Every figure and number "
            "below was computed from a tiny hand-built fixture. Nothing here describes real "
            "transactions, and none of it may be quoted as a dataset result."
        )
    if report.origin is DatasetOrigin.REFERENCE:
        return (
            "> Input is the reference dataset path (`data/raw/creditcard.csv`). Figures and "
            "statistics were computed from that local copy; the file is not committed."
        )
    return (
        "> Input is a **custom file**, not the reference dataset path. The numbers describe "
        "that file only and should not be quoted as results for the reference dataset."
    )


def _json_number(value: float) -> float | None:
    """JSON has no NaN/Infinity: emit ``null`` instead of writing invalid JSON."""
    return float(value) if math.isfinite(value) else None


def _markdown_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _number(table: pd.DataFrame, row: Any, column: str) -> float:
    """Read a numeric cell as ``float`` (pandas types ``.loc`` as a broad scalar union)."""
    return float(cast(float, table.loc[row, column]))


def _amount_table(table: pd.DataFrame) -> str:
    labels = {0: "legitimate (0)", 1: "fraud (1)"}
    columns = [label for label in (0, 1) if label in table.index]
    rows: list[tuple[str, ...]] = []
    for statistic, name in (
        ("count", "transactions"),
        ("mean", "mean"),
        ("std", "std"),
        ("min", "min"),
        ("q25", "25th percentile"),
        ("median", "median"),
        ("q75", "75th percentile"),
        ("max", "max"),
        ("zero_rows", "zero-amount rows"),
        ("zero_share", "zero-amount share"),
        ("skew", "skew"),
    ):
        values = []
        for label in columns:
            value = _number(table, label, statistic)
            if statistic in {"count", "zero_rows"}:
                values.append(f"{int(value):,}")
            elif statistic.endswith("share"):
                values.append(f"{value:.4%}")
            else:
                values.append(_fmt_float(value, 4))
        rows.append((name, *values))
    return _markdown_table(("statistic", *[labels[label] for label in columns]), rows)


def _amount_observations(report: EdaReport) -> str:
    table = report.amount_by_class
    if 0 not in table.index or 1 not in table.index:
        return "Only one class is present, so a between-class comparison is not possible."
    median_legit = _number(table, 0, "median")
    median_fraud = _number(table, 1, "median")
    if median_fraud > median_legit:
        comparison = (
            f"the median fraudulent amount is **higher** ({median_fraud:,.2f} vs "
            f"{median_legit:,.2f})"
        )
    elif median_fraud < median_legit:
        comparison = (
            f"the median fraudulent amount is **lower** ({median_fraud:,.2f} vs "
            f"{median_legit:,.2f})"
        )
    else:
        comparison = f"both classes share the same median ({median_legit:,.2f})"

    zero_rows = int(_number(table, 1, "zero_rows"))
    zero_share = _number(table, 1, "zero_share")
    if zero_rows:
        zero_line = (
            f"- {zero_share:.4%} of fraudulent rows ({zero_rows:,}) carry a zero amount, so "
            'a naive "amount > 0" rule would silently discard them.'
        )
    else:
        zero_line = "- No fraudulent row in this file carries a zero amount."

    overall_skew = float(cast(float, report.feature_statistics.loc[AMOUNT_COLUMN, "skew"]))
    if math.isnan(overall_skew):
        skew_line = "- Amount skew is undefined here (too few rows to estimate it)."
    else:
        shape = (
            "right-skewed"
            if overall_skew > 0.5
            else "left-skewed"
            if overall_skew < -0.5
            else "close to symmetric"
        )
        skew_line = (
            f"- Amount is {shape} in this file (overall skew {overall_skew:,.2f}); the "
            "figures use a log x-axis for readability only, not as a preprocessing step."
        )
    return "\n".join(
        [
            f"- In this file {comparison}. This is a description of the sample, not "
            "evidence that amount causes fraud.",
            skew_line,
            zero_line,
        ]
    )


def _feature_observations(report: EdaReport) -> str:
    statistics = report.feature_statistics
    most_skewed = statistics["skew"].abs().sort_values(ascending=False).head(5)
    table = _markdown_table(
        ("feature", "skew", "min", "median", "max"),
        [
            (
                str(feature),
                _fmt_float(float(cast(float, statistics.loc[feature, "skew"])), 4),
                _fmt_float(float(cast(float, statistics.loc[feature, "min"])), 4),
                _fmt_float(float(cast(float, statistics.loc[feature, "median"])), 4),
                _fmt_float(float(cast(float, statistics.loc[feature, "max"])), 4),
            )
            for feature in most_skewed.index
        ],
    )
    pca_tails = statistics.loc[list(FEATURE_COLUMNS), "max_abs_z"].dropna()
    if pca_tails.empty:
        tail_note = "The tail behaviour of the components could not be estimated."
    else:
        tails = pca_tails.sort_values(ascending=False)
        extremes = int((tails > 10).sum())
        tail_note = (
            f"The components are heavy-tailed: the most extreme value of "
            f"`{tails.index[0]}` sits {float(tails.iloc[0]):,.1f} standard deviations from "
            f"its mean, and {extremes} of the {len(tails)} components have at least one value "
            "beyond 10 standard deviations. That is why the histograms show a narrow peak "
            "with long tails; a mean/standard-deviation summary alone hides it."
        )
    return (
        "The five features with the largest absolute skew:\n\n"
        f"{table}\n\n"
        f"{tail_note}\n\n"
        "Centring, scaling and any other transform are deliberately **not** fitted here: "
        "doing so on the full file would leak evaluation-period statistics. They will be "
        "fitted on training rows inside the shared transformer."
    )


def _schema_table(dtypes: dict[str, str]) -> str:
    roles = {
        TIME_COLUMN: "elapsed seconds since the first transaction",
        AMOUNT_COLUMN: "transaction amount (currency undisclosed)",
        TARGET_COLUMN: "label: 1 = fraud, 0 = legitimate",
    }
    return _markdown_table(
        ("column", "dtype", "role"),
        [
            (column, dtype, roles.get(column, "PCA component (anonymised)"))
            for column, dtype in dtypes.items()
        ],
    )


def _correlation_table(correlations: pd.DataFrame) -> str:
    top = correlations.head(TOP_CORRELATIONS)
    return _markdown_table(
        ("feature", "Pearson r", "Spearman r", "abs(Pearson r)"),
        [
            (
                str(feature),
                _fmt_float(float(row["pearson"]), 4),
                _fmt_float(float(row["spearman"]), 4),
                _fmt_float(float(row["abs_pearson"]), 4),
            )
            for feature, row in top.iterrows()
        ],
    )


def _fmt_float(value: float, digits: int) -> str:
    if math.isnan(value):
        return "n/a"
    return f"{value:,.{digits}f}"


def _fmt_correlation(value: float) -> str:
    """Correlations near zero need more than four decimals to avoid reading as exactly 0."""
    if math.isnan(value):
        return "n/a"
    if value and abs(value) < 1e-3:
        return f"{value:.2e}"
    return f"{value:,.4f}"


def _fmt_optional(value: float | None, digits: int) -> str:
    if value is None:
        return "n/a (no fraudulent rows)"
    return f"{value:,.{digits}f}"


def _fmt_size(size_bytes: int) -> str:
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):,.1f} MiB"
    return f"{size_bytes / 1024:,.1f} KiB"


def _plural(count: int, singular: str) -> str:
    return f"{count:,} {singular if count == 1 else singular + 's'}"


def _pca_note(report: EdaReport) -> str:
    """Describe the inter-component correlation, flagging output that contradicts PCA."""
    maximum = report.max_inter_feature_correlation
    if math.isnan(maximum):
        return (
            "The correlation between PCA components could not be computed for this file, so "
            "the PCA description above is unverified here."
        )
    if maximum > 0.1:
        return (
            f"The largest absolute correlation between two V columns is "
            f"**{_fmt_correlation(maximum)}**, which is **higher than expected** for PCA "
            "output: genuine components are linearly uncorrelated. Either this file is not "
            "the published PCA version or it has been altered, so the PCA description in §1 "
            "should not be relied on."
        )
    extra = report.max_component_extra
    if extra is None:
        return (
            f"The largest absolute correlation between two V columns is "
            f"**{_fmt_correlation(maximum)}** - as expected, because PCA produces linearly "
            "uncorrelated components. A V-V correlation heatmap is therefore mostly an "
            "identity matrix and is not a useful diagnostic."
        )
    return (
        f"The largest absolute correlation between two V columns is "
        f"**{_fmt_correlation(maximum)}** - as expected, because PCA produces linearly "
        "uncorrelated components, so a V-V heatmap is mostly an identity matrix. A component "
        "can still correlate with the columns that were *not* PCA-transformed: the strongest "
        f"such association is `{extra.component}` with `{extra.column}` at r = "
        f"**{_fmt_float(extra.correlation, 4)}**, which is worth knowing when those columns "
        "are used alongside the components."
    )


def _duplicate_leak_note(duplicates: int) -> str:
    if duplicates:
        return (
            f"- The {duplicates:,} duplicate rows would put identical transactions on both "
            "sides of a random split and inflate evaluation metrics. This is one more reason "
            "for a temporal split (AGENTS.md §4), with duplicate handling decided explicitly "
            "and logged."
        )
    return (
        "- No duplicate rows were found in this file, so a random split would not duplicate a "
        "transaction. Duplicate handling must still be explicit and logged for any future "
        "file."
    )
