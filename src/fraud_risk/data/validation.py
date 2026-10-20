"""Dataset contract validation.

Responsibility
    Decide whether a loaded ``pandas.DataFrame`` satisfies the documented schema, and report
    what it contains. Validation **observes**; it never repairs. There is no imputation, no
    row dropping, no dtype coercion and no de-duplication here — those are preprocessing
    decisions that belong to ``features`` and must be explicit, logged and fitted on training
    data only.

What is checked
    1. Duplicate column names.
    2. At least one data row.
    3. Presence of the 31 documented columns (``EXPECTED_COLUMNS``).
    4. Null/NaN values in any documented column.
    5. Numeric, finite values in every non-target column.
    6. The target column contains only the labels ``0`` and ``1``.

What is reported
    Row count, feature count, missing-value count, duplicate-row count, class counts and class
    proportions.

Notes
    - The label column is ``Class``; it maps to the canonical ``is_fraud`` field. Removing it
      from an inference payload is a separate guard that belongs to the serving path (M6).
    - ``Class`` is *not* accepted on faith: a file whose target contains ``2``, ``-1``, ``0.5``
      or a string is rejected. Whether both classes are present is reported, not enforced —
      that is a training-data decision, not a schema violation.

Status
    Implemented. The dataset fingerprint and the inference-side leakage guard remain in M1/M6
    — see ``docs/ROADMAP.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype

from fraud_risk.data.dataset import EXPECTED_COLUMNS, TARGET_COLUMN, describe_columns
from fraud_risk.data.errors import (
    DuplicateColumnsError,
    EmptyDatasetError,
    InvalidNumericError,
    InvalidTargetError,
    MissingColumnsError,
    MissingValuesError,
)

#: Every documented column except the target (``Time``, ``V1..V28``, ``Amount``).
INPUT_COLUMNS: tuple[str, ...] = tuple(
    column for column in EXPECTED_COLUMNS if column != TARGET_COLUMN
)
#: The only labels the published dataset defines.
BINARY_LABELS: tuple[int, int] = (0, 1)


@dataclass(frozen=True)
class DatasetReport:
    """Summary of a dataset that passed validation. Carries no row-level values."""

    path: Path | None
    row_count: int
    feature_count: int
    missing_value_count: int
    duplicate_row_count: int
    class_counts: dict[int, int]
    class_proportions: dict[int, float]


def validate_dataset(frame: pd.DataFrame, *, path: Path | None = None) -> DatasetReport:
    """Validate ``frame`` against the documented contract and summarise it.

    The frame is never modified; the caller keeps ownership of its data.

    Raises
    ------
    DuplicateColumnsError
        Two documented columns share a name.
    EmptyDatasetError
        The frame has no data rows.
    MissingColumnsError
        A documented column is absent.
    MissingValuesError
        Any documented column contains a null/NaN value.
    InvalidNumericError
        A non-target column is non-numeric, boolean, complex, or holds ``inf``/``-inf``.
    InvalidTargetError
        ``Class`` contains a value other than ``0`` or ``1``.
    """
    duplicates = _duplicate_columns(frame)
    if duplicates:
        raise DuplicateColumnsError(duplicates)

    if frame.empty:
        raise EmptyDatasetError(
            "no data rows; expected the documented header plus at least one transaction"
        )

    missing_columns = tuple(column for column in EXPECTED_COLUMNS if column not in frame.columns)
    if missing_columns:
        raise MissingColumnsError(missing_columns, describe_columns())

    missing_values = _missing_value_counts(frame)
    if missing_values:
        raise MissingValuesError(missing_values)

    numeric_problems = _numeric_problems(frame)
    if numeric_problems:
        raise InvalidNumericError(numeric_problems)

    row_count = len(frame)
    class_counts = _class_counts(frame)

    return DatasetReport(
        path=path,
        row_count=row_count,
        feature_count=len(INPUT_COLUMNS),
        missing_value_count=int(frame[list(EXPECTED_COLUMNS)].isna().sum().sum()),
        duplicate_row_count=int(frame.duplicated().sum()),
        class_counts=class_counts,
        class_proportions={label: count / row_count for label, count in class_counts.items()},
    )


def format_report(report: DatasetReport) -> str:
    """Render a validation report as a short, row-free block for a terminal."""
    lines = ["Validation: OK"]
    if report.path is not None:
        lines.append(f"  path       : {report.path}")
    lines.extend(
        [
            f"  rows       : {report.row_count}",
            f"  features   : {report.feature_count} (+ target {TARGET_COLUMN})",
            f"  missing    : {report.missing_value_count} values",
            f"  duplicates : {report.duplicate_row_count} rows",
        ]
    )
    for label, count in report.class_counts.items():
        proportion = report.class_proportions[label]
        lines.append(f"  class {label}    : {count} ({proportion:.4%})")
    return "\n".join(lines)


def _duplicate_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    duplicated = frame.columns[frame.columns.duplicated()].unique().tolist()
    return tuple(sorted(str(name) for name in duplicated))


def _missing_value_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame[list(EXPECTED_COLUMNS)].isna().sum()
    return {str(column): int(count) for column, count in counts.items() if int(count) > 0}


def _numeric_problems(frame: pd.DataFrame) -> dict[str, str]:
    """Explain, per column, why a non-target column is not usable numeric data."""
    problems: dict[str, str] = {}
    for column in INPUT_COLUMNS:
        series = frame[column]
        if is_bool_dtype(series):
            problems[column] = "boolean column, expected a numeric measurement"
            continue
        if is_complex_dtype(series):
            problems[column] = "complex values are not valid transaction measurements"
            continue
        if not is_numeric_dtype(series):
            problems[column] = f"non-numeric dtype {series.dtype}"
            continue

        # float64 makes the finiteness check uniform across int and nullable numeric dtypes.
        # Nulls were rejected above, so no value can be silently introduced here.
        values = series.to_numpy(dtype="float64")
        non_finite = int((~np.isfinite(values)).sum())
        if non_finite:
            problems[column] = f"{non_finite} non-finite value(s) (inf or -inf)"
    return problems


def _label_text(value: object) -> str:
    """Render a label for an error message without numpy's ``np.int64(2)`` wrapper."""
    return repr(value.item() if isinstance(value, np.generic) else value)


def _class_counts(frame: pd.DataFrame) -> dict[int, int]:
    values = frame[TARGET_COLUMN].to_numpy()
    valid = np.isin(values, BINARY_LABELS)
    if not valid.all():
        invalid = values[~valid]
        shown = tuple(sorted({_label_text(value) for value in invalid})[:10])
        raise InvalidTargetError(TARGET_COLUMN, int((~valid).sum()), shown)

    positive = int((values == 1).sum())
    total = int(values.size)
    return {0: total - positive, 1: positive}
