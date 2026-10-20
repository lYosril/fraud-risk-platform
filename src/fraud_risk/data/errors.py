"""Typed errors for dataset loading and validation.

Kept in one module so a caller can catch :class:`DataError` for the whole data layer while
still being able to handle one specific failure. Contract violations also inherit from
``ValueError``: an invalid dataset is bad input, not an I/O accident.

None of these errors carries row-level data. They name columns, counts and — for labels only,
which are not sensitive — the unexpected label values.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


class DataError(Exception):
    """Base class for every failure raised by the data layer."""


class DatasetFileError(DataError):
    """The file could not be read at all."""

    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"cannot read dataset {path}: {reason}")


class DatasetValidationError(DataError, ValueError):
    """The file was readable but violates the dataset contract."""


class EmptyDatasetError(DatasetValidationError):
    """There are no data rows to validate."""

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(f"dataset is empty: {detail}")


class DuplicateColumnsError(DatasetValidationError):
    """A column name appears more than once, which breaks name-based selection."""

    def __init__(self, columns: Sequence[str]) -> None:
        self.columns: tuple[str, ...] = tuple(columns)
        super().__init__(
            f"duplicate column name(s): {', '.join(self.columns)}; "
            "each column must appear exactly once"
        )


class MissingColumnsError(DatasetValidationError):
    """One or more required columns are absent."""

    def __init__(self, columns: Sequence[str], expected: str) -> None:
        self.columns: tuple[str, ...] = tuple(columns)
        self.expected = expected
        super().__init__(
            f"missing required column(s): {', '.join(self.columns)}; "
            f"the dataset must contain {expected}"
        )


class MissingValuesError(DatasetValidationError):
    """Null/NaN values are present. Validation never imputes or drops rows."""

    def __init__(self, columns: Mapping[str, int]) -> None:
        self.columns: dict[str, int] = dict(columns)
        self.cell_count = sum(self.columns.values())
        detail = ", ".join(f"{name}={count}" for name, count in self.columns.items())
        super().__init__(
            f"{self.cell_count} missing value(s) across {len(self.columns)} column(s): "
            f"{detail}. Resolve them in the source data; validation never imputes or drops "
            "rows."
        )


class InvalidNumericError(DatasetValidationError):
    """A column that must be numeric is not, or contains non-finite values."""

    def __init__(self, problems: Mapping[str, str]) -> None:
        self.problems: dict[str, str] = dict(problems)
        detail = "; ".join(f"{name}: {reason}" for name, reason in self.problems.items())
        super().__init__(f"invalid numeric data: {detail}")


class InvalidTargetError(DatasetValidationError):
    """The label column contains values other than the documented binary labels."""

    def __init__(self, column: str, count: int, values: Sequence[str]) -> None:
        self.column = column
        self.count = count
        self.values: tuple[str, ...] = tuple(values)
        shown = ", ".join(self.values) or "<unprintable>"
        super().__init__(
            f"column {column!r} must contain only the labels 0 and 1; "
            f"{count} value(s) are not: {shown}"
        )
