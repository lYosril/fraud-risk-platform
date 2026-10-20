"""Raw dataset loading.

Responsibility
    Read an approved local file into a ``pandas.DataFrame`` exactly as stored, and refuse
    files that cannot be read at all. Loading performs **no** cleaning: no dtype coercion,
    no imputation, no row filtering. Contract checks live in ``data.validation``, and
    preprocessing belongs to ``features``.

Rules
    - Read-only with respect to ``data/raw/``; nothing is written back there.
    - No network downloads: the dataset is placed in ``data/raw/`` by the operator.
    - No module-level loading: nothing is read until a caller passes an explicit path.

Status
    Loading is implemented. The dataset fingerprint (content hash, row count, time range)
    is still pending in M1 — see ``docs/ROADMAP.md``.
"""

from __future__ import annotations

import csv
import warnings
from collections import Counter
from pathlib import Path

import pandas as pd

from fraud_risk.data.errors import DatasetFileError, DuplicateColumnsError, EmptyDatasetError


def load_csv(path: Path) -> pd.DataFrame:
    """Read ``path`` into a DataFrame without changing any value.

    The raw header is inspected with the standard library first, because ``pandas`` silently
    renames duplicate headers (``Class``, ``Class.1``). Detecting that before parsing keeps the
    failure actionable instead of producing a frame with invented column names.

    Parameters
    ----------
    path:
        The CSV to read, typically ``data/raw/creditcard.csv``.

    Raises
    ------
    DatasetFileError
        The file is missing, is not readable, is not UTF-8, or has rows whose width does not
        match the header. Ragged rows are refused because pandas would otherwise shift the
        values into the wrong columns without raising.
    EmptyDatasetError
        The file has no header line.
    DuplicateColumnsError
        Two or more columns share a name.

    Nothing is validated about *content* here: column presence, dtypes, missing values and
    labels are the job of :func:`fraud_risk.data.validation.validate_dataset`.
    """
    if not path.is_file():
        raise DatasetFileError(str(path), "file does not exist")

    header = _read_header(path)
    duplicates = tuple(sorted(name for name, count in Counter(header).items() if count > 1))
    if duplicates:
        raise DuplicateColumnsError(duplicates)

    try:
        # ``index_col=False`` stops pandas from inventing an index column; a ragged row would
        # otherwise be shifted silently (first field becomes the index, the last one lands in
        # ``Class``). pandas only warns in that case, so the warning is promoted to an error
        # below: a shifted row is corrupt data, not a detail.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            frame = pd.read_csv(path, index_col=False)
    except pd.errors.EmptyDataError as exc:
        raise EmptyDatasetError(f"{path} contains no header row") from exc
    except pd.errors.ParserError as exc:
        raise DatasetFileError(str(path), f"malformed CSV: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise DatasetFileError(str(path), f"not valid UTF-8 text: {exc}") from exc
    except OSError as exc:
        raise DatasetFileError(str(path), exc.strerror or str(exc)) from exc

    parser_warnings = [
        warning for warning in caught if issubclass(warning.category, pd.errors.ParserWarning)
    ]
    if parser_warnings:
        raise DatasetFileError(str(path), f"malformed CSV: {parser_warnings[0].message}")

    return frame


def _read_header(path: Path) -> tuple[str, ...]:
    """Return the raw header, before pandas can rename duplicate columns."""
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle), None)
    except UnicodeDecodeError as exc:
        raise DatasetFileError(str(path), f"not valid UTF-8 text: {exc}") from exc
    except OSError as exc:
        raise DatasetFileError(str(path), exc.strerror or str(exc)) from exc

    if not header:
        raise EmptyDatasetError(f"{path} is an empty file")
    return tuple(header)
