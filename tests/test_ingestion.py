"""Tests for raw dataset loading.

Fixtures are tiny, deterministic and synthetic. The real dataset is never read: the
committed fixture in ``data/sample/`` is the only file used, plus temporary files written by
the tests themselves.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

from fraud_risk.data.dataset import EXPECTED_COLUMNS, FEATURE_COLUMNS, FILENAME
from fraud_risk.data.errors import (
    DatasetFileError,
    DuplicateColumnsError,
    EmptyDatasetError,
)
from fraud_risk.data.ingestion import load_csv

REPO_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_FIXTURE = REPO_ROOT / "data" / "sample" / "creditcard.synthetic.csv"


def row(
    *,
    time: str = "0",
    values: str = "0.0",
    amount: str = "10.0",
    label: str = "0",
) -> list[str]:
    """One row with exactly the documented width (1 + 28 + 1 + 1 fields)."""
    return [time, *([values] * len(FEATURE_COLUMNS)), amount, label]


def write_csv(path: Path, rows: list[list[str]] | None = None) -> Path:
    """Write a CSV with the documented header and the given rows."""
    payload = rows if rows is not None else [row(time=str(index)) for index in range(3)]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(EXPECTED_COLUMNS)
        writer.writerows(payload)
    return path


def test_loads_the_committed_synthetic_fixture() -> None:
    """The tracked fixture loads with its header intact and every row present."""
    frame = load_csv(SYNTHETIC_FIXTURE)

    assert list(frame.columns) == list(EXPECTED_COLUMNS)
    assert frame.shape == (8, 31)


def test_all_rows_are_loaded(tmp_path: Path) -> None:
    """Loading never filters rows; that decision belongs to preprocessing."""
    path = write_csv(tmp_path / FILENAME, [row(time=str(index)) for index in range(5)])

    assert len(load_csv(path)) == 5


def test_missing_file_raises_a_typed_error(tmp_path: Path) -> None:
    """A missing path is actionable and typed, not a bare FileNotFoundError."""
    missing = tmp_path / "nope.csv"

    with pytest.raises(DatasetFileError) as excinfo:
        load_csv(missing)

    assert excinfo.value.path.endswith("nope.csv")
    assert "does not exist" in str(excinfo.value)


def test_empty_file_raises_empty_dataset_error(tmp_path: Path) -> None:
    """A zero-byte file has no header and is reported as empty."""
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")

    with pytest.raises(EmptyDatasetError, match="empty"):
        load_csv(path)


def test_header_only_file_loads_zero_rows(tmp_path: Path) -> None:
    """Header-only is a valid *load* and an invalid *dataset* — the split is deliberate."""
    path = write_csv(tmp_path / FILENAME, [])

    frame = load_csv(path)

    assert len(frame) == 0
    assert list(frame.columns) == list(EXPECTED_COLUMNS)


def test_duplicate_header_is_rejected_before_pandas_renames_it(tmp_path: Path) -> None:
    """pandas would silently produce ``Class`` and ``Class.1``; loading refuses instead."""
    path = tmp_path / FILENAME
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([*EXPECTED_COLUMNS, "Class"])
        writer.writerow([*(["0"] * len(EXPECTED_COLUMNS)), "0"])

    # Demonstrates the hazard this guard exists for.
    renamed = pd.read_csv(path)
    assert "Class.1" in renamed.columns

    with pytest.raises(DuplicateColumnsError) as excinfo:
        load_csv(path)
    assert excinfo.value.columns == ("Class",)


def test_malformed_csv_raises_a_file_error(tmp_path: Path) -> None:
    """A row with too many fields is a parse failure, not a silent truncation."""
    path = tmp_path / FILENAME
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(EXPECTED_COLUMNS)
        writer.writerow([*(["0"] * len(EXPECTED_COLUMNS)), "extra"])

    with pytest.raises(DatasetFileError, match="malformed CSV"):
        load_csv(path)


def test_non_utf8_file_raises_a_file_error(tmp_path: Path) -> None:
    """Binary rubbish is reported as a readable failure, not a decode traceback."""
    path = tmp_path / FILENAME
    path.write_bytes(b"\xff\xfe\x00Time,V1\n")

    with pytest.raises(DatasetFileError, match="not valid UTF-8"):
        load_csv(path)


def test_loading_does_not_coerce_invalid_values(tmp_path: Path) -> None:
    """A non-numeric token survives loading untouched, so validation can see it."""
    path = write_csv(
        tmp_path / FILENAME,
        [row(time="0"), row(time="60", values="not-a-number")],
    )

    frame = load_csv(path)

    # The last PCA column must still hold the text, not NaN and not a coerced zero.
    assert "not-a-number" in frame["V28"].tolist()


def test_loading_does_not_modify_the_file(tmp_path: Path) -> None:
    """Ingestion is read-only with respect to the raw data."""
    path = write_csv(tmp_path / FILENAME)
    before = path.read_bytes()

    load_csv(path)

    assert path.read_bytes() == before
