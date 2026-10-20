"""Tests for dataset contract validation.

Every frame is built in memory from deterministic synthetic values; nothing here reads the
real dataset. Each important failure mode has its own test, and each test asserts on the
typed error rather than only on a message.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pytest

from fraud_risk.data.__main__ import EXIT_INVALID, EXIT_OK, main
from fraud_risk.data.dataset import EXPECTED_COLUMNS, FEATURE_COLUMNS, FILENAME
from fraud_risk.data.errors import (
    DatasetValidationError,
    DuplicateColumnsError,
    EmptyDatasetError,
    InvalidNumericError,
    InvalidTargetError,
    MissingColumnsError,
    MissingValuesError,
)
from fraud_risk.data.validation import (
    INPUT_COLUMNS,
    format_report,
    validate_dataset,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_FIXTURE = REPO_ROOT / "data" / "sample" / "creditcard.synthetic.csv"


def valid_frame(labels: Sequence[object] = (0, 1, 0, 0)) -> pd.DataFrame:
    """A tiny, deterministic frame that satisfies the documented schema."""
    size = len(labels)
    data: dict[str, list[object]] = {
        "Time": [index * 60 for index in range(size)],
        "Amount": [10.0 + index for index in range(size)],
        "Class": list(labels),
    }
    for column in FEATURE_COLUMNS:
        data[column] = [0.1 * (index + 1) for index in range(size)]
    return pd.DataFrame(data)[list(EXPECTED_COLUMNS)]


# ---------------------------------------------------------------------------
# Happy path and reporting
# ---------------------------------------------------------------------------
def test_valid_frame_is_accepted_and_summarised() -> None:
    """The report states rows, features, missing values, duplicates and both classes."""
    report = validate_dataset(valid_frame())

    assert report.row_count == 4
    assert report.feature_count == 30
    assert report.missing_value_count == 0
    assert report.duplicate_row_count == 0
    assert report.class_counts == {0: 3, 1: 1}
    assert report.class_proportions[0] == pytest.approx(0.75)
    assert report.class_proportions[1] == pytest.approx(0.25)


def test_feature_count_excludes_the_target() -> None:
    """30 features plus ``Class``; the target is not counted as a feature."""
    assert len(EXPECTED_COLUMNS) == 31
    assert len(INPUT_COLUMNS) == 30
    assert validate_dataset(valid_frame()).feature_count == 30


def test_proportions_sum_to_one() -> None:
    """Class proportions are a partition of the rows."""
    report = validate_dataset(valid_frame())

    assert sum(report.class_proportions.values()) == pytest.approx(1.0)


def test_report_carries_the_path() -> None:
    """The caller's path is echoed in the report rather than inferred."""
    report = validate_dataset(valid_frame(), path=Path("data/raw/creditcard.csv"))

    assert report.path == Path("data/raw/creditcard.csv")


def test_duplicate_rows_are_counted_not_removed() -> None:
    """Duplicate rows are reported; deleting them is a preprocessing decision."""
    base = valid_frame(labels=(0,))
    frame = pd.concat([base, base], ignore_index=True)

    report = validate_dataset(frame)

    assert report.row_count == 2
    assert report.duplicate_row_count == 1


def test_single_class_is_accepted_and_reported_with_a_zero() -> None:
    """A one-class file is schema-valid; the distribution is reported, not enforced."""
    report = validate_dataset(valid_frame(labels=(0, 0, 0, 0)))

    assert report.class_counts == {0: 4, 1: 0}
    assert report.class_proportions == {0: 1.0, 1: 0.0}


def test_extra_columns_are_ignored() -> None:
    """The contract covers the 31 documented columns; extras do not change the report."""
    frame = valid_frame()
    frame["merchant_note"] = ["a", "b", "c", "d"]

    assert validate_dataset(frame).feature_count == 30


def test_validation_does_not_modify_the_frame() -> None:
    """Validation observes only: same rows, same values, no coercion."""
    frame = valid_frame()
    before = frame.copy(deep=True)

    validate_dataset(frame)

    pd.testing.assert_frame_equal(frame, before)


def test_format_report_shows_counts_without_row_values() -> None:
    """The rendered report is safe to paste: counts and labels, never a transaction value."""
    frame = valid_frame()
    frame.loc[0, "Amount"] = 987654.321

    text = format_report(validate_dataset(frame))

    assert "Validation: OK" in text
    assert "rows       : 4" in text
    assert "features   : 30" in text
    assert "class 1    : 1 (25.0000%)" in text
    assert "987654.321" not in text


# ---------------------------------------------------------------------------
# Structural failures
# ---------------------------------------------------------------------------
def test_duplicate_columns_are_rejected() -> None:
    """A repeated column breaks name-based selection, so it cannot be validated."""
    frame = pd.DataFrame([[0, 0, 0]], columns=["Time", "Time", "Amount"])

    with pytest.raises(DuplicateColumnsError) as excinfo:
        validate_dataset(frame)

    assert excinfo.value.columns == ("Time",)


def test_empty_frame_is_rejected() -> None:
    """No rows means nothing to validate."""
    with pytest.raises(EmptyDatasetError):
        validate_dataset(pd.DataFrame())


def test_header_only_frame_is_rejected() -> None:
    """Correct columns but zero rows is still an empty dataset."""
    with pytest.raises(EmptyDatasetError, match="no data rows"):
        validate_dataset(valid_frame(labels=()))


def test_missing_column_is_rejected() -> None:
    """A single absent column is named precisely."""
    frame = valid_frame().drop(columns=["V7"])

    with pytest.raises(MissingColumnsError) as excinfo:
        validate_dataset(frame)

    assert excinfo.value.columns == ("V7",)
    assert "V7" in str(excinfo.value)


def test_multiple_missing_columns_are_all_reported() -> None:
    """Every absent column is listed in one actionable error."""
    frame = valid_frame().drop(columns=["V3", "Class"])

    with pytest.raises(MissingColumnsError) as excinfo:
        validate_dataset(frame)

    assert excinfo.value.columns == ("V3", "Class")


# ---------------------------------------------------------------------------
# Missing values: reported, never imputed
# ---------------------------------------------------------------------------
def test_missing_value_is_rejected_with_per_column_counts() -> None:
    """Nulls are counted per column and no row is dropped."""
    frame = valid_frame()
    frame.loc[0, "V1"] = float("nan")
    frame.loc[2, "V1"] = float("nan")
    frame.loc[1, "Amount"] = float("nan")

    with pytest.raises(MissingValuesError) as excinfo:
        validate_dataset(frame)

    assert excinfo.value.columns == {"V1": 2, "Amount": 1}
    assert excinfo.value.cell_count == 3
    assert len(frame) == 4  # nothing was removed


def test_missing_target_value_is_rejected_as_missing() -> None:
    """A null label is a missing-value failure, not an out-of-range label."""
    frame = valid_frame()
    frame.loc[1, "Class"] = float("nan")

    with pytest.raises(MissingValuesError) as excinfo:
        validate_dataset(frame)

    assert excinfo.value.columns == {"Class": 1}


# ---------------------------------------------------------------------------
# Invalid numeric values
# ---------------------------------------------------------------------------
def test_non_numeric_column_is_rejected() -> None:
    """Text in a feature column is not silently cast to NaN."""
    frame = valid_frame()
    frame["V1"] = ["a", "b", "c", "d"]

    with pytest.raises(InvalidNumericError) as excinfo:
        validate_dataset(frame)

    assert "V1" in excinfo.value.problems
    assert "non-numeric" in excinfo.value.problems["V1"]


def test_infinite_value_is_rejected() -> None:
    """``inf`` passes a dtype check but breaks scalers, so it is rejected explicitly."""
    frame = valid_frame()
    frame.loc[0, "Amount"] = float("inf")

    with pytest.raises(InvalidNumericError) as excinfo:
        validate_dataset(frame)

    assert "non-finite" in excinfo.value.problems["Amount"]


def test_boolean_column_is_rejected() -> None:
    """``is_numeric_dtype`` accepts booleans; a transaction measurement they are not."""
    frame = valid_frame()
    frame["V2"] = [True, False, True, False]

    with pytest.raises(InvalidNumericError) as excinfo:
        validate_dataset(frame)

    assert "boolean" in excinfo.value.problems["V2"]


# ---------------------------------------------------------------------------
# Target-label validity
# ---------------------------------------------------------------------------
def test_float_labels_zero_and_one_are_accepted() -> None:
    """``1.0``/``0.0`` are the labels 1 and 0; the dtype does not matter."""
    report = validate_dataset(valid_frame(labels=(0.0, 1.0, 0.0, 0.0)))

    assert report.class_counts == {0: 3, 1: 1}


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        ((0, 2, 0, 0), ("2",)),
        ((-1, 0, 0, 0), ("-1",)),
        ((0.5, 0, 0, 0), ("0.5",)),
    ],
)
def test_out_of_range_labels_are_rejected(
    labels: Sequence[object], expected: tuple[str, ...]
) -> None:
    """Only 0 and 1 are labels; anything else is named in the error."""
    with pytest.raises(InvalidTargetError) as excinfo:
        validate_dataset(valid_frame(labels=labels))

    assert excinfo.value.count == 1
    assert excinfo.value.values == expected


def test_string_labels_are_rejected() -> None:
    """A textual target is rejected by value, with the offending values quoted."""
    with pytest.raises(InvalidTargetError) as excinfo:
        validate_dataset(valid_frame(labels=("no", "yes", 0, 0)))

    assert excinfo.value.count == 2
    assert excinfo.value.values == ("'no'", "'yes'")
    assert "0 and 1" in str(excinfo.value)


def test_validation_errors_share_a_typed_base() -> None:
    """Callers can catch the whole data layer or one precise failure."""
    with pytest.raises(DatasetValidationError):
        validate_dataset(valid_frame().drop(columns=["V1"]))

    assert issubclass(MissingValuesError, DatasetValidationError)
    assert issubclass(InvalidTargetError, ValueError)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_cli_reports_validation_ok(capsys: pytest.CaptureFixture[str]) -> None:
    """The committed fixture validates end to end through the command line."""
    assert main(["--path", str(SYNTHETIC_FIXTURE)]) == EXIT_OK

    output = capsys.readouterr().out
    assert "Validation: OK" in output
    assert "rows       : 8" in output
    assert "class 1    : 2 (25.0000%)" in output


def test_cli_reports_validation_failure(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A corrupt contract exits 1 and names the typed failure, never a false 'OK'."""
    path = tmp_path / FILENAME
    valid_frame(labels=(0, 2, 0, 0)).to_csv(path, index=False)

    assert main(["--path", str(path)]) == EXIT_INVALID

    output = capsys.readouterr().out
    assert "Validation: FAILED" in output
    assert "InvalidTargetError" in output
    assert "Validation: OK" not in output


# ---------------------------------------------------------------------------
# No implicit I/O
# ---------------------------------------------------------------------------
def test_importing_the_modules_does_not_load_any_data(tmp_path: Path) -> None:
    """Neither module may read a CSV at import time; a caller must pass a path."""
    script = (
        "import pandas as pd\n"
        "def _refuse(*args, **kwargs):\n"
        "    raise AssertionError('read_csv was called during import')\n"
        "pd.read_csv = _refuse\n"
        "import fraud_risk.data.ingestion\n"
        "import fraud_risk.data.validation\n"
        "print('ok')\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")

    result = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        cwd=tmp_path,
        env=env,
    )

    assert result.stdout.strip() == "ok"
