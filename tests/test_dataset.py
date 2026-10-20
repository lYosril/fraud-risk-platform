"""Tests for the dataset specification, presence check and acquisition guards.

Every fixture here is synthetic and tiny. The real credit-card dataset is never downloaded,
read or asserted against, and no Kaggle subprocess is ever executed: the acquisition tests
replace ``subprocess.run`` or stop before it is reached.
"""

from __future__ import annotations

import csv
import importlib.util
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from fraud_risk.data import dataset
from fraud_risk.data.__main__ import (
    EXIT_ACQUISITION_FAILED,
    EXIT_INVALID,
    EXIT_MISSING,
    EXIT_OK,
    main,
)
from fraud_risk.data.dataset import (
    DATASET_SLUG,
    DEFAULT_DATASET_PATH,
    EXPECTED_COLUMNS,
    FEATURE_COLUMNS,
    FILENAME,
    TARGET_COLUMN,
    DatasetAcquisitionError,
    describe,
    inspect_dataset,
    kaggle_cli_command,
    kaggle_credentials_present,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_FIXTURE = REPO_ROOT / "data" / "sample" / "creditcard.synthetic.csv"


def write_dataset(
    path: Path,
    rows: list[list[str]],
    header: Sequence[str] = EXPECTED_COLUMNS,
) -> Path:
    """Write a small CSV fixture and return its path."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def valid_row(*, time: str = "0", amount: str = "10.0", label: str = "0") -> list[str]:
    """One row with exactly the expected width (1 + 28 + 1 + 1 fields)."""
    return [time, *(["0.0"] * len(FEATURE_COLUMNS)), amount, label]


@pytest.fixture()
def no_kaggle_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure the developer's own Kaggle credentials cannot influence a test."""
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)


# ---------------------------------------------------------------------------
# Specification
# ---------------------------------------------------------------------------
def test_expected_header_matches_the_published_file() -> None:
    """The published file has 31 columns, with ``Class`` last."""
    assert len(EXPECTED_COLUMNS) == 31
    assert EXPECTED_COLUMNS[0] == "Time"
    assert EXPECTED_COLUMNS[1:29] == FEATURE_COLUMNS
    assert EXPECTED_COLUMNS[-2:] == ("Amount", "Class")
    assert TARGET_COLUMN == "Class"


def test_default_location_is_the_git_ignored_raw_directory() -> None:
    """The documented drop point is ``data/raw/creditcard.csv``."""
    expected = Path("data/raw/creditcard.csv")

    assert expected == DEFAULT_DATASET_PATH
    assert DEFAULT_DATASET_PATH.name == FILENAME


def test_committed_synthetic_fixture_is_valid() -> None:
    """The tracked fixture has the right shape, and is clearly not the real dataset."""
    inspection = inspect_dataset(SYNTHETIC_FIXTURE)

    assert inspection.ok is True
    assert inspection.row_count == 8
    assert inspection.header == EXPECTED_COLUMNS


# ---------------------------------------------------------------------------
# Presence and shape
# ---------------------------------------------------------------------------
def test_missing_file_is_reported_as_missing(tmp_path: Path) -> None:
    """A missing file is reported, with the source URL and the expected location."""
    inspection = inspect_dataset(tmp_path / "absent.csv")

    assert inspection.exists is False
    assert inspection.ok is False

    report = describe(inspection)
    assert "MISSING" in report
    assert "kaggle.com/datasets/mlg-ulb/creditcardfraud" in report
    assert str(inspection.path) in report


def test_valid_file_reports_ok(tmp_path: Path) -> None:
    """A well-formed file passes and its row count is real, not assumed."""
    path = write_dataset(tmp_path / FILENAME, [valid_row(), valid_row(time="60")])

    inspection = inspect_dataset(path)

    assert inspection.ok is True
    assert inspection.row_count == 2
    assert inspection.malformed_row is None
    assert "OK" in describe(inspection)


@pytest.mark.parametrize("dropped", ["Time", "V7", "Amount", "Class"])
def test_missing_required_column_is_reported(tmp_path: Path, dropped: str) -> None:
    """Each required column is checked; the report names the one that is absent."""
    header = tuple(column for column in EXPECTED_COLUMNS if column != dropped)
    path = write_dataset(tmp_path / FILENAME, [["0"] * len(header)], header=header)

    inspection = inspect_dataset(path)

    assert inspection.ok is False
    assert inspection.missing_columns == (dropped,)
    assert dropped in describe(inspection)


def test_unexpected_column_is_reported(tmp_path: Path) -> None:
    """A different file with extra columns is rejected rather than silently accepted."""
    header = (*EXPECTED_COLUMNS, "Extra")
    path = write_dataset(tmp_path / FILENAME, [["0"] * len(header)], header=header)

    inspection = inspect_dataset(path)

    assert inspection.ok is False
    assert inspection.unexpected_columns == ("Extra",)
    assert "Extra" in describe(inspection)


def test_column_order_does_not_matter(tmp_path: Path) -> None:
    """Columns are matched by name (as pandas does), not by position."""
    reordered = (EXPECTED_COLUMNS[-1], *EXPECTED_COLUMNS[:-1])
    path = write_dataset(tmp_path / FILENAME, [["0"] * len(reordered)], header=reordered)

    assert inspect_dataset(path).ok is True


def test_duplicate_column_is_reported(tmp_path: Path) -> None:
    """A repeated column would break name-based selection, so it is not accepted."""
    header = (*EXPECTED_COLUMNS, TARGET_COLUMN)
    path = write_dataset(tmp_path / FILENAME, [["0"] * len(header)], header=header)

    inspection = inspect_dataset(path)

    assert inspection.ok is False
    assert inspection.duplicate_columns == (TARGET_COLUMN,)
    assert "duplicate" in describe(inspection)


def test_empty_file_is_reported_as_unreadable(tmp_path: Path) -> None:
    """A zero-byte file is an error, not an empty dataset."""
    path = tmp_path / FILENAME
    path.write_text("", encoding="utf-8")

    inspection = inspect_dataset(path)

    assert inspection.exists is True
    assert inspection.ok is False
    assert "UNREADABLE" in describe(inspection)
    assert "empty" in describe(inspection).lower()


def test_header_without_data_rows_is_invalid(tmp_path: Path) -> None:
    """A header-only file is not a usable dataset."""
    path = write_dataset(tmp_path / FILENAME, [])

    inspection = inspect_dataset(path)

    assert inspection.row_count == 0
    assert inspection.ok is False
    assert "no data rows" in describe(inspection)


def test_malformed_row_is_reported_by_number(tmp_path: Path) -> None:
    """A row of the wrong width is located by position; the scan stops there."""
    rows = [
        valid_row(time="0"),
        valid_row(time="60"),
        ["2", *(["0.0"] * 27), "3.0"],  # 29 fields instead of 31
        valid_row(time="180"),
    ]
    path = write_dataset(tmp_path / FILENAME, rows)

    inspection = inspect_dataset(path)

    assert inspection.ok is False
    assert inspection.malformed_row == 3
    assert "data row 3" in describe(inspection)


# ---------------------------------------------------------------------------
# No row-level leakage into the report
# ---------------------------------------------------------------------------
def test_report_never_echoes_cell_values(tmp_path: Path) -> None:
    """A value from the file must not appear in the report."""
    sentinel = "SENTINEL-DO-NOT-ECHO"
    path = write_dataset(tmp_path / FILENAME, [valid_row(amount=sentinel)])

    report = describe(inspect_dataset(path))

    # Guard against a vacuous pass: the sentinel really is in the file.
    assert sentinel in path.read_text(encoding="utf-8")
    assert sentinel not in report


def test_report_never_echoes_values_from_a_malformed_row(tmp_path: Path) -> None:
    """Error reporting names the row number, never its contents."""
    sentinel = "LEAKED-VALUE"
    path = write_dataset(tmp_path / FILENAME, [["0", *(["0.0"] * 26), sentinel, "0"]])

    inspection = inspect_dataset(path)
    report = describe(inspection)

    assert inspection.malformed_row == 1
    assert sentinel in path.read_text(encoding="utf-8")
    assert sentinel not in report


# ---------------------------------------------------------------------------
# CLI exit codes
# ---------------------------------------------------------------------------
def test_cli_exit_code_ok_for_a_valid_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """0 means the file is present and structurally sound."""
    path = write_dataset(tmp_path / FILENAME, [valid_row()])

    assert main(["--path", str(path)]) == EXIT_OK
    assert "OK" in capsys.readouterr().out


def test_cli_exit_code_missing_when_the_file_is_absent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """2 means the operator still has to supply the data."""
    assert main(["--path", str(tmp_path / "absent.csv")]) == EXIT_MISSING
    assert "MISSING" in capsys.readouterr().out


def test_cli_exit_code_invalid_for_a_wrong_header(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """1 means a file exists but is the wrong file (or is damaged)."""
    header = tuple(column for column in EXPECTED_COLUMNS if column != TARGET_COLUMN)
    path = write_dataset(tmp_path / FILENAME, [["0"] * len(header)], header=header)

    assert main(["--path", str(path)]) == EXIT_INVALID
    assert "INVALID" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Acquisition guards: never scrapes, never fabricates a success
# ---------------------------------------------------------------------------
def test_credentials_absent_without_file_or_environment(
    tmp_path: Path, no_kaggle_environment: None
) -> None:
    """No credential file and no variables means "not configured"."""
    assert kaggle_credentials_present(home=tmp_path) is False


def test_credentials_detected_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The documented environment variables are recognised."""
    monkeypatch.setenv("KAGGLE_USERNAME", "example-user")
    monkeypatch.setenv("KAGGLE_KEY", "placeholder-not-a-real-key")

    assert kaggle_credentials_present(home=tmp_path) is True


def test_credentials_detected_from_the_config_file_by_presence_only(
    tmp_path: Path, no_kaggle_environment: None
) -> None:
    """Only existence is checked: the contents are never parsed or validated here."""
    config = tmp_path / ".kaggle" / "kaggle.json"
    config.parent.mkdir()
    config.write_text("{}", encoding="utf-8")  # not valid credentials, deliberately

    assert kaggle_credentials_present(home=tmp_path) is True


def test_cli_command_is_absent_when_kaggle_is_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the CLI there is nothing to invoke."""
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)

    assert kaggle_cli_command() is None


def test_acquisition_without_the_cli_raises_with_instructions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The error says how to install it, instead of failing obscurely."""
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)

    with pytest.raises(DatasetAcquisitionError, match="not installed"):
        dataset.acquire_with_kaggle_cli(tmp_path)


def test_acquisition_without_credentials_never_runs_a_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_kaggle_environment: None
) -> None:
    """A missing secret is caught before anything is launched."""
    monkeypatch.setattr(shutil, "which", lambda _name: "kaggle")

    def fail_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise AssertionError("a subprocess was launched without credentials")

    monkeypatch.setattr(subprocess, "run", fail_run)

    with pytest.raises(DatasetAcquisitionError, match="credentials"):
        dataset.acquire_with_kaggle_cli(tmp_path, home=tmp_path)


def test_acquisition_invokes_the_official_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On success the documented Kaggle CLI arguments are used, with no credentials on them."""
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        (tmp_path / FILENAME).write_text("stub\n", encoding="utf-8")
        return subprocess.CompletedProcess(args=argv, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(shutil, "which", lambda _name: "kaggle")
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setenv("KAGGLE_USERNAME", "example-user")
    monkeypatch.setenv("KAGGLE_KEY", "placeholder-not-a-real-key")

    acquired = dataset.acquire_with_kaggle_cli(tmp_path, home=tmp_path)

    assert acquired == tmp_path / FILENAME
    argv = calls[0]
    assert argv[1:5] == ["datasets", "download", "-d", DATASET_SLUG]
    assert "--unzip" in argv
    assert all("placeholder-not-a-real-key" not in argument for argument in argv)


def test_a_failed_download_is_raised_not_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-zero exit becomes an actionable error."""
    monkeypatch.setattr(shutil, "which", lambda _name: "kaggle")
    monkeypatch.setenv("KAGGLE_USERNAME", "example-user")
    monkeypatch.setenv("KAGGLE_KEY", "placeholder-not-a-real-key")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="401 Unauthorized"
        ),
    )

    with pytest.raises(DatasetAcquisitionError, match="401 Unauthorized"):
        dataset.acquire_with_kaggle_cli(tmp_path, home=tmp_path)


def test_a_reported_success_without_a_file_is_not_believed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exit code 0 alone is not evidence: the artifact is checked on disk."""
    monkeypatch.setattr(shutil, "which", lambda _name: "kaggle")
    monkeypatch.setenv("KAGGLE_USERNAME", "example-user")
    monkeypatch.setenv("KAGGLE_KEY", "placeholder-not-a-real-key")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        ),
    )

    with pytest.raises(DatasetAcquisitionError, match="does not exist"):
        dataset.acquire_with_kaggle_cli(tmp_path, home=tmp_path)


def test_download_flag_reports_failure_instead_of_claiming_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--download`` without the CLI exits 3 and never pretends a file arrived."""
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)

    exit_code = main(["--path", str(tmp_path / FILENAME), "--download"])

    output = capsys.readouterr().out
    assert exit_code == EXIT_ACQUISITION_FAILED
    assert "Acquisition failed" in output
    assert not (tmp_path / FILENAME).exists()


def test_download_is_skipped_when_the_file_already_exists(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An existing file is never overwritten by an unattended download."""
    path = write_dataset(tmp_path / FILENAME, [valid_row()])

    assert main(["--path", str(path), "--download"]) == EXIT_OK
    assert "already present" in capsys.readouterr().out
