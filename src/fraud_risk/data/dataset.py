"""The reference dataset: specification, presence check and optional acquisition.

This module owns everything about *getting* and *recognising* the credit-card fraud
dataset. It deliberately does **not** turn the file into a model-ready frame — typed
loading, the dataset fingerprint and the full schema contract belong to
``data.ingestion``/``data.validation`` (M1).

What it does
    1. Declares the expected artifact: filename, 31 columns, target column, and the
       documented directory the operator is expected to place it in.
    2. Inspects a local file for presence, size, header, column set, row count and
       consistent row width. It never echoes a cell value, so its output is safe to paste
       into an issue or a log.
    3. Optionally downloads the dataset with the **official** Kaggle CLI when credentials
       are already configured. Nothing is scraped, and no authentication is bypassed: if
       the CLI or the credentials are absent, acquisition fails loudly with instructions.

Safety
    - Nothing is fabricated: "missing" and "download failed" are reported as such, and a
      claimed-successful download is re-checked on disk before it is believed.
    - Credentials are only ever checked for *presence*; their contents are never read,
      logged or passed on a command line.

See ``docs/DATA_CARD.md`` for provenance and licensing, and ``data/README.md`` for the
operator workflow.
"""

from __future__ import annotations

import csv
import importlib.util
import os
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# -- Dataset identity --------------------------------------------------------
DATASET_NAME = "Credit Card Fraud Detection"
DATASET_PUBLISHER = "Machine Learning Group, Université Libre de Bruxelles (ULB)"
DATASET_URL = "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
#: Kaggle dataset slug, used only by the official CLI.
DATASET_SLUG = "mlg-ulb/creditcardfraud"
FILENAME = "creditcard.csv"
#: Where the operator is expected to put the file. Git-ignored; see ``data/README.md``.
DEFAULT_DATASET_PATH = Path("data") / "raw" / FILENAME
#: The committed synthetic fixture. Same 31 columns, tiny, and **not** a sample of the real
#: data: it exists so the tooling and its tests can run without the dataset present.
SAMPLE_DATASET_PATH = Path("data") / "sample" / "creditcard.synthetic.csv"

TIME_COLUMN = "Time"
AMOUNT_COLUMN = "Amount"
#: The label column in the published file. Maps to the canonical ``is_fraud`` field.
TARGET_COLUMN = "Class"
#: ``V1..V28`` are PCA components; they carry no human meaning (see docs/DATA_CARD.md).
FEATURE_COLUMNS: tuple[str, ...] = tuple(f"V{index}" for index in range(1, 29))
EXPECTED_COLUMNS: tuple[str, ...] = (
    TIME_COLUMN,
    *FEATURE_COLUMNS,
    AMOUNT_COLUMN,
    TARGET_COLUMN,
)


class DatasetAcquisitionError(RuntimeError):
    """The dataset could not be acquired locally.

    Raised instead of returning a "probably fine" result, so a caller can never mistake a
    failed attempt for a completed download.
    """


@dataclass(frozen=True)
class DatasetInspection:
    """Result of a non-destructive presence and shape check.

    Contains no row-level data by construction: only counts, column names and error text.
    """

    path: Path
    exists: bool
    size_bytes: int = 0
    row_count: int = 0
    header: tuple[str, ...] = ()
    missing_columns: tuple[str, ...] = ()
    unexpected_columns: tuple[str, ...] = ()
    duplicate_columns: tuple[str, ...] = ()
    #: 1-based number of the first data row whose field count is wrong, if any.
    malformed_row: int | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        """True only when the file exists and passes the header/readability check."""
        return (
            self.exists
            and self.error is None
            and not self.missing_columns
            and not self.unexpected_columns
            and not self.duplicate_columns
            and self.malformed_row is None
            and self.row_count > 0
        )


def inspect_dataset(path: Path = DEFAULT_DATASET_PATH) -> DatasetInspection:
    """Check that ``path`` looks like the expected dataset, without loading it into memory.

    Streams the file once, counting rows and verifying that every row has the same width
    as the header. It stops at the first malformed row and reports its number rather than
    its contents.
    """
    if not path.is_file():
        return DatasetInspection(path=path, exists=False)

    size_bytes = path.stat().st_size
    if size_bytes == 0:
        return DatasetInspection(path=path, exists=True, error="file is empty")

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header_row = next(reader, None)
            if not header_row:
                return DatasetInspection(
                    path=path, exists=True, size_bytes=size_bytes, error="no header row"
                )

            header = tuple(header_row)
            expected = set(EXPECTED_COLUMNS)
            present = set(header)
            missing = tuple(sorted(expected - present))
            unexpected = tuple(sorted(present - expected))
            duplicates = tuple(sorted(name for name, count in Counter(header).items() if count > 1))

            width = len(header)
            row_count = 0
            malformed_row: int | None = None
            for record in reader:
                row_count += 1
                if len(record) != width:
                    malformed_row = row_count
                    break
    except UnicodeDecodeError:
        return DatasetInspection(
            path=path,
            exists=True,
            size_bytes=size_bytes,
            error="not valid UTF-8 text",
        )
    except csv.Error as exc:
        return DatasetInspection(
            path=path,
            exists=True,
            size_bytes=size_bytes,
            error=f"not readable as CSV: {exc}",
        )
    except OSError as exc:
        return DatasetInspection(
            path=path,
            exists=True,
            size_bytes=size_bytes,
            error=f"cannot read file: {exc.strerror or exc}",
        )

    return DatasetInspection(
        path=path,
        exists=True,
        size_bytes=size_bytes,
        row_count=row_count,
        header=header,
        missing_columns=missing,
        unexpected_columns=unexpected,
        duplicate_columns=duplicates,
        malformed_row=malformed_row,
    )


def describe(inspection: DatasetInspection) -> str:
    """Render an inspection as a short, row-free report for a terminal or a bug report."""
    lines = [f"Dataset: {DATASET_NAME} ({FILENAME})", f"  path      : {inspection.path}"]

    if not inspection.exists:
        lines.extend(
            [
                "  status    : MISSING",
                f"  expected  : {describe_columns()}",
                f"  download  : {DATASET_URL}",
                f"  next step : place {FILENAME} in {inspection.path.parent}, or run with "
                "--download when the Kaggle CLI and credentials are configured",
            ]
        )
        return "\n".join(lines)

    if inspection.error is not None:
        lines.extend(["  status    : UNREADABLE", f"  reason    : {inspection.error}"])
        return "\n".join(lines)

    lines.extend(
        [
            f"  status    : {'OK' if inspection.ok else 'INVALID'}",
            f"  size      : {_format_size(inspection.size_bytes)}",
            f"  data rows : {inspection.row_count}",
            f"  columns   : {describe_columns()}",
            f"  target    : {TARGET_COLUMN} (1 = fraud, 0 = legitimate)",
        ]
    )
    if inspection.missing_columns:
        lines.append(f"  missing   : {', '.join(inspection.missing_columns)}")
    if inspection.unexpected_columns:
        lines.append(f"  unexpected: {', '.join(inspection.unexpected_columns)}")
    if inspection.duplicate_columns:
        lines.append(f"  duplicate : {', '.join(inspection.duplicate_columns)}")
    if inspection.malformed_row is not None:
        lines.append(
            f"  malformed : data row {inspection.malformed_row} does not have "
            f"{len(inspection.header)} fields"
        )
    if inspection.row_count == 0 and not inspection.missing_columns:
        lines.append("  empty     : header found but no data rows")
    lines.append("  note      : presence and shape only; the contract checks run next")
    return "\n".join(lines)


# -- Optional acquisition through the official Kaggle CLI --------------------
def kaggle_config_path(home: Path | None = None) -> Path:
    """Location the official Kaggle CLI reads its credentials from."""
    base = Path.home() if home is None else home
    return base / ".kaggle" / "kaggle.json"


def kaggle_credentials_present(home: Path | None = None) -> bool:
    """Whether the Kaggle CLI would find credentials.

    Checks for the documented environment variables or the credential file **without
    reading either**; this function must never handle secret material.
    """
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return kaggle_config_path(home).is_file()


def kaggle_cli_command() -> list[str] | None:
    """argv prefix for the official Kaggle CLI, or ``None`` when it is not installed."""
    executable = shutil.which("kaggle")
    if executable is not None:
        return [executable]
    if importlib.util.find_spec("kaggle") is not None:
        return [sys.executable, "-m", "kaggle"]
    return None


def acquire_with_kaggle_cli(
    destination: Path = DEFAULT_DATASET_PATH.parent,
    *,
    home: Path | None = None,
) -> Path:
    """Download the dataset with the official Kaggle CLI, as the already-authenticated user.

    Never scrapes the dataset page and never handles credentials itself: it only checks that
    they exist and lets the CLI use them. Raises :class:`DatasetAcquisitionError` when the
    CLI, the credentials or the download itself is unavailable.
    """
    command = kaggle_cli_command()
    if command is None:
        msg = (
            "the Kaggle CLI is not installed. Install it with `python -m pip install kaggle`, "
            f"or download {FILENAME} manually from {DATASET_URL} and place it in {destination}."
        )
        raise DatasetAcquisitionError(msg)

    if not kaggle_credentials_present(home):
        msg = (
            "no Kaggle credentials were found. Put kaggle.json in "
            f"{kaggle_config_path(home)} or export KAGGLE_USERNAME and KAGGLE_KEY. "
            "Credentials are never read or printed by this tool (see data/README.md), and "
            f"nothing is scraped from the website. You can also download {FILENAME} manually "
            f"from {DATASET_URL}."
        )
        raise DatasetAcquisitionError(msg)

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    argv = [
        *command,
        "datasets",
        "download",
        "-d",
        DATASET_SLUG,
        "-p",
        str(destination),
        "--unzip",
    ]
    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell, no user input
        argv, capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        output = (completed.stderr or completed.stdout).strip().splitlines()
        reason = (output[-1] if output else f"exit code {completed.returncode}")[:200]
        msg = f"the Kaggle CLI failed: {reason}"
        raise DatasetAcquisitionError(msg)

    target = destination / FILENAME
    if not target.is_file():
        msg = f"the Kaggle CLI reported success but {target} does not exist"
        raise DatasetAcquisitionError(msg)
    return target


def describe_columns() -> str:
    """Human-readable description of the documented column set."""
    return (
        f"{TIME_COLUMN}, V1..V28, {AMOUNT_COLUMN}, {TARGET_COLUMN} "
        f"({len(EXPECTED_COLUMNS)} columns)"
    )


def _format_size(size_bytes: int) -> str:
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MiB"
    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KiB"
    return f"{size_bytes} bytes"
