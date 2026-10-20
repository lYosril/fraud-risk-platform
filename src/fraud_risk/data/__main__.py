"""Command-line entry point: ``python -m fraud_risk.data``.

Reports whether the expected dataset file is present and structurally sound, and can
optionally acquire it through the official Kaggle CLI. It never prints cell values, so the
output is safe to paste into a bug report.

Exit codes
    0  the file exists and passes the shape and contract checks
    1  the file exists but is malformed, unreadable, has the wrong columns, or violates the
       data contract (missing values, non-numeric columns, labels other than 0/1)
    2  the file is missing
    3  acquisition was requested but could not be completed

Usage
    python -m fraud_risk.data
    python -m fraud_risk.data --path data/sample/creditcard.synthetic.csv
    python -m fraud_risk.data --download
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from fraud_risk.data.dataset import (
    DATASET_URL,
    DEFAULT_DATASET_PATH,
    FILENAME,
    DatasetAcquisitionError,
    acquire_with_kaggle_cli,
    describe,
    inspect_dataset,
)
from fraud_risk.data.errors import DataError
from fraud_risk.data.ingestion import load_csv
from fraud_risk.data.validation import format_report, validate_dataset

EXIT_OK = 0
EXIT_INVALID = 1
EXIT_MISSING = 2
EXIT_ACQUISITION_FAILED = 3


def _emit(text: str) -> None:
    """Write CLI output to stdout.

    Deliberately not ``logging``: this is the command's primary output stream, not a
    diagnostic, and ``AGENTS.md`` reserves ``print`` for the dashboard.
    """
    sys.stdout.write(text + "\n")


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the dataset helper."""
    parser = argparse.ArgumentParser(
        prog="python -m fraud_risk.data",
        description=(
            "Check that the credit-card fraud dataset is present and readable, and "
            "optionally acquire it with the official Kaggle CLI."
        ),
        epilog=(
            f"Exit codes: 0 ok, 1 invalid, 2 missing, 3 acquisition failed. Dataset: {DATASET_URL}"
        ),
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"file to inspect (default: {DEFAULT_DATASET_PATH})",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help=(
            "download with the official Kaggle CLI when it is installed and credentials "
            "are already configured; otherwise fail with instructions (never scrapes)"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the helper and return a process exit code."""
    options = build_parser().parse_args(argv)
    path: Path = options.path

    if options.download:
        if path.is_file():
            _emit(f"{path} is already present; not downloading again.")
        else:
            _emit(f"Attempting to acquire {FILENAME} with the official Kaggle CLI ...")
            try:
                acquired = acquire_with_kaggle_cli(path.parent)
            except DatasetAcquisitionError as exc:
                _emit(f"Acquisition failed: {exc}")
                return EXIT_ACQUISITION_FAILED
            _emit(f"Acquired {acquired}")
        _emit("")

    inspection = inspect_dataset(path)
    _emit(describe(inspection))

    if not inspection.exists:
        return EXIT_MISSING
    if not inspection.ok:
        return EXIT_INVALID

    _emit("")
    try:
        report = validate_dataset(load_csv(path), path=path)
    except DataError as exc:
        _emit(f"Validation: FAILED\n  {type(exc).__name__}: {exc}")
        return EXIT_INVALID

    _emit(format_report(report))
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())
