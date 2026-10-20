"""Leakage-safe train / validation / test splitting.

Responsibility
    Cut one validated transaction frame into three disjoint subsets and describe the cut in a
    small, machine-readable record. Feature values are never persisted: the record holds
    sizes, class counts, the feature schema and a SHA-256 digest of each split's row
    positions, so a later run can prove it reproduced the same partition without storing a
    copy of the dataset.

Strategies
    ``temporal`` - the repository default for this dataset (AGENTS.md, ARCHITECTURE 6).
        Rows are ordered by ``Time`` and cut 60/20/20. Each boundary is advanced to the next
        distinct timestamp so that transactions sharing a second cannot straddle two splits;
        without that, near-duplicate rows would leak across the boundary. Fractions are
        honoured as closely as the data allows and the realised class rates are recorded.
    ``stratified`` - random, class-proportional split with a fixed seed.
        This is the standard tool for independent observations and is what the stage's
        requirements ask for. It must not be used to report the final score on this dataset:
        the transactions are time-ordered, so a random split lets the model interpolate
        between neighbouring transactions and produces a number it could not achieve in
        production. The validation and test subsets of a temporal split stay untouched until
        final evaluation.

What this module does not do
    No resampling, no SMOTE, no undersampling. A sampler is a training-time operation: if one
    is ever justified it must live inside the training pipeline, fitted on training rows
    only, never before the split and never touching validation or test.

    Nothing here reads configuration from the environment either. The seed is an explicit
    argument, so a run cannot silently change because a shell variable changed; the CLI wires
    it from ``Settings.random_seed``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

from fraud_risk.config import get_settings
from fraud_risk.data.dataset import (
    DEFAULT_DATASET_PATH,
    TIME_COLUMN,
)
from fraud_risk.data.errors import DataError
from fraud_risk.data.ingestion import load_csv
from fraud_risk.data.validation import validate_dataset
from fraud_risk.features.engineering import (
    EXCLUDED_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    FeatureSchemaError,
    ScalingStrategy,
    build_preprocessor,
    split_features,
)

EXIT_OK = 0
EXIT_INVALID = 1
EXIT_MISSING = 2

#: Where the CLI writes the split record. ``reports/`` is git-ignored.
DEFAULT_METADATA_PATH = Path("reports") / "splits" / "split_metadata.json"


class SplitError(ValueError):
    """The requested partition cannot be produced from this frame."""


class SplitStrategy(StrEnum):
    """How rows are assigned to train, validation and test."""

    TEMPORAL = "temporal"
    STRATIFIED = "stratified"


@dataclass(frozen=True)
class SplitConfig:
    """Everything that determines the partition. Required: strategy and seed.

    Fractions have defaults because 60/20/20 is the documented policy, but the two values
    that decide *which* rows land where are never implicit.
    """

    strategy: SplitStrategy
    seed: int
    train_fraction: float = 0.6
    validation_fraction: float = 0.2
    test_fraction: float = 0.2

    def __post_init__(self) -> None:
        try:
            strategy = SplitStrategy(self.strategy)
        except ValueError as exc:
            msg = f"unknown split strategy {self.strategy!r}"
            raise SplitError(msg) from exc
        object.__setattr__(self, "strategy", strategy)

        fractions = {
            "train_fraction": self.train_fraction,
            "validation_fraction": self.validation_fraction,
            "test_fraction": self.test_fraction,
        }
        for name, value in fractions.items():
            if not 0.0 < value < 1.0:
                msg = f"{name} must be strictly between 0 and 1 (got {value!r})"
                raise SplitError(msg)
        total = sum(fractions.values())
        if abs(total - 1.0) > 1e-9:
            msg = f"split fractions must sum to 1 (got {total!r})"
            raise SplitError(msg)

        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            msg = f"seed must be an int (got {self.seed!r})"
            raise SplitError(msg)
        if not 0 <= self.seed < 2**32:
            msg = f"seed must be in [0, 2**32) (got {self.seed!r})"
            raise SplitError(msg)


@dataclass(frozen=True)
class DatasetSplit:
    """One subset of the frame. Only :meth:`describe` is ever written to disk."""

    name: str
    positions: NDArray[np.int64]
    X: pd.DataFrame = field(repr=False)
    y: pd.Series = field(repr=False)
    time_range: tuple[float, float] | None = None

    @property
    def row_count(self) -> int:
        """Number of rows in this subset."""
        return int(self.positions.size)

    @property
    def positive_count(self) -> int:
        """Number of fraud rows in this subset."""
        return int(self.y.eq(1).sum())

    @property
    def positive_rate(self) -> float:
        """Fraud rate in this subset."""
        return float(self.y.eq(1).mean())

    def describe(self) -> dict[str, Any]:
        """Summarise this subset without exposing feature values."""
        return {
            "rows": self.row_count,
            "class_counts": {
                "0": int(self.y.eq(0).sum()),
                "1": int(self.y.eq(1).sum()),
            },
            "positive_rate": round(self.positive_rate, 8),
            "position_sha256": position_digest(self.positions),
            "time_range_seconds": list(self.time_range) if self.time_range is not None else None,
        }


@dataclass(frozen=True)
class DatasetSplits:
    """The three disjoint subsets plus the information needed to reproduce and audit them."""

    config: SplitConfig
    feature_columns: tuple[str, ...]
    row_count: int
    train: DatasetSplit
    validation: DatasetSplit
    test: DatasetSplit

    def describe(self) -> dict[str, Any]:
        """Return the persisted record: sizes, schema, class counts, digests and choices.

        Deliberately contains no row-level values. Row positions are summarised by digest,
        which is enough to prove two runs produced the same partition.
        """
        schema = {column: str(dtype) for column, dtype in self.train.X.dtypes.items()}
        return {
            "strategy": self.config.strategy.value,
            "seed": self.config.seed,
            "fractions": {
                "train": self.config.train_fraction,
                "validation": self.config.validation_fraction,
                "test": self.config.test_fraction,
            },
            "row_count": self.row_count,
            "feature_count": len(self.feature_columns),
            "feature_columns": list(self.feature_columns),
            "feature_schema": schema,
            "excluded_columns": dict(EXCLUDED_COLUMNS),
            "resampling": "none",
            "splits": {
                "train": self.train.describe(),
                "validation": self.validation.describe(),
                "test": self.test.describe(),
            },
        }


def position_digest(positions: NDArray[np.int64]) -> str:
    """Hash a split's row positions in a platform-independent way.

    Sorting first makes the digest independent of the order in which a strategy happened to
    emit its rows; the fixed ``<i8`` dtype keeps it identical across platforms.
    """
    ordered = np.sort(np.asarray(positions, dtype="<i8"))
    return hashlib.sha256(ordered.tobytes()).hexdigest()


def split_dataset(
    frame: pd.DataFrame,
    config: SplitConfig,
    *,
    feature_columns: Sequence[str] = MODEL_FEATURE_COLUMNS,
) -> DatasetSplits:
    """Split ``frame`` into train, validation and test without leaking rows across them.

    Raises :class:`FeatureSchemaError` when the frame cannot produce the model matrix and
    :class:`SplitError` when the requested partition is impossible (too few rows, a single
    class, a single timestamp, ...). It never returns a "best effort" split.
    """
    features, labels = split_features(frame, feature_columns=feature_columns)
    row_count = len(frame)
    if row_count < 3:
        msg = f"at least 3 rows are needed for three non-empty splits (got {row_count})"
        raise SplitError(msg)

    time_values = _time_values(frame)
    if config.strategy is SplitStrategy.TEMPORAL:
        if time_values is None:
            msg = f"a temporal split needs the {TIME_COLUMN!r} column"
            raise SplitError(msg)
        train_positions, validation_positions, test_positions = _temporal_partition(
            time_values, config
        )
    else:
        labels_array = np.asarray(labels.to_numpy(), dtype="int8")
        train_positions, validation_positions, test_positions = _stratified_partition(
            labels_array, config
        )

    _check_partition((train_positions, validation_positions, test_positions), row_count)

    def make(name: str, positions: NDArray[np.int64]) -> DatasetSplit:
        return DatasetSplit(
            name=name,
            positions=positions,
            X=features.iloc[positions].reset_index(drop=True),
            y=labels.iloc[positions].reset_index(drop=True),
            time_range=_time_range(time_values, positions),
        )

    return DatasetSplits(
        config=config,
        feature_columns=tuple(feature_columns),
        row_count=row_count,
        train=make("train", train_positions),
        validation=make("validation", validation_positions),
        test=make("test", test_positions),
    )


def write_metadata(splits: DatasetSplits, path: Path = DEFAULT_METADATA_PATH) -> Path:
    """Write :meth:`DatasetSplits.describe` as JSON and return the path written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(splits.describe(), indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _time_values(frame: pd.DataFrame) -> NDArray[np.float64] | None:
    """Return ``Time`` as float64, or ``None`` when the column is absent."""
    if TIME_COLUMN not in frame.columns:
        return None
    try:
        return np.asarray(frame.loc[:, TIME_COLUMN].to_numpy(dtype="float64"))
    except (TypeError, ValueError) as exc:
        msg = f"{TIME_COLUMN} must be numeric to order a temporal split"
        raise SplitError(msg) from exc


def _advance_to_timestamp_boundary(times: NDArray[np.float64], cut: int) -> int:
    """Move ``cut`` forward until no timestamp is divided between two splits.

    ``times`` is sorted. If the row before the boundary shares its second with the row
    after it, the boundary is moved past the whole group. This is what stops a transaction
    and its near-duplicate (same second, often the same amount) from landing in different
    subsets.
    """
    cut = max(1, min(cut, int(times.size)))
    while cut < times.size and times[cut - 1] == times[cut]:
        cut += 1
    return cut


def _temporal_partition(
    times: NDArray[np.float64],
    config: SplitConfig,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Chronological 60/20/20 cut, made at timestamp boundaries."""
    row_count = int(times.size)
    order = np.asarray(np.argsort(times, kind="stable"), dtype="int64")
    ordered_times = times[order]

    train_cut = _advance_to_timestamp_boundary(
        ordered_times, int(row_count * config.train_fraction)
    )
    validation_cut = _advance_to_timestamp_boundary(
        ordered_times,
        train_cut + int(row_count * config.validation_fraction),
    )

    if train_cut == 0 or validation_cut <= train_cut or validation_cut >= row_count:
        msg = (
            "a temporal split cannot produce three non-empty subsets: too few distinct "
            f"{TIME_COLUMN} values in {row_count} rows"
        )
        raise SplitError(msg)

    return order[:train_cut], order[train_cut:validation_cut], order[validation_cut:]


def _stratified_partition(
    labels: NDArray[np.int8],
    config: SplitConfig,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Random class-proportional cut with a fixed seed.

    Not the default for time-ordered data: see the module docstring.
    """
    counts = np.bincount(labels)
    if counts.size < 2 or int(counts.min()) < 3:
        msg = (
            "a stratified split needs at least two classes with three rows each "
            f"(got counts {counts.tolist()})"
        )
        raise SplitError(msg)

    positions = np.asarray(np.arange(labels.size), dtype="int64")
    remainder_fraction = config.validation_fraction + config.test_fraction
    try:
        train_positions, remainder = train_test_split(
            positions,
            test_size=remainder_fraction,
            stratify=labels,
            random_state=config.seed,
        )
        validation_positions, test_positions = train_test_split(
            remainder,
            test_size=config.test_fraction / remainder_fraction,
            stratify=labels[remainder],
            random_state=config.seed,
        )
    except ValueError as exc:
        msg = f"stratified split failed: {exc}"
        raise SplitError(msg) from exc

    return (
        np.sort(np.asarray(train_positions, dtype="int64")),
        np.sort(np.asarray(validation_positions, dtype="int64")),
        np.sort(np.asarray(test_positions, dtype="int64")),
    )


def _check_partition(parts: Sequence[NDArray[np.int64]], row_count: int) -> None:
    """Fail loudly unless the three parts are non-empty, disjoint and cover every row."""
    if any(part.size == 0 for part in parts):
        msg = "each of train, validation and test must contain at least one row"
        raise SplitError(msg)

    merged = np.concatenate(parts)
    if int(merged.size) != row_count or int(np.unique(merged).size) != row_count:
        msg = "splits are not a disjoint partition of the rows"
        raise SplitError(msg)
    if int(merged.min()) < 0 or int(merged.max()) >= row_count:
        msg = "split positions fall outside the frame"
        raise SplitError(msg)


def _time_range(
    times: NDArray[np.float64] | None,
    positions: NDArray[np.int64],
) -> tuple[float, float] | None:
    """Return the subset's ``Time`` span, or ``None`` when ``Time`` is absent."""
    if times is None or positions.size == 0:
        return None
    selected = times[positions]
    return float(selected.min()), float(selected.max())


# ---------------------------------------------------------------------------
# CLI: python -m fraud_risk.models.splitting
# ---------------------------------------------------------------------------
def _emit(text: str) -> None:
    """Write CLI output to stdout (``print`` is reserved for the dashboard)."""
    sys.stdout.write(text + "\n")


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the split helper."""
    parser = argparse.ArgumentParser(
        prog="python -m fraud_risk.models.splitting",
        description=(
            "Split the validated dataset into train/validation/test, fit the shared "
            "preprocessor on training rows only, and write a compact split record."
        ),
        epilog=(
            "Exit codes: 0 ok, 1 invalid data or impossible split, 2 missing dataset. "
            "Default strategy is temporal (AGENTS.md); stratified is available for "
            "independent data and is not suitable for the final score on this dataset."
        ),
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"validated dataset file (default: {DEFAULT_DATASET_PATH})",
    )
    parser.add_argument(
        "--strategy",
        choices=[strategy.value for strategy in SplitStrategy],
        default=SplitStrategy.TEMPORAL.value,
        help="partition rule (default: temporal)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed (default: Settings.random_seed)",
    )
    parser.add_argument(
        "--scaling",
        choices=[strategy.value for strategy in ScalingStrategy],
        default=ScalingStrategy.NONE.value,
        help="preprocessor scaling; 'none' is correct for tree models (default: none)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help=f"split record to write (default: {DEFAULT_METADATA_PATH})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the split helper and return a process exit code."""
    options = build_parser().parse_args(argv)
    path: Path = options.path

    if not path.is_file():
        _emit(f"Dataset not found: {path}")
        _emit("Place it at data/raw/creditcard.csv or pass --path (see data/README.md).")
        return EXIT_MISSING

    try:
        frame = load_csv(path)
        report = validate_dataset(frame, path=path)
        seed = get_settings().random_seed if options.seed is None else int(options.seed)
        config = SplitConfig(strategy=SplitStrategy(options.strategy), seed=seed)
        splits = split_dataset(frame, config)
        preprocessor = build_preprocessor(ScalingStrategy(options.scaling))
        preprocessor.fit(splits.train.X)
        validation_matrix = preprocessor.transform(splits.validation.X)
    except (DataError, FeatureSchemaError, SplitError) as exc:
        _emit(f"Splitting: FAILED\n  {type(exc).__name__}: {exc}")
        return EXIT_INVALID

    output = write_metadata(splits, Path(options.output))

    _emit(f"Dataset : {path}")
    _emit(f"Rows    : {report.row_count}  model features: {len(splits.feature_columns)}")
    _emit(f"Strategy: {config.strategy.value}  seed: {config.seed}")
    for split in (splits.train, splits.validation, splits.test):
        _emit(
            f"  {split.name:<10} {split.row_count:>7} rows  "
            f"fraud {split.positive_count:>5}  rate {split.positive_rate:.4%}"
        )
    _emit(
        f"Scaling : {options.scaling}  "
        f"fitted on {splits.train.row_count} training rows; "
        f"validation matrix {validation_matrix.shape[0]} x {validation_matrix.shape[1]}"
    )
    _emit(f"Record  : {output}")
    if config.strategy is SplitStrategy.STRATIFIED:
        _emit(
            "Caution : stratified rows are random; do not report a final score from this "
            "split on time-ordered data (docs/ARCHITECTURE.md, section 6)."
        )
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())
