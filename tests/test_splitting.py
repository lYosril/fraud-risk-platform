"""Tests for leakage-safe train / validation / test splitting.

Frames are deterministic and synthetic; the real dataset is never read. The tests check the
properties that make a split trustworthy rather than one particular row assignment: the three
subsets partition the frame, the label never appears among the features, the same seed
reproduces the same partition, and temporal boundaries never divide a timestamp.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_risk.data.dataset import (
    AMOUNT_COLUMN,
    DEFAULT_DATASET_PATH,
    EXPECTED_COLUMNS,
    FEATURE_COLUMNS,
    SAMPLE_DATASET_PATH,
    TARGET_COLUMN,
    TIME_COLUMN,
)
from fraud_risk.features.engineering import MODEL_FEATURE_COLUMNS, FeatureSchemaError
from fraud_risk.models.splitting import (
    EXIT_INVALID,
    EXIT_MISSING,
    EXIT_OK,
    DatasetSplit,
    DatasetSplits,
    SplitConfig,
    SplitError,
    SplitStrategy,
    main,
    split_dataset,
    write_metadata,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_FIXTURE = REPO_ROOT / "data" / "sample" / "creditcard.synthetic.csv"


def make_frame(rows: int = 300, positives: int = 30, seed: int = 3) -> pd.DataFrame:
    """A deterministic, time-ordered frame with a known fraud rate."""
    rng = np.random.default_rng(seed)
    labels = np.zeros(rows, dtype="int64")
    labels[rng.choice(rows, positives, replace=False)] = 1

    data: dict[str, object] = {TIME_COLUMN: np.arange(rows, dtype="float64") * 60.0}
    for column in FEATURE_COLUMNS:
        data[column] = rng.normal(size=rows)
    data[AMOUNT_COLUMN] = np.round(rng.lognormal(mean=4.0, sigma=1.5, size=rows), 2)
    data[TARGET_COLUMN] = labels
    return pd.DataFrame(data)[list(EXPECTED_COLUMNS)]


def temporal(seed: int = 7) -> SplitConfig:
    return SplitConfig(strategy=SplitStrategy.TEMPORAL, seed=seed)


def stratified(seed: int = 7) -> SplitConfig:
    return SplitConfig(strategy=SplitStrategy.STRATIFIED, seed=seed)


ALL_SPLITS = ("train", "validation", "test")


def parts(splits: DatasetSplits) -> list[DatasetSplit]:
    return [splits.train, splits.validation, splits.test]


# ---------------------------------------------------------------------------
# Partition invariants
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("config", [temporal(), stratified()], ids=["temporal", "stratified"])
def test_splits_partition_every_row_exactly_once(config: SplitConfig) -> None:
    """No overlap, no gaps: every row belongs to exactly one subset."""
    frame = make_frame()

    splits = split_dataset(frame, config)

    merged = np.concatenate([split.positions for split in parts(splits)])
    assert merged.size == len(frame)
    assert np.array_equal(np.sort(merged), np.arange(len(frame)))


@pytest.mark.parametrize("config", [temporal(), stratified()], ids=["temporal", "stratified"])
def test_row_counts_follow_the_configured_fractions(config: SplitConfig) -> None:
    splits = split_dataset(make_frame(), config)

    assert splits.train.row_count == 180
    assert splits.validation.row_count == 60
    assert splits.test.row_count == 60


@pytest.mark.parametrize("config", [temporal(), stratified()], ids=["temporal", "stratified"])
def test_input_frame_is_not_mutated(config: SplitConfig) -> None:
    frame = make_frame()
    before = frame.copy(deep=True)

    split_dataset(frame, config)

    pd.testing.assert_frame_equal(frame, before)


# ---------------------------------------------------------------------------
# Temporal strategy
# ---------------------------------------------------------------------------
def test_temporal_split_is_chronological() -> None:
    """Training is the earlier data, the test set is the later data."""
    splits = split_dataset(make_frame(), temporal())

    assert splits.train.positions.max() < splits.validation.positions.min()
    assert splits.validation.positions.max() < splits.test.positions.min()
    assert splits.train.time_range is not None
    assert splits.validation.time_range is not None
    assert splits.test.time_range is not None
    assert splits.train.time_range[1] < splits.validation.time_range[0]
    assert splits.validation.time_range[1] < splits.test.time_range[0]


def test_temporal_boundaries_never_split_a_timestamp() -> None:
    """Rows sharing a second stay in one subset, so near-duplicates cannot leak."""
    frame = make_frame(rows=105)
    # Eleven groups of ten rows (the last one has five), with wide gaps between groups.
    frame[TIME_COLUMN] = np.repeat(np.arange(11, dtype="float64") * 600.0, [10] * 10 + [5])

    splits = split_dataset(frame, temporal())

    groups: list[set[float]] = []
    for split in parts(splits):
        groups.append(set(np.unique(frame[TIME_COLUMN].to_numpy()[split.positions])))
    assert groups[0].isdisjoint(groups[1])
    assert groups[1].isdisjoint(groups[2])
    assert groups[0].isdisjoint(groups[2])
    # The 60% cut (63 rows) was advanced to the end of the group that contains it.
    assert splits.train.row_count == 70


def test_temporal_split_requires_the_time_column() -> None:
    with pytest.raises(SplitError, match="Time"):
        split_dataset(make_frame().drop(columns=[TIME_COLUMN]), temporal())


def test_temporal_split_rejects_a_single_timestamp() -> None:
    frame = make_frame(rows=30)
    frame[TIME_COLUMN] = 42.0

    with pytest.raises(SplitError, match="distinct"):
        split_dataset(frame, temporal())


# ---------------------------------------------------------------------------
# Stratified strategy
# ---------------------------------------------------------------------------
def test_stratified_split_preserves_the_class_rate() -> None:
    """30 frauds in 300 rows (10%) split 60/20/20 -> 18/6/6."""
    splits = split_dataset(make_frame(rows=300, positives=30), stratified())

    assert splits.train.positive_rate == pytest.approx(0.10, abs=0.02)
    assert splits.validation.positive_rate == pytest.approx(0.10, abs=0.02)
    assert splits.test.positive_rate == pytest.approx(0.10, abs=0.02)
    assert splits.train.positive_count == 18
    assert splits.validation.positive_count == 6
    assert splits.test.positive_count == 6


def test_same_seed_reproduces_the_same_partition() -> None:
    frame = make_frame()

    first = split_dataset(frame, stratified(seed=99))
    second = split_dataset(frame, stratified(seed=99))

    for left, right in zip(parts(first), parts(second), strict=True):
        np.testing.assert_array_equal(left.positions, right.positions)
    assert first.describe() == second.describe()


def test_a_different_seed_changes_the_partition() -> None:
    frame = make_frame()

    first = split_dataset(frame, stratified(seed=1))
    second = split_dataset(frame, stratified(seed=2))

    assert first.train.describe()["position_sha256"] != second.train.describe()["position_sha256"]


def test_stratified_split_rejects_a_single_class() -> None:
    frame = make_frame()
    frame[TARGET_COLUMN] = 0

    with pytest.raises(SplitError, match="two classes"):
        split_dataset(frame, stratified())


def test_stratified_split_rejects_too_few_minority_rows() -> None:
    frame = make_frame(positives=2)

    with pytest.raises(SplitError, match="three rows each"):
        split_dataset(frame, stratified())


# ---------------------------------------------------------------------------
# Label separation and schema
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("config", [temporal(), stratified()], ids=["temporal", "stratified"])
def test_label_and_time_are_not_features(config: SplitConfig) -> None:
    splits = split_dataset(make_frame(), config)

    for split in parts(splits):
        assert list(split.X.columns) == list(MODEL_FEATURE_COLUMNS)
        assert TARGET_COLUMN not in split.X.columns
        assert TIME_COLUMN not in split.X.columns


@pytest.mark.parametrize("config", [temporal(), stratified()], ids=["temporal", "stratified"])
def test_schema_is_identical_across_splits(config: SplitConfig) -> None:
    splits = split_dataset(make_frame(), config)
    reference = splits.train.X

    for split in parts(splits)[1:]:
        assert list(split.X.columns) == list(reference.columns)
        assert split.X.dtypes.equals(reference.dtypes)


@pytest.mark.parametrize("config", [temporal(), stratified()], ids=["temporal", "stratified"])
def test_labels_follow_their_positions(config: SplitConfig) -> None:
    """The row at position ``i`` keeps the label it had in the source frame."""
    frame = make_frame()
    labels = frame[TARGET_COLUMN].reset_index(drop=True)

    splits = split_dataset(frame, config)

    for split in parts(splits):
        expected = labels.iloc[split.positions].reset_index(drop=True)
        pd.testing.assert_series_equal(split.y, expected)


def test_split_rejects_a_frame_without_the_features() -> None:
    with pytest.raises(FeatureSchemaError, match="V1"):
        split_dataset(make_frame().drop(columns=["V1"]), temporal())


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------
def test_fractions_must_sum_to_one() -> None:
    with pytest.raises(SplitError, match="sum to 1"):
        SplitConfig(strategy=SplitStrategy.TEMPORAL, seed=1, train_fraction=0.7)


def test_every_fraction_must_be_positive() -> None:
    with pytest.raises(SplitError, match="strictly between"):
        SplitConfig(
            strategy=SplitStrategy.TEMPORAL,
            seed=1,
            train_fraction=0.8,
            test_fraction=0.0,
        )


def test_fraction_must_be_below_one() -> None:
    with pytest.raises(SplitError, match="strictly between"):
        SplitConfig(strategy=SplitStrategy.TEMPORAL, seed=1, train_fraction=1.0, test_fraction=0.0)


def test_negative_seed_is_rejected() -> None:
    with pytest.raises(SplitError, match="seed"):
        SplitConfig(strategy=SplitStrategy.TEMPORAL, seed=-1)


def test_strategy_string_is_resolved() -> None:
    config = SplitConfig(strategy="temporal", seed=1)  # type: ignore[arg-type]

    assert config.strategy is SplitStrategy.TEMPORAL


# ---------------------------------------------------------------------------
# Persisted record
# ---------------------------------------------------------------------------
def test_metadata_describes_the_split_without_storing_values(tmp_path: Path) -> None:
    frame = make_frame()
    frame.loc[0, AMOUNT_COLUMN] = 987654.321

    path = write_metadata(split_dataset(frame, temporal()), tmp_path / "split.json")

    text = path.read_text(encoding="utf-8")
    record = json.loads(text)

    assert record["strategy"] == "temporal"
    assert record["seed"] == 7
    assert record["row_count"] == len(frame)
    assert record["feature_count"] == len(MODEL_FEATURE_COLUMNS)
    assert record["feature_columns"] == list(MODEL_FEATURE_COLUMNS)
    assert list(record["feature_schema"]) == list(MODEL_FEATURE_COLUMNS)
    assert record["feature_schema"][AMOUNT_COLUMN] == "float64"
    assert record["resampling"] == "none"
    assert set(record["excluded_columns"]) == {TARGET_COLUMN, TIME_COLUMN}
    assert record["splits"]["train"]["rows"] == 180
    assert len(record["splits"]["train"]["position_sha256"]) == 64
    assert sum(record["splits"][name]["rows"] for name in ALL_SPLITS) == len(frame)

    # Compact and value-free: not one Amount value survives into the record.
    assert "987654" not in text
    assert len(text) < 20_000


def test_metadata_is_byte_identical_for_the_same_seed(tmp_path: Path) -> None:
    frame = make_frame()

    first = write_metadata(split_dataset(frame, stratified(seed=5)), tmp_path / "a.json")
    second = write_metadata(split_dataset(frame, stratified(seed=5)), tmp_path / "b.json")

    assert first.read_bytes() == second.read_bytes()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_cli_writes_a_temporal_record(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / "split.json"

    exit_code = main(
        [
            "--path",
            str(SYNTHETIC_FIXTURE),
            "--strategy",
            "temporal",
            "--seed",
            "3",
            "--output",
            str(output),
        ]
    )

    assert exit_code == EXIT_OK
    printed = capsys.readouterr().out
    assert "Strategy: temporal" in printed
    assert "Rows    : 8" in printed
    assert "model features: 29" in printed

    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["row_count"] == 8
    assert sum(record["splits"][name]["rows"] for name in ALL_SPLITS) == 8


def test_cli_explains_a_missing_dataset(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--path", str(tmp_path / "absent.csv")]) == EXIT_MISSING
    assert "Dataset not found" in capsys.readouterr().out


def test_cli_refuses_an_impossible_stratified_split(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The committed fixture has two frauds; a stratified split cannot be honest about it."""
    output = tmp_path / "split.json"

    exit_code = main(
        [
            "--path",
            str(SYNTHETIC_FIXTURE),
            "--strategy",
            "stratified",
            "--seed",
            "1",
            "--output",
            str(output),
        ]
    )

    assert exit_code == EXIT_INVALID
    assert "SplitError" in capsys.readouterr().out
    assert not output.exists()


def test_module_is_runnable_end_to_end(tmp_path: Path) -> None:
    """``python -m fraud_risk.models.splitting`` works in a fresh interpreter."""
    output = tmp_path / "split.json"
    environment = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}

    result = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [
            sys.executable,
            "-m",
            "fraud_risk.models.splitting",
            "--path",
            str(SYNTHETIC_FIXTURE),
            "--strategy",
            "temporal",
            "--seed",
            "0",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
        env=environment,
    )

    assert result.returncode == EXIT_OK, result.stderr
    assert "Strategy: temporal" in result.stdout
    assert output.is_file()


def test_default_paths_point_at_the_documented_locations() -> None:
    """The CLI defaults are the documented dataset and the git-ignored reports tree."""
    assert Path("data") / "raw" / "creditcard.csv" == DEFAULT_DATASET_PATH
    assert SAMPLE_DATASET_PATH.exists()
