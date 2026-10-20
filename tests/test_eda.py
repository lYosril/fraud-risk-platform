"""Tests for the exploratory data analysis stage.

Every frame is built in memory from deterministic synthetic values; the real dataset is never
read. The rendering assertions are on **computed** values, never on hardcoded dataset counts.
The two subprocess tests exercise the reproducible script end to end against the committed
synthetic fixture.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

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
from fraud_risk.data.errors import DatasetValidationError
from fraud_risk.data.ingestion import load_csv
from fraud_risk.data.validation import INPUT_COLUMNS
from fraud_risk.features.eda import (
    DatasetOrigin,
    EdaReport,
    analyse,
    classify_origin,
    render_markdown,
    report_to_dict,
    write_report,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EDA_SCRIPT = REPO_ROOT / "notebooks" / "eda.py"
SYNTHETIC_FIXTURE = REPO_ROOT / "data" / "sample" / "creditcard.synthetic.csv"
CUSTOM_PATH = Path("custom.csv")


def sample_frame() -> pd.DataFrame:
    """Six deterministic rows: two frauds, a zero amount, three hourly buckets."""
    rows = 6
    data: dict[str, list[object]] = {
        TIME_COLUMN: [0, 1800, 3600, 5400, 7200, 9000],
        AMOUNT_COLUMN: [0.0, 12.5, 99.99, 5.0, 2500.0, 3.25],
        TARGET_COLUMN: [0, 1, 0, 0, 1, 0],
    }
    for position, column in enumerate(FEATURE_COLUMNS):
        data[column] = [round((((row * 7 + position * 3) % 11) - 5) / 10, 3) for row in range(rows)]
    return pd.DataFrame(data)[list(EXPECTED_COLUMNS)]


def analyse_sample(frame: pd.DataFrame | None = None, *, path: Path = CUSTOM_PATH) -> EdaReport:
    return analyse(sample_frame() if frame is None else frame, path=path)


# ---------------------------------------------------------------------------
# Computed summaries
# ---------------------------------------------------------------------------
def test_shape_schema_and_origin_come_from_the_input() -> None:
    frame = sample_frame()

    report = analyse_sample(frame)

    assert report.validation.row_count == len(frame)
    assert report.validation.feature_count == len(INPUT_COLUMNS)
    assert len(EXPECTED_COLUMNS) == report.validation.feature_count + 1
    assert report.dtypes[TARGET_COLUMN].startswith("int")
    assert report.size_bytes == 0  # the path does not exist; nothing is invented
    assert report.origin is DatasetOrigin.CUSTOM


def test_class_balance_is_computed_not_assumed() -> None:
    report = analyse_sample()

    assert report.class_balance.counts == {0: 4, 1: 2}
    assert report.class_balance.proportions[1] == pytest.approx(2 / 6)
    assert report.class_balance.negatives_per_positive == pytest.approx(2.0)


def test_duplicate_rows_are_counted_and_not_removed() -> None:
    frame = sample_frame()
    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)

    report = analyse_sample(duplicated)

    assert report.validation.row_count == len(frame) + 1
    assert report.validation.duplicate_row_count == 1
    assert report.validation.class_counts[0] == 5


def test_time_is_summarised_in_elapsed_hours() -> None:
    report = analyse_sample()

    assert report.time_summary.span_hours == pytest.approx(2.5)
    assert report.time_summary.is_sorted is True
    assert list(report.hourly.index) == [0, 1, 2]
    assert int(report.hourly["transactions"].sum()) == 6
    assert int(report.hourly["frauds"].sum()) == 2


def test_amount_is_summarised_per_class() -> None:
    table = analyse_sample().amount_by_class

    assert set(table.index) == {0, 1}
    assert table.loc[1, "count"] == pytest.approx(2.0)
    assert table.loc[1, "max"] == pytest.approx(2500.0)
    assert table.loc[0, "zero_rows"] == pytest.approx(1.0)
    assert table.loc[1, "zero_rows"] == pytest.approx(0.0)


def test_feature_statistics_cover_every_input_column() -> None:
    statistics = analyse_sample().feature_statistics

    assert list(statistics.index) == list(INPUT_COLUMNS)
    assert statistics.loc["V1", "count"] == pytest.approx(6.0)
    assert "max_abs_z" in statistics.columns


def test_correlations_are_sorted_and_bounded() -> None:
    correlations = analyse_sample().correlations

    values = correlations["pearson"].dropna()
    assert not values.empty
    assert values.between(-1.0, 1.0).all()
    assert correlations["abs_pearson"].is_monotonic_decreasing


def test_analysis_does_not_mutate_the_input_frame() -> None:
    frame = sample_frame()
    expected = frame.copy(deep=True)

    analyse_sample(frame)

    pd.testing.assert_frame_equal(frame, expected)


def test_invalid_frame_raises_the_typed_validation_error() -> None:
    frame = sample_frame().drop(columns=["V1"])

    with pytest.raises(DatasetValidationError):
        analyse_sample(frame)


def test_single_class_file_is_reported_without_crashing() -> None:
    frame = sample_frame()
    frame[TARGET_COLUMN] = 0

    report = analyse_sample(frame)

    assert report.class_balance.counts == {0: 6, 1: 0}
    assert report.class_balance.negatives_per_positive is None
    assert report.correlations["abs_pearson"].isna().all()
    assert "n/a" in render_markdown(report)


# ---------------------------------------------------------------------------
# Origin labelling
# ---------------------------------------------------------------------------
def test_origin_is_decided_by_resolved_path() -> None:
    assert classify_origin(DEFAULT_DATASET_PATH) is DatasetOrigin.REFERENCE
    assert classify_origin(SAMPLE_DATASET_PATH) is DatasetOrigin.SYNTHETIC
    assert classify_origin(CUSTOM_PATH) is DatasetOrigin.CUSTOM


def test_synthetic_fixture_is_labelled_as_synthetic() -> None:
    frame = load_csv(SYNTHETIC_FIXTURE)

    report = analyse(frame, path=SAMPLE_DATASET_PATH)
    text = render_markdown(report)

    assert report.origin is DatasetOrigin.SYNTHETIC
    assert "SYNTHETIC FIXTURE" in text
    assert "none of it may be quoted as a dataset result" in text
    assert f"Rows: {len(frame):,}" in text


def test_custom_input_is_not_presented_as_the_reference_dataset() -> None:
    text = render_markdown(analyse_sample())

    assert "custom file" in text
    assert "NOT THE REFERENCE DATASET" not in text


def test_report_states_that_correlation_is_not_causation() -> None:
    text = render_markdown(analyse_sample())

    assert "associations in one sample, not causal effects" in text
    assert "evidence that amount causes fraud" in text


def test_report_flags_components_that_contradict_pca() -> None:
    """Identical V columns cannot be genuine PCA output; the report must say so."""
    frame = sample_frame()
    for column in FEATURE_COLUMNS:
        frame[column] = [0.1 * row for row in range(len(frame))]

    text = render_markdown(analyse_sample(frame))

    assert "higher than expected" in text


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------
def test_summary_dict_is_strict_json_and_carries_computed_values() -> None:
    payload = report_to_dict(analyse_sample())

    # allow_nan=False fails on NaN/Infinity, which are not valid JSON.
    json.dumps(payload, allow_nan=False)
    assert payload["input"]["rows"] == len(sample_frame())
    assert payload["class_balance"]["counts"] == {"0": 4, "1": 2}


def test_write_report_writes_tables_markdown_and_json(tmp_path: Path) -> None:
    report = analyse_sample()

    written = write_report(report, tmp_path)

    names = {path.name for path in written}
    assert names == {
        "feature_statistics.csv",
        "correlations.csv",
        "amount_by_class.csv",
        "hourly.csv",
        "summary.md",
        "summary.json",
    }
    statistics = pd.read_csv(tmp_path / "feature_statistics.csv", index_col="feature")
    assert len(statistics) == len(INPUT_COLUMNS)
    payload = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert payload["duplicate_rows"] == report.validation.duplicate_row_count
    assert (
        (tmp_path / "summary.md")
        .read_text(encoding="utf-8")
        .startswith("# Exploratory data analysis")
    )


# ---------------------------------------------------------------------------
# End-to-end script
# ---------------------------------------------------------------------------
def _run_script(*arguments: str) -> subprocess.CompletedProcess[str]:
    if importlib.util.find_spec("matplotlib") is None:
        pytest.skip("the EDA script needs matplotlib, which is a dev-extra dependency")
    environment = {**os.environ, "MPLBACKEND": "Agg"}
    source = str(REPO_ROOT / "src")
    environment["PYTHONPATH"] = os.pathsep.join([source, environment.get("PYTHONPATH", "")]).rstrip(
        os.pathsep
    )
    return subprocess.run(  # noqa: S603 - fixed interpreter and repo-local script
        [sys.executable, str(EDA_SCRIPT), *arguments],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=environment,
        timeout=300,
        check=False,
    )


def test_script_runs_end_to_end_on_the_synthetic_fixture(tmp_path: Path) -> None:
    result = _run_script(
        "--path",
        str(SYNTHETIC_FIXTURE),
        "--output-dir",
        str(tmp_path),
    )

    assert result.returncode == 0, result.stderr
    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert f"Rows: {len(load_csv(SYNTHETIC_FIXTURE)):,}" in summary
    assert "SYNTHETIC FIXTURE" in summary

    figures = sorted((tmp_path / "figures").glob("*.png"))
    assert len(figures) == 6
    for figure in figures:
        assert figure.stat().st_size > 5_000
    json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))


def test_script_refuses_a_missing_dataset(tmp_path: Path) -> None:
    missing = tmp_path / "absent.csv"

    result = _run_script("--path", str(missing), "--output-dir", str(tmp_path))

    assert result.returncode == 2
    assert "EDA aborted" in result.stderr
    assert "absent.csv" in result.stderr
    assert not (tmp_path / "summary.md").exists()
