"""Tests for the SHAP explainability layer.

Synthetic data and a small calibrated XGBoost pipeline only; no test reads the real dataset.
The suite covers the properties the module promises: it explains the selected tree family,
displays the exact model inputs, keeps probability and decision policy out of the explanation,
checks SHAP additivity, returns the documented schema, and writes reports without a global
row-level data export.
"""

from __future__ import annotations

import csv
import dataclasses
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from fraud_risk.data.dataset import (
    AMOUNT_COLUMN,
    EXPECTED_COLUMNS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    TIME_COLUMN,
)
from fraud_risk.features.engineering import (
    MODEL_FEATURE_COLUMNS,
    ScalingStrategy,
    build_preprocessor,
)
from fraud_risk.models.evaluation import CostPolicy
from fraud_risk.models.explainability import (
    ADDITIVITY_TOLERANCE,
    DEFAULT_MAX_ROWS,
    EXPLAINABILITY_SCHEMA_VERSION,
    OUTPUT_SPACE,
    ContributionDirection,
    ExplanationError,
    GlobalExplanation,
    TransactionExplanation,
    explain_transaction,
    explain_transactions,
    global_feature_importance,
    load_explanation_model,
    render_markdown,
    write_explanation_report,
)
from fraud_risk.models.final_evaluation import RUN_FILENAME
from fraud_risk.models.splitting import (
    DatasetSplits,
    SplitConfig,
    SplitStrategy,
    split_dataset,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "notebooks" / "explainability.py"


def make_frame(rows: int = 300, positives: int = 40, seed: int = 17) -> pd.DataFrame:
    """A deterministic frame with signal, an imbalance and the dataset schema."""
    rng = np.random.default_rng(seed)
    labels = np.zeros(rows, dtype="int64")
    labels[rng.choice(rows, positives, replace=False)] = 1

    data: dict[str, object] = {TIME_COLUMN: np.arange(rows, dtype="float64") * 60.0}
    for column in FEATURE_COLUMNS:
        data[column] = rng.normal(size=rows) + 1.2 * labels
    data[AMOUNT_COLUMN] = np.round(rng.lognormal(mean=4.0, sigma=1.5, size=rows), 2)
    data[TARGET_COLUMN] = labels
    return pd.DataFrame(data)[list(EXPECTED_COLUMNS)]


def make_splits(frame: pd.DataFrame | None = None) -> DatasetSplits:
    return split_dataset(
        make_frame() if frame is None else frame,
        SplitConfig(strategy=SplitStrategy.STRATIFIED, seed=3),
    )


def make_bundle(
    directory: Path,
    splits: DatasetSplits,
    *,
    scaling: ScalingStrategy = ScalingStrategy.NONE,
    family: str = "xgboost",
) -> Path:
    """A small calibrated XGBoost pipeline in the comparison bundle's on-disk shape."""
    positives = int(splits.train.y.sum())
    negatives = len(splits.train.y) - positives
    pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(scaling)),
            (
                "classifier",
                CalibratedClassifierCV(
                    XGBClassifier(
                        n_estimators=20,
                        max_depth=3,
                        learning_rate=0.2,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        scale_pos_weight=negatives / positives,
                        tree_method="hist",
                        eval_metric="logloss",
                        verbosity=0,
                        n_jobs=1,
                        random_state=3,
                    ),
                    method="sigmoid",
                    cv=3,
                ),
            ),
        ]
    )
    pipeline.fit(splits.train.X, splits.train.y)
    bundle: dict[str, Any] = {
        "schema_version": 2,
        "created_at": "2026-01-01T00:00:00+00:00",
        "seed": splits.config.seed,
        "feature_columns": list(MODEL_FEATURE_COLUMNS),
        "threshold": 0.5,
        "cost_policy": CostPolicy(100.0, 1.0).to_dict(),
        "model": {
            "family": family,
            "params": {"n_estimators": 20, "max_depth": 3},
            "imbalance": {"method": "scale_pos_weight"},
            "scaling": scaling.value,
            "calibration": {"method": "sigmoid", "folds": 3},
        },
        "pipeline": pipeline,
    }
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "model.joblib"
    joblib.dump(bundle, path)
    (directory / RUN_FILENAME).write_text(
        json.dumps({"split": splits.describe()}, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture(scope="module")
def fixture(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """One fitted tree bundle and one scaled tree bundle shared by the read-only tests."""
    root = tmp_path_factory.mktemp("explainability")
    frame = make_frame()
    splits = make_splits(frame)
    dataset_path = root / "creditcard.csv"
    frame.to_csv(dataset_path, index=False)
    plain_path = make_bundle(root / "plain", splits)
    scaled_path = make_bundle(
        root / "scaled",
        splits,
        scaling=ScalingStrategy.STANDARD,
    )
    return {
        "root": root,
        "frame": frame,
        "splits": splits,
        "dataset_path": dataset_path,
        "bundle_path": plain_path,
        "scaled_bundle_path": scaled_path,
        "model": load_explanation_model(plain_path),
        "scaled_model": load_explanation_model(scaled_path),
    }


def _feature_index(name: str) -> int:
    return list(MODEL_FEATURE_COLUMNS).index(name)


# ---------------------------------------------------------------------------
# Bundle loading and schema
# ---------------------------------------------------------------------------
def test_loads_the_selected_xgboost_family_and_caches_explainers(fixture: dict[str, Any]) -> None:
    model = fixture["model"]

    assert model.family == "xgboost"
    assert model.n_folds == 3
    assert model.feature_columns == MODEL_FEATURE_COLUMNS
    assert len(model.explainers) == len(model.estimators) == 3
    assert len(model.base_values) == 3
    assert np.isfinite(model.base_values).all()
    assert np.isfinite(model.calibration_slopes).all()
    assert model.to_dict()["output_space"] == OUTPUT_SPACE


def test_unsupported_family_is_rejected_not_silently_explained(
    tmp_path: Path,
    fixture: dict[str, Any],
) -> None:
    path = make_bundle(
        tmp_path / "logistic",
        fixture["splits"],
        family="logistic_regression",
    )

    with pytest.raises(ExplanationError, match="xgboost"):
        load_explanation_model(path)


def test_missing_bundle_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ExplanationError, match="not found"):
        load_explanation_model(tmp_path / "absent.joblib")


def test_explanation_schema_has_no_probability_or_threshold_fields() -> None:
    fields = {field.name for field in dataclasses.fields(TransactionExplanation)}
    global_fields = {field.name for field in dataclasses.fields(GlobalExplanation)}

    assert "probability" not in fields | global_fields
    assert "threshold" not in fields | global_fields
    assert "decision" not in fields | global_fields


# ---------------------------------------------------------------------------
# Individual explanations
# ---------------------------------------------------------------------------
def test_transaction_explanation_has_the_documented_schema(fixture: dict[str, Any]) -> None:
    model = fixture["model"]
    features = fixture["splits"].validation.X.iloc[[0]]

    explanation = explain_transaction(model, features, top_k=5)
    record = explanation.to_dict()

    assert record["schema_version"] == EXPLAINABILITY_SCHEMA_VERSION
    assert set(record) == {"schema_version", "model", "explanation", "limitations"}
    assert set(record["model"]) == {"path", "sha256", "family"}
    assert set(record["explanation"]) >= {
        "output_space",
        "explainer",
        "feature_perturbation",
        "aggregation",
        "n_folds",
        "feature_count",
        "feature_columns",
        "top_k",
        "base_value",
        "output_value",
        "other_features_contribution",
        "additivity_residual",
        "contributions",
    }
    assert record["explanation"]["top_k"] == 5
    assert record["explanation"]["feature_columns"] == list(MODEL_FEATURE_COLUMNS)
    assert len(record["explanation"]["contributions"]) == 5
    assert "probability" not in record["explanation"]
    assert "threshold" not in record["explanation"]
    assert any("not proof of causation" in item for item in record["limitations"])
    assert any("PCA" in item for item in record["limitations"])


def test_contributions_are_ranked_and_additive(fixture: dict[str, Any]) -> None:
    model = fixture["model"]
    features = fixture["splits"].validation.X.iloc[[0]]

    explanation = explain_transaction(model, features, top_k=8)

    absolute = [item.abs_contribution for item in explanation.contributions]
    assert absolute == sorted(absolute, reverse=True)
    assert [item.rank for item in explanation.contributions] == list(range(1, 9))
    assert explanation.base_value + sum(
        item.contribution for item in explanation.contributions
    ) + explanation.other_features_contribution == pytest.approx(
        explanation.output_value,
        abs=ADDITIVITY_TOLERANCE,
    )
    assert abs(explanation.additivity_residual) <= ADDITIVITY_TOLERANCE
    for item in explanation.contributions:
        assert item.direction is (
            ContributionDirection.INCREASES
            if item.contribution > 0
            else ContributionDirection.DECREASES
            if item.contribution < 0
            else ContributionDirection.NO_EFFECT
        )


def test_displayed_values_are_the_exact_model_inputs(fixture: dict[str, Any]) -> None:
    model = fixture["model"]
    features = fixture["splits"].validation.X.iloc[[4]]
    transformed = model.preprocessor.transform(features.loc[:, list(model.feature_columns)])
    transformed = np.asarray(transformed, dtype="float64")

    explanation = explain_transaction(model, features, top_k=10)

    for item in explanation.contributions:
        column = _feature_index(item.feature)
        assert item.raw_value == pytest.approx(float(features.iloc[0, column]))
        assert item.model_value == pytest.approx(float(transformed[0, column]))
        assert item.raw_value == pytest.approx(item.model_value)


def test_scaled_bundle_shows_raw_and_model_values_separately(fixture: dict[str, Any]) -> None:
    model = fixture["scaled_model"]
    features = fixture["splits"].validation.X.iloc[[2]]
    transformed = np.asarray(
        model.preprocessor.transform(features.loc[:, list(model.feature_columns)]),
        dtype="float64",
    )

    explanation = explain_transaction(model, features, top_k=10)

    differences = [abs(item.raw_value - item.model_value) for item in explanation.contributions]
    assert max(differences) > 1e-6
    for item in explanation.contributions:
        column = _feature_index(item.feature)
        assert item.raw_value == pytest.approx(float(features.iloc[0, column]))
        assert item.model_value == pytest.approx(float(transformed[0, column]))


def test_batch_explanations_return_one_result_per_row(fixture: dict[str, Any]) -> None:
    model = fixture["model"]
    features = fixture["splits"].validation.X.iloc[:4]

    explanations = explain_transactions(model, features, top_k=3)

    assert len(explanations) == len(features)
    for row, explanation in enumerate(explanations):
        assert explanation.top_k == 3
        assert explanation.model_path == str(model.path)
        assert explanation.model_sha256 == model.sha256
        assert np.isfinite(explanation.output_value)
        assert np.isfinite(explanation.additivity_residual)
        assert row >= 0


def test_explanations_are_reproducible(fixture: dict[str, Any]) -> None:
    model = fixture["model"]
    features = fixture["splits"].validation.X.iloc[[7]]

    first = explain_transaction(model, features, top_k=5).to_dict()
    second = explain_transaction(model, features, top_k=5).to_dict()

    assert first == second


# ---------------------------------------------------------------------------
# Global importance and reports
# ---------------------------------------------------------------------------
def test_global_importance_is_sorted_normalised_and_row_free(fixture: dict[str, Any]) -> None:
    model = fixture["model"]
    features = fixture["splits"].validation.X

    explanation = global_feature_importance(model, features, max_rows=25)
    record = explanation.to_dict()

    assert explanation.rows_explained <= 25
    assert explanation.rows_available == len(features)
    assert len(explanation.importance) == len(MODEL_FEATURE_COLUMNS)
    assert [item.rank for item in explanation.importance] == list(
        range(1, len(MODEL_FEATURE_COLUMNS) + 1)
    )
    assert all(np.isfinite(item.mean_abs_contribution) for item in explanation.importance)
    assert all(item.mean_abs_contribution >= 0 for item in explanation.importance)
    assert explanation.importance[0].mean_abs_contribution >= (
        explanation.importance[-1].mean_abs_contribution
    )
    assert sum(item.share_of_total_abs for item in explanation.importance) == pytest.approx(1.0)
    assert "rows_available" in record["explanation"]
    assert "raw_value" not in json.dumps(record)


def test_global_importance_uses_all_rows_when_the_cap_is_large(fixture: dict[str, Any]) -> None:
    model = fixture["model"]
    features = fixture["splits"].validation.X.iloc[:5]

    explanation = global_feature_importance(model, features, max_rows=DEFAULT_MAX_ROWS)

    assert explanation.rows_explained == len(features)
    assert explanation.sampling == "all supplied rows"


def test_report_writer_creates_strict_json_csv_and_markdown(
    tmp_path: Path,
    fixture: dict[str, Any],
) -> None:
    model = fixture["model"]
    features = fixture["splits"].validation.X.iloc[[1]]
    global_explanation = global_feature_importance(model, features, max_rows=10)
    transaction = explain_transaction(model, features, top_k=4)

    report = write_explanation_report(global_explanation, transaction, output_dir=tmp_path)

    for path in report.written_paths():
        assert path.is_file()
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["global"]["explanation"]["rows_explained"] == 1
    assert summary["transaction"]["explanation"]["top_k"] == 4
    with (tmp_path / "global_importance.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(MODEL_FEATURE_COLUMNS)
    assert set(rows[0]) == {
        "rank",
        "feature",
        "mean_abs_contribution",
        "mean_contribution",
        "share_of_total_abs",
    }
    markdown = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert markdown.startswith("# SHAP explanation report")
    assert "not probability points" in markdown
    assert "not a data export" in markdown


def test_individual_report_values_are_limited_to_top_k(
    tmp_path: Path,
    fixture: dict[str, Any],
) -> None:
    model = fixture["model"]
    features = fixture["splits"].validation.X.iloc[[3]]
    transaction = explain_transaction(model, features, top_k=3)

    write_explanation_report(
        global_feature_importance(model, features, max_rows=1),
        transaction,
        output_dir=tmp_path,
    )

    individual = json.loads((tmp_path / "transaction_explanation.json").read_text(encoding="utf-8"))
    assert len(individual["explanation"]["contributions"]) == 3
    assert "raw_value" not in json.dumps(individual["model"])


def test_markdown_states_the_output_space_and_limitations(fixture: dict[str, Any]) -> None:
    model = fixture["model"]
    features = fixture["splits"].validation.X.iloc[[0]]
    global_explanation = global_feature_importance(model, features, max_rows=5)
    transaction = explain_transaction(model, features, top_k=3)

    text = render_markdown(global_explanation, transaction)

    assert OUTPUT_SPACE in text
    assert "pre_calibration_tree_margin" in text
    assert "not causal effects" in text
    assert "PCA components" in text
    assert "threshold" in text.lower()


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def test_missing_or_non_finite_features_are_rejected(fixture: dict[str, Any]) -> None:
    model = fixture["model"]
    features = fixture["splits"].validation.X.iloc[[0]]

    with pytest.raises(ExplanationError, match="schema"):
        explain_transaction(model, features.drop(columns=["V1"]))
    non_finite = features.copy()
    non_finite.loc[non_finite.index[0], "V2"] = np.inf
    with pytest.raises(ExplanationError, match="non-finite"):
        explain_transaction(model, non_finite)
    with pytest.raises(ExplanationError, match="exactly one row"):
        explain_transaction(model, fixture["splits"].validation.X.iloc[[0, 1]])


def test_extra_input_columns_are_ignored(fixture: dict[str, Any]) -> None:
    model = fixture["model"]
    features = fixture["splits"].validation.X.iloc[[0]].assign(Time=123.0, Class=0)

    explanation = explain_transaction(model, features, top_k=2)

    assert all(item.feature in MODEL_FEATURE_COLUMNS for item in explanation.contributions)


def test_invalid_top_k_and_max_rows_are_rejected(fixture: dict[str, Any]) -> None:
    model = fixture["model"]
    features = fixture["splits"].validation.X.iloc[[0]]

    with pytest.raises(ExplanationError, match="top_k"):
        explain_transaction(model, features, top_k=0)
    with pytest.raises(ExplanationError, match="top_k"):
        explain_transaction(model, features, top_k=True)
    with pytest.raises(ExplanationError, match="max_rows"):
        global_feature_importance(model, features, max_rows=0)


# ---------------------------------------------------------------------------
# End-to-end rendering script
# ---------------------------------------------------------------------------
def _run_script(*arguments: str) -> subprocess.CompletedProcess[str]:
    if importlib.util.find_spec("matplotlib") is None:
        pytest.skip("the explainability script needs matplotlib, a dev-extra dependency")
    environment = {**os.environ, "MPLBACKEND": "Agg"}
    source = str(REPO_ROOT / "src")
    environment["PYTHONPATH"] = os.pathsep.join([source, environment.get("PYTHONPATH", "")]).rstrip(
        os.pathsep
    )
    return subprocess.run(  # noqa: S603 - fixed interpreter and repo-local script
        [sys.executable, str(SCRIPT), *arguments],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=environment,
        timeout=300,
        check=False,
    )


@pytest.mark.slow
def test_script_writes_plots_and_reports(
    tmp_path: Path,
    fixture: dict[str, Any],
) -> None:
    result = _run_script(
        "--path",
        str(fixture["dataset_path"]),
        "--model",
        str(fixture["bundle_path"]),
        "--output-dir",
        str(tmp_path),
        "--max-rows",
        "20",
        "--row",
        "0",
        "--top-k",
        "5",
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "summary.md").is_file()
    assert (tmp_path / "global_importance.csv").is_file()
    assert (tmp_path / "transaction_explanation.json").is_file()
    assert (tmp_path / "provenance.json").is_file()
    figures = sorted((tmp_path / "figures").glob("*.png"))
    assert len(figures) == 2
    for figure in figures:
        assert figure.stat().st_size > 5_000
    provenance = json.loads((tmp_path / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["transaction_row_index"] == 0
    assert provenance["global_max_rows"] == 20
    assert "not a dataset export" in json.dumps(provenance).lower()
    # The script's console output must not echo feature values from the explained row.
    assert "raw_value" not in result.stdout


@pytest.mark.slow
def test_script_refuses_a_missing_model_or_dataset(
    tmp_path: Path,
    fixture: dict[str, Any],
) -> None:
    missing_model = _run_script(
        "--path",
        str(fixture["dataset_path"]),
        "--model",
        str(tmp_path / "absent.joblib"),
        "--output-dir",
        str(tmp_path / "out"),
    )
    assert missing_model.returncode == 2
    assert "no model bundle" in missing_model.stderr

    missing_data = _run_script(
        "--path",
        str(tmp_path / "absent.csv"),
        "--model",
        str(fixture["bundle_path"]),
        "--output-dir",
        str(tmp_path / "out"),
    )
    assert missing_data.returncode == 2
    assert "no dataset" in missing_data.stderr
    assert not (tmp_path / "out").exists()
