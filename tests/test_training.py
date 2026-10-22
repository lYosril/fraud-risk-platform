"""Tests for the three-family training comparison.

Synthetic frames only: no test reads the real dataset. The suite covers the protocol the
stage must guarantee - all three families fitted on training rows, selection on validation
expected cost, a frozen threshold scored once on test, per-class metrics and cost impact, and
a corrupted test split proving it cannot influence fitting or selection.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
import pytest

from fraud_risk.data.dataset import (
    AMOUNT_COLUMN,
    EXPECTED_COLUMNS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    TIME_COLUMN,
)
from fraud_risk.features.engineering import MODEL_FEATURE_COLUMNS
from fraud_risk.models.evaluation import CostPolicy, classification_metrics
from fraud_risk.models.splitting import (
    EXIT_INVALID,
    EXIT_MISSING,
    EXIT_OK,
    DatasetSplits,
    SplitConfig,
    SplitStrategy,
    split_dataset,
)
from fraud_risk.models.training import (
    COMPARISON_FILENAME,
    METRICS_FILENAME,
    MODEL_FILENAME,
    RUN_FILENAME,
    CandidateScore,
    ModelFamily,
    TrainingConfig,
    TrainingError,
    main,
    train_and_compare,
)
from fraud_risk.tracking import REGISTRY_NOTE, TrackingConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_FIXTURE = REPO_ROOT / "data" / "sample" / "creditcard.synthetic.csv"


def make_frame(rows: int = 300, positives: int = 30, seed: int = 7) -> pd.DataFrame:
    """A deterministic frame with signal, an imbalance and the dataset schema."""
    rng = np.random.default_rng(seed)
    labels = np.zeros(rows, dtype="int64")
    labels[rng.choice(rows, positives, replace=False)] = 1

    data: dict[str, object] = {TIME_COLUMN: np.arange(rows, dtype="float64") * 60.0}
    for column in FEATURE_COLUMNS:
        data[column] = rng.normal(size=rows) + 0.6 * labels
    data[AMOUNT_COLUMN] = np.round(rng.lognormal(mean=4.0, sigma=1.5, size=rows), 2)
    data[TARGET_COLUMN] = labels
    return pd.DataFrame(data)[list(EXPECTED_COLUMNS)]


def make_splits(frame: pd.DataFrame | None = None) -> DatasetSplits:
    return split_dataset(
        make_frame() if frame is None else frame,
        SplitConfig(strategy=SplitStrategy.STRATIFIED, seed=3),
    )


def small_config(
    families: tuple[ModelFamily, ...] = (
        ModelFamily.LOGISTIC_REGRESSION,
        ModelFamily.RANDOM_FOREST,
        ModelFamily.XGBOOST,
    ),
) -> TrainingConfig:
    """The full protocol with fixture-sized tree budgets, so tests stay fast."""
    return TrainingConfig(
        seed=5,
        cost_policy=CostPolicy(false_negative_cost=100.0, false_positive_cost=1.0),
        families=families,
        c_values=(0.1, 1.0),
        calibration_folds=3,
        max_iter=500,
        rf_n_estimators=20,
        rf_max_depth=6,
        rf_min_samples_leaf=2,
        xgb_n_estimators=30,
        xgb_max_depth=4,
        n_jobs=1,
    )


def load_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# The comparison and its artifacts
# ---------------------------------------------------------------------------
def test_comparison_trains_all_three_families(tmp_path: Path) -> None:
    run = train_and_compare(make_splits(), small_config(), output_dir=tmp_path / "comparison")

    assert {score.candidate.family for score in run.candidates} == {
        ModelFamily.LOGISTIC_REGRESSION,
        ModelFamily.RANDOM_FOREST,
        ModelFamily.XGBOOST,
    }
    assert len(run.candidates) == 8  # 4 logistic + 2 forest + 2 xgboost

    comparison = load_json(run.artifact_dir / COMPARISON_FILENAME)
    candidates = comparison["candidates"]
    families = {candidate["family"] for candidate in candidates}
    assert families == {"logistic_regression", "random_forest", "xgboost"}
    assert Counter(candidate["family"] for candidate in candidates) == {
        "logistic_regression": 4,
        "random_forest": 2,
        "xgboost": 2,
    }
    candidate_keys = {
        "name",
        "family",
        "params",
        "imbalance",
        "scaling",
        "fit_seconds",
        "converged",
        "validation",
    }
    validation_keys = {
        "average_precision",
        "threshold",
        "expected_cost_units",
        "metrics",
        "cost_impact",
    }
    for candidate in candidates:
        assert candidate_keys <= set(candidate)
        assert validation_keys <= set(candidate["validation"])
    assert comparison["selection"]["chosen"]["name"] == run.selected.candidate.name


def test_selection_minimises_validation_expected_cost_not_accuracy(tmp_path: Path) -> None:
    run = train_and_compare(make_splits(), small_config(), output_dir=tmp_path / "comparison")

    comparison = load_json(run.artifact_dir / COMPARISON_FILENAME)
    candidates = comparison["candidates"]
    costs = {
        candidate["name"]: candidate["validation"]["expected_cost_units"]
        for candidate in candidates
    }
    chosen = comparison["selection"]["chosen"]["name"]

    assert costs[chosen] == min(costs.values())
    assert comparison["selection"]["accuracy_used"] is False
    assert "expected cost" in comparison["selection"]["criterion"]
    for candidate in candidates:
        assert "accuracy" not in candidate["validation"]["metrics"]


def test_artifacts_are_self_contained_and_reloadable(tmp_path: Path) -> None:
    splits = make_splits()
    run = train_and_compare(splits, small_config(), output_dir=tmp_path / "comparison")

    for filename in (MODEL_FILENAME, COMPARISON_FILENAME, METRICS_FILENAME, RUN_FILENAME):
        assert (run.artifact_dir / filename).is_file()

    bundle = joblib.load(run.artifact_dir / MODEL_FILENAME)
    assert bundle["schema_version"] == 2
    assert bundle["feature_columns"] == list(MODEL_FEATURE_COLUMNS)
    assert bundle["threshold"] == run.threshold
    assert bundle["model"]["family"] == run.selected.candidate.family.value
    assert bundle["model"]["params"] == run.selected.candidate.params
    assert bundle["model"]["imbalance"] == run.selected.candidate.imbalance
    assert bundle["cost_policy"]["false_negative_cost"] == 100.0

    probabilities = bundle["pipeline"].predict_proba(splits.test.X.iloc[:5])
    assert probabilities.shape == (5, 2)
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_metrics_record_has_every_required_number_and_per_class_metrics(
    tmp_path: Path,
) -> None:
    run = train_and_compare(make_splits(), small_config(), output_dir=tmp_path / "comparison")
    record = load_json(run.artifact_dir / METRICS_FILENAME)

    assert record["threshold"]["value"] == run.threshold
    required = {"precision", "recall", "f1", "pr_auc", "roc_auc", "false_positive_rate"}
    for split in ("validation", "test"):
        for group in ("model", "majority_class"):
            metrics = record[split][group]
            assert required <= set(metrics)
            assert set(metrics["confusion_matrix"]) == {
                "true_negative",
                "false_positive",
                "false_negative",
                "true_positive",
            }
            assert set(metrics["per_class"]) == {"legitimate", "fraud"}
            assert {"precision", "recall", "f1", "support"} <= set(metrics["per_class"]["fraud"])
        model = record[split]["model"]
        assert model["per_class"]["fraud"]["precision"] == model["precision"]
        assert model["per_class"]["fraud"]["recall"] == model["recall"]
        assert model["per_class"]["legitimate"]["recall"] == pytest.approx(
            1.0 - model["false_positive_rate"], abs=1e-6
        )


def test_cost_impact_matches_the_confusion_matrix(tmp_path: Path) -> None:
    run = train_and_compare(make_splits(), small_config(), output_dir=tmp_path / "comparison")
    record = load_json(run.artifact_dir / METRICS_FILENAME)
    policy = record["cost_policy"]

    for split in ("validation", "test"):
        model = record[split]["model"]
        impact = record[split]["model_cost_impact"]
        matrix = model["confusion_matrix"]
        assert impact["false_negatives"]["count"] == matrix["false_negative"]
        assert impact["false_positives"]["count"] == matrix["false_positive"]
        expected_total = (
            matrix["false_negative"] * policy["false_negative_cost"]
            + matrix["false_positive"] * policy["false_positive_cost"]
        )
        assert impact["total_cost_units"] == pytest.approx(expected_total, abs=0.011)
        assert impact["false_negatives"]["meaning"]
        assert impact["false_positives"]["meaning"]


def test_run_record_captures_provenance_and_computational_limits(tmp_path: Path) -> None:
    splits = make_splits()
    run = train_and_compare(splits, small_config(), output_dir=tmp_path / "comparison")
    record = load_json(run.artifact_dir / RUN_FILENAME)

    assert record["seed"] == 5
    assert record["dataset"] == {"rows": 300, "features": 29}
    assert list(record["feature_schema"]) == list(MODEL_FEATURE_COLUMNS)
    assert set(record["feature_schema"].values()) == {"float64"}
    assert record["split"]["strategy"] == "stratified"
    assert record["split"]["seed"] == 3

    grid = record["grid"]
    assert set(grid["families"]) == {
        "logistic_regression",
        "random_forest",
        "xgboost",
    }
    assert grid["random_forest"]["n_estimators"] == 20
    assert grid["xgboost"]["n_estimators"] == 30
    assert grid["calibration_folds"] == 3
    assert grid["n_jobs"] == 1

    positives = int(splits.train.y.sum())
    negatives = len(splits.train.y) - positives
    assert record["imbalance"]["xgboost_scale_pos_weight"] == pytest.approx(
        negatives / positives, abs=1e-6
    )
    assert record["imbalance"]["resampling"].startswith("none")

    computational = record["computational"]
    assert computational["candidate_count"] == 8
    assert computational["total_fit_seconds"] >= 0
    assert computational["max_fit_seconds"] >= 0
    assert "bounded grids" in computational["limits"]

    assert record["library_versions"]["xgboost"]
    assert record["git_revision"] is None or isinstance(record["git_revision"], str)
    assert datetime.fromisoformat(record["created_at"]).tzinfo is not None


def test_xgboost_reweighting_is_derived_from_training_rows_only(tmp_path: Path) -> None:
    splits = make_splits()
    run = train_and_compare(splits, small_config(), output_dir=tmp_path / "comparison")

    xgb_candidates = [
        score for score in run.candidates if score.candidate.family is ModelFamily.XGBOOST
    ]
    positives = int(splits.train.y.sum())
    negatives = len(splits.train.y) - positives
    ratio = negatives / positives

    assert len(xgb_candidates) == 2
    values = [score.candidate.imbalance["value"] for score in xgb_candidates]
    assert any(value == 1.0 for value in values)
    assert any(value == pytest.approx(ratio, abs=1e-6) for value in values)
    assert all(
        score.candidate.imbalance["method"] == "scale_pos_weight" for score in xgb_candidates
    )


def test_frozen_threshold_is_the_one_applied_to_test(tmp_path: Path) -> None:
    splits = make_splits()
    run = train_and_compare(splits, small_config(), output_dir=tmp_path / "comparison")

    scores = run.model.predict_proba(splits.test.X)[:, 1]
    expected = classification_metrics(splits.test.y, scores, threshold=run.threshold)

    assert run.test.model.to_dict() == expected.to_dict()


# ---------------------------------------------------------------------------
# Test-set isolation and reproducibility
# ---------------------------------------------------------------------------
def test_test_split_never_influences_selection_or_fitting(tmp_path: Path) -> None:
    """Replace the test features and labels; the fitted models must not move."""
    splits = make_splits()
    corrupted_test = replace(
        splits.test,
        X=splits.test.X + 1_000_000.0,
        y=1 - splits.test.y,
    )
    corrupted = replace(splits, test=corrupted_test)

    baseline = train_and_compare(splits, small_config(), output_dir=tmp_path / "baseline")
    perturbed = train_and_compare(corrupted, small_config(), output_dir=tmp_path / "perturbed")

    assert baseline.selected.candidate.name == perturbed.selected.candidate.name
    assert baseline.threshold == perturbed.threshold
    assert baseline.selected.expected_cost == perturbed.selected.expected_cost
    np.testing.assert_allclose(
        baseline.model.predict_proba(splits.validation.X),
        perturbed.model.predict_proba(splits.validation.X),
    )
    assert baseline.validation.model.to_dict() == perturbed.validation.model.to_dict()
    assert baseline.test.model.to_dict() != perturbed.test.model.to_dict()


def test_the_same_configuration_is_reproducible(tmp_path: Path) -> None:
    splits = make_splits()

    first = train_and_compare(splits, small_config(), output_dir=tmp_path / "first")
    second = train_and_compare(splits, small_config(), output_dir=tmp_path / "second")

    def stable(score: CandidateScore) -> tuple[object, ...]:
        # fit_seconds is a measurement, not a result; every model output must match.
        return (
            score.candidate.to_dict(),
            score.converged,
            score.average_precision,
            score.threshold,
            score.expected_cost,
            score.metrics.to_dict(),
        )

    assert [stable(score) for score in first.candidates] == [
        stable(score) for score in second.candidates
    ]
    assert first.run_fingerprint == second.run_fingerprint
    assert first.selected.candidate.name == second.selected.candidate.name
    assert first.threshold == second.threshold
    assert first.validation.model.to_dict() == second.validation.model.to_dict()
    assert first.test.model.to_dict() == second.test.model.to_dict()


# ---------------------------------------------------------------------------
# Baseline, failure modes and configuration
# ---------------------------------------------------------------------------
def test_majority_class_baseline_is_always_reported(tmp_path: Path) -> None:
    run = train_and_compare(make_splits(), small_config(), output_dir=tmp_path / "comparison")

    majority = run.validation.majority_class

    assert majority.recall == 0.0
    assert majority.f1 == 0.0
    assert majority.accuracy > 0.8  # 10% fraud, so "never fraud" looks excellent
    assert majority.pr_auc == pytest.approx(0.1, abs=0.02)
    assert majority.roc_auc == pytest.approx(0.5)


def test_single_class_validation_split_is_rejected(tmp_path: Path) -> None:
    splits = make_splits()
    empty_validation = replace(
        splits.validation,
        y=pd.Series(np.zeros(len(splits.validation.y), dtype="int64")),
    )

    with pytest.raises(TrainingError, match="validation split contains only one class"):
        train_and_compare(
            replace(splits, validation=empty_validation),
            small_config(),
            output_dir=tmp_path / "comparison",
        )


def test_too_few_minority_rows_are_rejected(tmp_path: Path) -> None:
    splits = make_splits()
    minority = pd.Series(np.zeros(len(splits.train.y), dtype="int64"))
    minority.iloc[:2] = 1

    with pytest.raises(TrainingError, match="minority class"):
        train_and_compare(
            replace(splits, train=replace(splits.train, y=minority)),
            small_config(),
            output_dir=tmp_path / "comparison",
        )


def test_bad_configuration_values_raise() -> None:
    policy = CostPolicy(100.0, 1.0)

    with pytest.raises(TrainingError, match="seed"):
        TrainingConfig(seed=-1, cost_policy=policy)
    with pytest.raises(TrainingError, match="families"):
        TrainingConfig(seed=1, cost_policy=policy, families=())
    with pytest.raises(TrainingError, match="unique"):
        TrainingConfig(
            seed=1,
            cost_policy=policy,
            families=(ModelFamily.XGBOOST, ModelFamily.XGBOOST),
        )
    with pytest.raises(TrainingError, match="strictly positive"):
        TrainingConfig(seed=1, cost_policy=policy, c_values=(0.0,))
    with pytest.raises(TrainingError, match="calibration_folds"):
        TrainingConfig(seed=1, cost_policy=policy, calibration_folds=1)
    with pytest.raises(TrainingError, match="max_iter"):
        TrainingConfig(seed=1, cost_policy=policy, max_iter=0)
    with pytest.raises(TrainingError, match="tree counts"):
        TrainingConfig(seed=1, cost_policy=policy, rf_n_estimators=0)
    with pytest.raises(TrainingError, match="tree counts"):
        TrainingConfig(seed=1, cost_policy=policy, xgb_n_estimators=0)
    with pytest.raises(TrainingError, match="tree depths"):
        TrainingConfig(seed=1, cost_policy=policy, rf_max_depth=0)
    with pytest.raises(TrainingError, match="tree depths"):
        TrainingConfig(seed=1, cost_policy=policy, xgb_max_depth=0)
    with pytest.raises(TrainingError, match="rf_min_samples_leaf"):
        TrainingConfig(seed=1, cost_policy=policy, rf_min_samples_leaf=0)
    with pytest.raises(TrainingError, match="xgb_learning_rate"):
        TrainingConfig(seed=1, cost_policy=policy, xgb_learning_rate=0.0)
    with pytest.raises(TrainingError, match="xgb_learning_rate"):
        TrainingConfig(seed=1, cost_policy=policy, xgb_learning_rate=1.5)
    with pytest.raises(TrainingError, match="n_jobs"):
        TrainingConfig(seed=1, cost_policy=policy, n_jobs=0)
    with pytest.raises(TrainingError, match="n_jobs"):
        TrainingConfig(seed=1, cost_policy=policy, n_jobs=-2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_cli_trains_a_subset_and_reports(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "creditcard.csv"
    make_frame().to_csv(path, index=False)
    output_dir = tmp_path / "artifacts"

    exit_code = main(
        [
            "--path",
            str(path),
            "--strategy",
            "stratified",
            "--seed",
            "4",
            "--families",
            "logistic_regression",
            "--n-jobs",
            "1",
            "--output-dir",
            str(output_dir),
            "--no-mlflow",
        ]
    )

    assert exit_code == EXIT_OK
    printed = capsys.readouterr().out
    assert "Validation ranking" in printed
    assert "Selected" in printed
    assert "pr-auc" in printed
    assert "majority" in printed
    assert "Artifacts" in printed
    for filename in (MODEL_FILENAME, COMPARISON_FILENAME, METRICS_FILENAME, RUN_FILENAME):
        assert (output_dir / filename).is_file()

    comparison = load_json(output_dir / COMPARISON_FILENAME)
    assert comparison["selection"]["chosen"]["family"] == "logistic_regression"


def test_cli_reports_a_missing_dataset(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--path", str(tmp_path / "absent.csv")]) == EXIT_MISSING
    assert "Dataset not found" in capsys.readouterr().out


def test_cli_rejects_an_unknown_family(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "creditcard.csv"
    make_frame().to_csv(path, index=False)

    exit_code = main(
        [
            "--path",
            str(path),
            "--families",
            "neural_network",
            "--no-mlflow",
        ]
    )

    assert exit_code == EXIT_INVALID
    assert "unknown model family" in capsys.readouterr().out


def test_cli_fails_loudly_when_a_split_has_no_fraud(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_dir = tmp_path / "artifacts"

    exit_code = main(
        [
            "--path",
            str(SYNTHETIC_FIXTURE),
            "--strategy",
            "temporal",
            "--seed",
            "1",
            "--output-dir",
            str(output_dir),
            "--no-mlflow",
        ]
    )

    assert exit_code == EXIT_INVALID
    assert "FAILED" in capsys.readouterr().out
    assert not output_dir.exists()


# ---------------------------------------------------------------------------
# MLflow tracking integration
# ---------------------------------------------------------------------------
def _tracking_config(tmp_path: Path, name: str) -> TrackingConfig:
    """A private SQLite-backed tracking store with isolated artifact storage."""
    database = tmp_path / "tracking" / f"{name}.db"
    database.parent.mkdir(parents=True, exist_ok=True)
    return TrackingConfig(
        tracking_uri=f"sqlite:///{database.as_posix()}",
        experiment_name=name,
        artifact_location=(tmp_path / "mlartifacts" / name).as_uri(),
    )


def test_tracking_is_disabled_by_default_and_recorded_as_such(tmp_path: Path) -> None:
    run = train_and_compare(make_splits(), small_config(), output_dir=tmp_path / "comparison")

    assert run.tracking is None
    record = load_json(run.artifact_dir / RUN_FILENAME)
    assert record["schema_version"] == 3
    assert record["tracking"] == {"enabled": False, "registry": REGISTRY_NOTE}

    bundle = joblib.load(run.artifact_dir / MODEL_FILENAME)
    assert bundle["tracking"]["enabled"] is False
    assert bundle["run_fingerprint"] == run.run_fingerprint


def test_tracking_records_a_real_run_with_metadata_and_artifacts(tmp_path: Path) -> None:
    from mlflow.tracking import MlflowClient

    splits = make_splits()
    tracking = _tracking_config(tmp_path, "fraud-detection-test")

    run = train_and_compare(
        splits,
        small_config(),
        output_dir=tmp_path / "comparison",
        tracking=tracking,
    )

    assert run.tracking is not None
    assert run.tracking.run_fingerprint == run.run_fingerprint
    assert run.tracking.registry == REGISTRY_NOTE

    client = MlflowClient(tracking_uri=tracking.tracking_uri)
    stored = client.get_run(run.tracking.run_id)

    assert stored.info.status == "FINISHED"
    assert stored.data.tags["run.fingerprint"] == run.run_fingerprint
    assert stored.data.tags["training.status"] == "completed"
    assert stored.data.tags["tracking.registry"] == REGISTRY_NOTE
    assert stored.data.tags["split.strategy"] == "stratified"
    assert stored.data.tags["split.seed"] == "3"

    assert stored.data.params["seed"] == "5"
    assert stored.data.params["split.strategy"] == "stratified"
    assert stored.data.params["split.seed"] == "3"
    assert stored.data.params["config.calibration_folds"] == "3"
    assert stored.data.params["feature.count"] == "29"
    assert stored.data.params["selected.model.family"] == run.selected.candidate.family.value

    assert stored.data.metrics["validation.pr_auc"] == pytest.approx(run.validation.model.pr_auc)
    assert stored.data.metrics["validation.expected_cost_units"] == pytest.approx(
        run.selected.expected_cost
    )
    assert stored.data.metrics["test.pr_auc"] == pytest.approx(run.test.model.pr_auc)

    artifact_paths = {item.path for item in client.list_artifacts(run.tracking.run_id, "artifacts")}
    assert artifact_paths == {
        f"artifacts/{MODEL_FILENAME}",
        f"artifacts/{COMPARISON_FILENAME}",
        f"artifacts/{METRICS_FILENAME}",
        f"artifacts/{RUN_FILENAME}",
    }

    record = load_json(run.artifact_dir / RUN_FILENAME)
    assert record["run_fingerprint"] == run.run_fingerprint
    assert record["tracking"]["run_id"] == run.tracking.run_id
    assert record["tracking"]["experiment_name"] == "fraud-detection-test"
    assert record["tracking"]["registry"] == REGISTRY_NOTE


def test_tracking_marks_a_failed_run_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mlflow.tracking import MlflowClient

    tracking = _tracking_config(tmp_path, "fraud-detection-failure")

    def explode(*args: object, **kwargs: object) -> Any:
        raise TrainingError("synthetic fitting failure")

    monkeypatch.setattr("fraud_risk.models.training._fit_candidate", explode)

    with pytest.raises(TrainingError, match="synthetic fitting failure"):
        train_and_compare(
            make_splits(),
            small_config(),
            output_dir=tmp_path / "comparison",
            tracking=tracking,
        )

    client = MlflowClient(tracking_uri=tracking.tracking_uri)
    experiment = client.get_experiment_by_name("fraud-detection-failure")
    assert experiment is not None
    runs = client.search_runs(
        [experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
    )

    assert len(runs) == 1
    failed = runs[0]
    assert failed.info.status == "FAILED"
    assert failed.data.tags["training.status"] == "failed"
    assert failed.data.tags["failure.type"] == "TrainingError"
    assert "synthetic fitting failure" in failed.data.tags["failure.message"]
