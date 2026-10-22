"""Tests for the frozen final-evaluation stage.

Synthetic frames and a small deterministic pipeline only. The suite pins the guarantees the
stage exists for: threshold selection sees validation only, the model and decision are frozen
before test, test is scored once, probabilities and decisions are reported separately, and the
evaluation configuration is reproducible from the artifacts.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

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
from fraud_risk.models import final_evaluation
from fraud_risk.models.evaluation import (
    CostPolicy,
    MinRecallPolicy,
    select_cost_threshold,
)
from fraud_risk.models.final_evaluation import (
    DECISION_FILENAME,
    FROZEN_MODEL_FILENAME,
    METRICS_FILENAME,
    RUN_FILENAME,
    FinalEvaluationError,
    evaluate_final,
    load_bundle,
    main,
)
from fraud_risk.models.splitting import (
    EXIT_INVALID,
    EXIT_MISSING,
    EXIT_OK,
    DatasetSplits,
    SplitConfig,
    SplitStrategy,
    split_dataset,
)


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


def make_bundle(
    tmp_path: Path,
    splits: DatasetSplits,
    *,
    name: str = "model.joblib",
) -> tuple[Path, dict[str, Any]]:
    """A small calibrated pipeline in the comparison bundle's on-disk shape."""
    pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(ScalingStrategy.STANDARD)),
            (
                "classifier",
                CalibratedClassifierCV(
                    LogisticRegression(max_iter=500, class_weight="balanced", random_state=3),
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
            "family": "logistic_regression",
            "params": {"C": 1.0, "class_weight": "balanced", "solver": "lbfgs"},
            "imbalance": {"method": "class_weight", "value": "balanced"},
            "scaling": "standard",
            "calibration": {"method": "sigmoid", "folds": 3},
        },
        "pipeline": pipeline,
    }
    path = tmp_path / name
    joblib.dump(bundle, path)
    return path, bundle


def write_run_record(bundle_path: Path, splits: DatasetSplits) -> None:
    (bundle_path.parent / RUN_FILENAME).write_text(
        json.dumps({"split": splits.describe()}, indent=2) + "\n",
        encoding="utf-8",
    )


def load_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


class CountingPipeline:
    """A pipeline proxy that records how many times the model is asked to score."""

    def __init__(self, pipeline: Any) -> None:
        self.pipeline = pipeline
        self.calls: list[int] = []

    def predict_proba(self, features: pd.DataFrame) -> Any:
        self.calls.append(len(features))
        return self.pipeline.predict_proba(features)


# ---------------------------------------------------------------------------
# Artifacts and report contents
# ---------------------------------------------------------------------------
def test_final_evaluation_freezes_before_test_and_reports_both_outputs(
    tmp_path: Path,
) -> None:
    splits = make_splits()
    bundle_path, _ = make_bundle(tmp_path, splits)
    model = load_bundle(bundle_path)
    output_dir = tmp_path / "final"

    result = evaluate_final(
        splits,
        model,
        MinRecallPolicy(min_recall=0.8),
        output_dir=output_dir,
        dataset_path=tmp_path / "creditcard.csv",
    )

    for filename in (DECISION_FILENAME, FROZEN_MODEL_FILENAME, METRICS_FILENAME, RUN_FILENAME):
        assert (output_dir / filename).is_file()

    decision = load_json(output_dir / DECISION_FILENAME)
    assert decision["status"] == "frozen before test evaluation"
    assert decision["selected_on"] == "validation"
    assert decision["policy"]["kind"] == "min_recall"
    assert decision["threshold"] == result.decision.threshold
    assert decision["model"]["sha256"] == model.sha256
    assert decision["model"]["frozen_copy_sha256"] == model.sha256
    assert decision["metrics"]["alerts"] == result.decision.alerts
    assert isinstance(decision["selection_split"]["test"], str)
    assert "test" not in decision

    frozen_path = output_dir / FROZEN_MODEL_FILENAME
    assert frozen_path.read_bytes() == bundle_path.read_bytes()

    metrics = load_json(output_dir / METRICS_FILENAME)
    for split_name, comparison in (
        ("validation", result.validation),
        ("test", result.test),
    ):
        record = metrics[split_name]
        model_record = record["model"]
        assert set(model_record) >= {
            "precision",
            "recall",
            "f1",
            "pr_auc",
            "roc_auc",
            "false_positive_rate",
            "confusion_matrix",
            "alerts",
            "alert_rate",
            "per_class",
        }
        assert record["probability_summary"]["count"] == comparison.model.rows
        assert record["decision_summary"]["threshold"] == result.decision.threshold
        assert record["decision_summary"]["alerts"] == comparison.model.alerts
        assert model_record["alerts"] == (
            model_record["confusion_matrix"]["true_positive"]
            + model_record["confusion_matrix"]["false_positive"]
        )
        assert model_record["alert_rate"] == pytest.approx(
            model_record["alerts"] / comparison.model.rows,
            abs=1e-8,
        )
        assert (
            record["model_cost_impact"]["false_negatives"]["count"]
            == (model_record["confusion_matrix"]["false_negative"])
        )
        assert (
            record["model_cost_impact"]["false_positives"]["count"]
            == (model_record["confusion_matrix"]["false_positive"])
        )

    assert result.validation.model.recall >= 0.8
    assert result.test.model.rows == len(splits.test.y)
    assert metrics["test"]["model"]["alerts"] == result.test.model.alerts

    run = load_json(output_dir / RUN_FILENAME)
    assert run["test_evaluation"]["model_scorings_in_this_run"] == 1
    assert run["test_evaluation"]["performed_after_decision_freeze"] is True
    assert isinstance(run["split"]["splits"]["test"], dict)
    assert run["selection"]["policy"]["kind"] == "min_recall"
    assert run["model"]["sha256"] == model.sha256
    assert set(run["artifacts"]) == {"decision", "frozen_model", "metrics", "run"}
    assert "production-ready" in run["notes"]["status"]


def test_corrupting_test_does_not_change_the_frozen_decision(tmp_path: Path) -> None:
    splits = make_splits()
    bundle_path, _ = make_bundle(tmp_path, splits)
    model = load_bundle(bundle_path)
    policy = MinRecallPolicy(min_recall=0.8)

    baseline = evaluate_final(splits, model, policy, output_dir=tmp_path / "baseline")
    corrupted = replace(
        splits,
        test=replace(
            splits.test,
            X=splits.test.X + 1_000_000.0,
            y=1 - splits.test.y,
        ),
    )
    perturbed = evaluate_final(corrupted, model, policy, output_dir=tmp_path / "perturbed")

    assert baseline.decision.threshold == perturbed.decision.threshold
    assert baseline.decision.to_dict() == perturbed.decision.to_dict()
    assert baseline.validation.model.to_dict() == perturbed.validation.model.to_dict()
    assert baseline.test.model.to_dict() != perturbed.test.model.to_dict()

    baseline_decision = load_json(baseline.artifact_dir / DECISION_FILENAME)
    perturbed_decision = load_json(perturbed.artifact_dir / DECISION_FILENAME)
    assert baseline_decision["threshold"] == perturbed_decision["threshold"]
    assert baseline_decision["validation_comparison"] == perturbed_decision["validation_comparison"]
    assert (
        baseline_decision["selection_split"]["validation"]
        == perturbed_decision["selection_split"]["validation"]
    )


def test_test_split_is_scored_once_after_the_decision_file_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    splits = make_splits()
    bundle_path, _ = make_bundle(tmp_path, splits)
    model = load_bundle(bundle_path)
    counting = CountingPipeline(model.pipeline)
    counting_model = replace(model, pipeline=counting)
    output_dir = tmp_path / "final"
    contexts: list[str] = []
    real_predict = final_evaluation._predict_probabilities

    def wrapped(
        pipeline: Any,
        features: pd.DataFrame,
        *,
        context: str,
    ) -> NDArray[np.float64]:
        contexts.append(context)
        if context == "test":
            assert (output_dir / DECISION_FILENAME).is_file()
            assert not (output_dir / METRICS_FILENAME).exists()
        return real_predict(pipeline, features, context=context)

    monkeypatch.setattr(final_evaluation, "_predict_probabilities", wrapped)

    evaluate_final(
        splits,
        counting_model,
        MinRecallPolicy(min_recall=0.8),
        output_dir=output_dir,
    )

    assert contexts == ["validation", "test"]
    assert len(counting.calls) == 2
    assert (output_dir / METRICS_FILENAME).is_file()


def test_cost_policy_selects_the_same_threshold_as_the_cost_sweep(tmp_path: Path) -> None:
    splits = make_splits()
    bundle_path, _ = make_bundle(tmp_path, splits)
    model = load_bundle(bundle_path)
    validation_scores = model.pipeline.predict_proba(splits.validation.X)[:, 1]
    expected = select_cost_threshold(
        splits.validation.y,
        validation_scores,
        model.cost_policy,
    )

    result = evaluate_final(
        splits,
        model,
        model.cost_policy,
        output_dir=tmp_path / "final",
    )

    assert result.decision.threshold == expected
    assert result.policy.kind == "expected_cost"
    assert result.decision.metrics.to_dict()["alerts"] == result.decision.alerts


def test_bundle_and_split_contract_errors_are_loud(tmp_path: Path) -> None:
    with pytest.raises(FinalEvaluationError, match="not found"):
        load_bundle(tmp_path / "absent.joblib")

    invalid = tmp_path / "invalid.joblib"
    joblib.dump({"schema_version": 1}, invalid)
    with pytest.raises(FinalEvaluationError, match="missing keys"):
        load_bundle(invalid)

    splits = make_splits()
    bundle_path, _ = make_bundle(tmp_path, splits)
    record = joblib.load(bundle_path)
    record["threshold"] = True
    bad_threshold = tmp_path / "bad_threshold.joblib"
    joblib.dump(record, bad_threshold)
    with pytest.raises(FinalEvaluationError, match="threshold must be a number"):
        load_bundle(bad_threshold)

    model = load_bundle(bundle_path)
    subset = split_dataset(
        make_frame(),
        SplitConfig(strategy=SplitStrategy.STRATIFIED, seed=3),
        feature_columns=MODEL_FEATURE_COLUMNS[:5],
    )
    with pytest.raises(FinalEvaluationError, match="feature columns"):
        evaluate_final(
            subset,
            model,
            MinRecallPolicy(0.5),
            output_dir=tmp_path / "subset",
        )


def test_single_class_validation_fails_before_artifacts(tmp_path: Path) -> None:
    splits = make_splits()
    bundle_path, _ = make_bundle(tmp_path, splits)
    model = load_bundle(bundle_path)
    single_class = replace(
        splits,
        validation=replace(
            splits.validation,
            y=pd.Series(np.zeros(len(splits.validation.y), dtype="int64")),
        ),
    )
    output_dir = tmp_path / "final"

    with pytest.raises(FinalEvaluationError, match="only one class"):
        evaluate_final(
            single_class,
            model,
            MinRecallPolicy(0.5),
            output_dir=output_dir,
        )

    assert not output_dir.exists()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_cli_reads_run_json_and_writes_artifacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    frame = make_frame()
    frame_path = tmp_path / "creditcard.csv"
    frame.to_csv(frame_path, index=False)
    splits = make_splits(frame)
    bundle_path, _ = make_bundle(tmp_path, splits)
    write_run_record(bundle_path, splits)
    output_dir = tmp_path / "final"

    exit_code = main(
        [
            "--path",
            str(frame_path),
            "--model",
            str(bundle_path),
            "--output-dir",
            str(output_dir),
            "--policy",
            "min-recall",
            "--min-recall",
            "0.8",
        ]
    )

    assert exit_code == EXIT_OK
    printed = capsys.readouterr().out
    assert "Threshold" in printed
    assert "Validation (selection)" in printed
    assert "Test (scored once after freeze)" in printed
    assert "alerts" in printed
    assert "Artifacts" in printed
    for filename in (DECISION_FILENAME, FROZEN_MODEL_FILENAME, METRICS_FILENAME, RUN_FILENAME):
        assert (output_dir / filename).is_file()
    assert load_json(output_dir / DECISION_FILENAME)["policy"]["kind"] == "min_recall"


def test_cli_uses_explicit_split_when_run_json_is_absent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    frame = make_frame()
    frame_path = tmp_path / "creditcard.csv"
    frame.to_csv(frame_path, index=False)
    splits = make_splits(frame)
    bundle_path, _ = make_bundle(tmp_path, splits)
    output_dir = tmp_path / "final"

    missing_provenance = main(
        [
            "--path",
            str(frame_path),
            "--model",
            str(bundle_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    assert missing_provenance == EXIT_INVALID
    assert "no run.json" in capsys.readouterr().out

    exit_code = main(
        [
            "--path",
            str(frame_path),
            "--model",
            str(bundle_path),
            "--output-dir",
            str(output_dir),
            "--strategy",
            "stratified",
            "--seed",
            "3",
            "--policy",
            "alert-capacity",
            "--max-alerts",
            "5",
        ]
    )

    assert exit_code == EXIT_OK
    decision = load_json(output_dir / DECISION_FILENAME)
    assert decision["policy"]["kind"] == "alert_capacity"
    assert decision["diagnostics"]["capacity"] == 5


def test_cli_rejects_invalid_policy_arguments(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    frame = make_frame()
    frame_path = tmp_path / "creditcard.csv"
    frame.to_csv(frame_path, index=False)
    splits = make_splits(frame)
    bundle_path, _ = make_bundle(tmp_path, splits)
    output_dir = tmp_path / "final"

    base = [
        "--path",
        str(frame_path),
        "--model",
        str(bundle_path),
        "--output-dir",
        str(output_dir),
        "--strategy",
        "stratified",
        "--seed",
        "3",
    ]

    assert main([*base, "--policy", "min-recall"]) == EXIT_INVALID
    assert "requires --min-recall" in capsys.readouterr().out

    assert main([*base, "--policy", "cost", "--max-alerts", "10"]) == EXIT_INVALID
    assert "does not accept" in capsys.readouterr().out

    assert (
        main(
            [
                *base,
                "--policy",
                "alert-capacity",
                "--max-alerts",
                "10",
                "--max-alert-rate",
                "0.01",
            ]
        )
        == EXIT_INVALID
    )
    assert "exactly one" in capsys.readouterr().out
    assert not output_dir.exists()


def test_cli_reports_missing_dataset_or_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    frame_path = tmp_path / "creditcard.csv"
    make_frame().to_csv(frame_path, index=False)
    bundle_path, _ = make_bundle(tmp_path, make_splits())

    assert main(["--path", str(tmp_path / "absent.csv"), "--model", str(bundle_path)]) == (
        EXIT_MISSING
    )
    assert "Dataset not found" in capsys.readouterr().out

    assert main(["--path", str(frame_path), "--model", str(tmp_path / "absent.joblib")]) == (
        EXIT_MISSING
    )
    assert "Model bundle not found" in capsys.readouterr().out
