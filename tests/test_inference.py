"""Tests for the reusable inference service and its API-facing schemas.

Synthetic bundles only: no test reads the real dataset or the real artifact store. The suite
covers the serving contract: one fitted pipeline is reused without retraining, input order is
enforced, missing/corrupt/incompatible artifacts fail clearly, and the response separates
probability, threshold and binary decision from the optional explanation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from fraud_risk.api.schemas import PredictionRequest, PredictionResponse
from fraud_risk.config import Settings
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
from fraud_risk.models.bundle import sha256_file
from fraud_risk.models.evaluation import CostPolicy
from fraud_risk.models.inference import (
    FraudInferenceService,
    InferenceError,
    InferenceInputError,
    InferenceModelError,
    InferenceRequest,
)
from fraud_risk.models.splitting import DatasetSplits, SplitConfig, SplitStrategy, split_dataset

FINGERPRINT = "unit-test-fingerprint"


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


def make_splits() -> DatasetSplits:
    return split_dataset(
        make_frame(),
        SplitConfig(strategy=SplitStrategy.STRATIFIED, seed=3),
    )


def make_pipeline(
    splits: DatasetSplits,
    *,
    tree: bool = False,
    feature_columns: tuple[str, ...] = MODEL_FEATURE_COLUMNS,
) -> Pipeline:
    """A small calibrated pipeline fitted on training rows only."""
    if tree:
        positives = int(splits.train.y.sum())
        negatives = len(splits.train.y) - positives
        estimator: Any = XGBClassifier(
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
        )
    else:
        estimator = LogisticRegression(max_iter=500, class_weight="balanced", random_state=3)

    pipeline = Pipeline(
        steps=[
            (
                "preprocess",
                build_preprocessor(
                    ScalingStrategy.STANDARD,
                    feature_columns=feature_columns,
                ),
            ),
            (
                "classifier",
                CalibratedClassifierCV(estimator, method="sigmoid", cv=3),
            ),
        ]
    )
    pipeline.fit(splits.train.X.loc[:, list(feature_columns)], splits.train.y)
    return pipeline


def make_bundle(
    path: Path,
    splits: DatasetSplits,
    *,
    tree: bool = False,
    feature_columns: tuple[str, ...] = MODEL_FEATURE_COLUMNS,
    run_fingerprint: str | None = FINGERPRINT,
) -> Path:
    """Write a pipeline in the comparison bundle's on-disk shape."""
    pipeline = make_pipeline(splits, tree=tree, feature_columns=feature_columns)
    bundle: dict[str, Any] = {
        "schema_version": 2,
        "created_at": "2026-01-01T00:00:00+00:00",
        "seed": splits.config.seed,
        "feature_columns": list(feature_columns),
        "threshold": 0.5,
        "cost_policy": CostPolicy(100.0, 1.0).to_dict(),
        "model": {
            "family": "xgboost" if tree else "logistic_regression",
            "params": {"test": True},
            "imbalance": {"method": "class_weight"},
            "scaling": ScalingStrategy.STANDARD.value,
            "calibration": {"method": "sigmoid", "folds": 3},
        },
        "pipeline": pipeline,
    }
    if run_fingerprint is not None:
        bundle["run_fingerprint"] = run_fingerprint
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def request_for(splits: DatasetSplits, row: int = 0) -> InferenceRequest:
    features = splits.validation.X.iloc[row]
    return InferenceRequest(
        transaction_id=f"transaction-{row}",
        features={column: float(features[column]) for column in FEATURE_COLUMNS},
        amount=float(features[AMOUNT_COLUMN]),
    )


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """One fitted logistic bundle and one fitted tree bundle shared by read-only tests."""
    root = tmp_path_factory.mktemp("inference")
    splits = make_splits()
    logistic_path = make_bundle(root / "logistic" / "model.joblib", splits)
    tree_path = make_bundle(root / "tree" / "model.joblib", splits, tree=True)
    return {
        "root": root,
        "splits": splits,
        "logistic_path": logistic_path,
        "tree_path": tree_path,
    }


# ---------------------------------------------------------------------------
# Valid input and artifact loading
# ---------------------------------------------------------------------------
def test_predicts_from_the_loaded_bundle_without_retraining(
    trained: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    splits: DatasetSplits = trained["splits"]
    service = FraudInferenceService(trained["logistic_path"])
    request = request_for(splits)

    expected = float(service.model.pipeline.predict_proba(splits.validation.X.iloc[[0]])[0, 1])
    loaded = service.model

    def refuse_fit(*args: object, **kwargs: object) -> None:
        raise AssertionError("inference must never fit or retrain the model")

    monkeypatch.setattr(Pipeline, "fit", refuse_fit)
    result = service.predict(request)

    assert service.model is loaded  # the pipeline is cached, not reloaded per request
    assert result.transaction_id == "transaction-0"
    assert result.probability == pytest.approx(expected)
    assert result.threshold == pytest.approx(0.5)
    assert result.prediction == int(expected >= 0.5)
    assert result.model_version == FINGERPRINT
    assert result.model_sha256 == sha256_file(trained["logistic_path"])
    assert result.model_family == "logistic_regression"
    assert result.explanation is None


def test_model_version_falls_back_to_the_content_hash(
    tmp_path: Path, trained: dict[str, Any]
) -> None:
    path = make_bundle(
        tmp_path / "no-fingerprint" / "model.joblib",
        trained["splits"],
        run_fingerprint=None,
    )

    service = FraudInferenceService(path)
    result = service.predict(request_for(trained["splits"]))

    assert result.model_version == sha256_file(path)


def test_artifact_path_can_come_from_settings(trained: dict[str, Any]) -> None:
    settings = Settings(  # type: ignore[call-arg]  # `_env_file` is runtime-only
        _env_file=None,
        model_artifact_path=trained["logistic_path"],
    )

    service = FraudInferenceService(settings=settings)

    assert service.artifact_path == trained["logistic_path"]


# ---------------------------------------------------------------------------
# Invalid input and stable ordering
# ---------------------------------------------------------------------------
def test_rejects_reordered_features(trained: dict[str, Any]) -> None:
    splits: DatasetSplits = trained["splits"]
    row = splits.validation.X.iloc[0]
    reversed_features = {column: float(row[column]) for column in reversed(FEATURE_COLUMNS)}

    with pytest.raises(InferenceInputError, match="feature order"):
        FraudInferenceService(trained["logistic_path"]).predict(
            InferenceRequest(
                transaction_id="reordered",
                features=reversed_features,
                amount=float(row[AMOUNT_COLUMN]),
            )
        )


def test_rejects_missing_extra_non_finite_and_negative_inputs(
    trained: dict[str, Any],
) -> None:
    splits: DatasetSplits = trained["splits"]
    service = FraudInferenceService(trained["logistic_path"])
    valid = request_for(splits)

    missing = dict(valid.features)
    del missing["V1"]
    with pytest.raises(InferenceInputError, match="missing: V1"):
        service.predict(InferenceRequest("missing", missing, amount=valid.amount))

    extra = {**valid.features, TIME_COLUMN: 1.0}
    with pytest.raises(InferenceInputError, match="unexpected: Time"):
        service.predict(InferenceRequest("extra", extra, amount=valid.amount))

    non_finite = {**valid.features, "V1": float("inf")}
    with pytest.raises(InferenceInputError, match="finite"):
        service.predict(InferenceRequest("infinite", non_finite, amount=valid.amount))

    with pytest.raises(InferenceInputError, match="non-negative"):
        service.predict(InferenceRequest("negative-amount", dict(valid.features), amount=-1.0))


def test_accepts_all_model_features_without_a_separate_amount(trained: dict[str, Any]) -> None:
    splits: DatasetSplits = trained["splits"]
    row = splits.validation.X.iloc[0]
    features = {column: float(row[column]) for column in MODEL_FEATURE_COLUMNS}

    result = FraudInferenceService(trained["logistic_path"]).predict(
        InferenceRequest(transaction_id="all-features", features=features)
    )

    assert 0.0 <= result.probability <= 1.0


# ---------------------------------------------------------------------------
# Missing, corrupt and incompatible artifacts
# ---------------------------------------------------------------------------
def test_missing_artifact_fails_with_a_model_error(tmp_path: Path) -> None:
    service = FraudInferenceService(tmp_path / "missing.joblib")

    with pytest.raises(InferenceModelError, match="model artifact unavailable"):
        service.predict(InferenceRequest("txn", {"V1": 0.0}))


def test_corrupt_artifact_fails_with_a_model_error(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.joblib"
    path.write_text("this is not a joblib artifact", encoding="utf-8")

    with pytest.raises(InferenceModelError, match="model artifact unavailable"):
        FraudInferenceService(path).predict(InferenceRequest("txn", {"V1": 0.0}))


def test_incompatible_preprocessing_order_fails_with_a_model_error(
    tmp_path: Path, trained: dict[str, Any]
) -> None:
    original = make_bundle(tmp_path / "original.joblib", trained["splits"])
    bundle = joblib.load(original)
    bundle["feature_columns"] = list(reversed(MODEL_FEATURE_COLUMNS))
    broken = tmp_path / "reordered.joblib"
    joblib.dump(bundle, broken)

    with pytest.raises(InferenceModelError, match="preprocessing are incompatible"):
        FraudInferenceService(broken).predict(request_for(trained["splits"]))


# ---------------------------------------------------------------------------
# Optional explanation and API schemas
# ---------------------------------------------------------------------------
def test_optional_explanation_is_separate_from_score_and_decision(
    trained: dict[str, Any],
) -> None:
    service = FraudInferenceService(trained["tree_path"])
    result = service.predict(request_for(trained["splits"]), explain=True, top_k=3)

    assert result.explanation is not None
    assert result.explanation.top_k == 3
    assert len(result.explanation.contributions) == 3
    assert result.explanation.output_space == "pre_calibration_tree_margin"
    assert all(item.feature in MODEL_FEATURE_COLUMNS for item in result.explanation.contributions)

    response = PredictionResponse.from_result(result)
    assert response.probability == result.probability
    assert response.threshold == result.threshold
    assert response.prediction == result.prediction
    assert response.decision == ("fraud" if result.prediction == 1 else "legitimate")
    assert response.model_version == FINGERPRINT
    assert response.explanation is not None
    assert len(response.explanation.contributions) == 3

    payload = response.model_dump()
    assert payload["explanation"]["output_space"] == "pre_calibration_tree_margin"
    assert "features" not in payload


def test_a_requested_explanation_for_an_unsupported_family_fails_loudly(
    trained: dict[str, Any],
) -> None:
    with pytest.raises(InferenceError, match="cannot produce a requested explanation"):
        FraudInferenceService(trained["logistic_path"]).predict(
            request_for(trained["splits"]),
            explain=True,
        )


def test_request_schema_rejects_labels_and_unknown_fields() -> None:
    features = dict.fromkeys(FEATURE_COLUMNS, 0.0)

    request = PredictionRequest(transaction_id="txn", amount=10.0, features=features)
    assert list(request.features) == list(FEATURE_COLUMNS)
    assert request.to_inference_request().amount == 10.0

    with pytest.raises(ValidationError, match="label field"):
        PredictionRequest(
            transaction_id="txn",
            amount=10.0,
            features={**features, "is_fraud": 0.0},
        )
    with pytest.raises(ValidationError, match="label field"):
        PredictionRequest(
            transaction_id="txn",
            amount=10.0,
            features={**features, TARGET_COLUMN: 0.0},
        )
    with pytest.raises(ValidationError, match="top-level amount"):
        PredictionRequest(
            transaction_id="txn",
            amount=10.0,
            features={**features, AMOUNT_COLUMN: 10.0},
        )
    with pytest.raises(ValidationError, match="finite"):
        PredictionRequest(
            transaction_id="txn",
            amount=10.0,
            features={**features, "V1": float("inf")},
        )
    with pytest.raises(ValidationError):
        PredictionRequest(
            transaction_id="txn",
            amount=-1.0,
            features=features,
        )
    with pytest.raises(ValidationError):
        PredictionRequest.model_validate(
            {
                "transaction_id": "txn",
                "amount": 10.0,
                "features": features,
                "is_fraud": 0,
            }
        )
    with pytest.raises(ValidationError, match="valid number"):
        PredictionRequest.model_validate(
            {
                "transaction_id": "txn",
                "amount": 10.0,
                "features": {**features, "V1": "0.1"},
            }
        )


def test_response_schema_rejects_an_out_of_range_probability() -> None:
    with pytest.raises(ValidationError):
        PredictionResponse(
            transaction_id="txn",
            probability=1.5,
            threshold=0.5,
            prediction=1,
            model_version=FINGERPRINT,
            model_sha256="a" * 64,
            model_family="logistic_regression",
        )


def test_response_serialises_without_a_feature_snapshot(trained: dict[str, Any]) -> None:
    result = FraudInferenceService(trained["logistic_path"]).predict(request_for(trained["splits"]))

    payload = json.loads(PredictionResponse.from_result(result).model_dump_json())

    assert set(payload) == {
        "transaction_id",
        "probability",
        "threshold",
        "prediction",
        "decision",
        "model_version",
        "model_sha256",
        "model_family",
        "explanation",
    }
    assert payload["explanation"] is None
