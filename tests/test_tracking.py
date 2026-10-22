"""Tests for the MLflow tracking boundary.

No test starts a tracking server: the configuration, URI redaction and the deterministic run
fingerprint are pure functions, and the training tests exercise a real local SQLite-backed
run. MLflow is imported lazily so this module can be collected without it.
"""

from __future__ import annotations

import pytest

from fraud_risk.config import Settings
from fraud_risk.tracking import (
    TrackingConfig,
    TrackingError,
    build_run_fingerprint,
    build_run_name,
    redact_uri,
)


def test_tracking_config_is_built_from_settings() -> None:
    settings = Settings(  # type: ignore[call-arg]  # `_env_file` is a runtime-only kwarg
        _env_file=None,
        mlflow_tracking_uri="http://user:pass@mlflow.internal:5000/path?token=secret",
        mlflow_experiment_name="fraud-staging",
    )

    config = TrackingConfig.from_settings(settings)

    assert config.tracking_uri == "http://user:pass@mlflow.internal:5000/path?token=secret"
    assert config.experiment_name == "fraud-staging"
    assert config.safe_tracking_uri == "http://mlflow.internal:5000/path"
    assert config.artifact_location is None


def test_tracking_config_rejects_blank_values() -> None:
    with pytest.raises(TrackingError, match="tracking_uri"):
        TrackingConfig(tracking_uri="   ", experiment_name="fraud-detection")
    with pytest.raises(TrackingError, match="experiment_name"):
        TrackingConfig(tracking_uri="sqlite:///mlflow.db", experiment_name="")
    with pytest.raises(TrackingError, match="artifact_location"):
        TrackingConfig(
            tracking_uri="sqlite:///mlflow.db",
            experiment_name="fraud-detection",
            artifact_location="  ",
        )


def test_uri_redaction_removes_credentials_and_query_but_keeps_the_locator() -> None:
    assert redact_uri("sqlite:///mlflow.db") == "sqlite:///mlflow.db"
    assert redact_uri("file:///C:/tmp/mlflow.db") == "file:///C:/tmp/mlflow.db"
    assert redact_uri("http://user:pass@mlflow.internal:5000/path?token=secret") == (
        "http://mlflow.internal:5000/path"
    )
    assert redact_uri("https://user@mlflow.internal/") == "https://mlflow.internal/"


def test_run_fingerprint_is_stable_and_sensitive_to_its_inputs() -> None:
    split = {"strategy": "stratified", "seed": 3, "row_count": 300}
    feature_columns = ["Time", "V1", "Amount"]
    config = {"seed": 5, "calibration_folds": 3}

    first = build_run_fingerprint(
        seed=5,
        split=split,
        feature_columns=feature_columns,
        config=config,
        git_revision="abc123",
    )
    second = build_run_fingerprint(
        seed=5,
        split=split,
        feature_columns=feature_columns,
        config=config,
        git_revision="abc123",
    )

    assert first == second
    assert len(first) == 64
    assert first != build_run_fingerprint(
        seed=6,
        split=split,
        feature_columns=feature_columns,
        config=config,
        git_revision="abc123",
    )
    assert first != build_run_fingerprint(
        seed=5,
        split={"strategy": "temporal", "seed": 3, "row_count": 300},
        feature_columns=feature_columns,
        config=config,
        git_revision="abc123",
    )
    assert first != build_run_fingerprint(
        seed=5,
        split=split,
        feature_columns=feature_columns,
        config=config,
        git_revision="def456",
    )


def test_run_name_is_deterministic_and_short() -> None:
    fingerprint = "a" * 64

    name = build_run_name("fraud-detection", fingerprint)

    assert name == "fraud-detection-aaaaaaaaaaaa"
    assert build_run_name("fraud-detection", fingerprint) == name
