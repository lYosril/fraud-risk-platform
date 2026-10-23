"""Failure-boundary tests for the MLflow tracking session.

No MLflow client or server is started: a small fake client records calls and can be told to
fail. The tests pin validation, terminal-state handling and the rule that tracking failures
must be reported without masking the original training failure.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import cast

import pytest

from fraud_risk.tracking import (
    TrackingConfig,
    TrackingError,
    TrackingRecord,
    TrackingSession,
    start_tracking,
    track_experiment,
)


class FakeClient:
    def __init__(self) -> None:
        self.params: list[tuple[str, str, object]] = []
        self.metrics: list[tuple[str, str, float]] = []
        self.artifacts: list[tuple[str, str, str | None]] = []
        self.tags: list[tuple[str, str, str]] = []
        self.terminated: list[tuple[str, str]] = []
        self.fail_on: str | None = None

    def _check(self, operation: str) -> None:
        if self.fail_on == operation:
            raise RuntimeError(f"synthetic {operation} failure")

    def log_param(self, run_id: str, key: str, value: object) -> None:
        self._check("log_param")
        self.params.append((run_id, key, value))

    def log_metric(self, run_id: str, key: str, value: float) -> None:
        self._check("log_metric")
        self.metrics.append((run_id, key, value))

    def log_artifact(self, run_id: str, path: str, *, artifact_path: str | None = None) -> None:
        self._check("log_artifact")
        self.artifacts.append((run_id, path, artifact_path))

    def set_tag(self, run_id: str, key: str, value: str) -> None:
        self._check("set_tag")
        self.tags.append((run_id, key, value))

    def set_terminated(self, run_id: str, status: str) -> None:
        self._check("set_terminated")
        self.terminated.append((run_id, status))


def make_record() -> TrackingRecord:
    return TrackingRecord(
        run_id="run-1",
        experiment_id="experiment-1",
        experiment_name="fraud-detection",
        tracking_uri="sqlite:///mlflow.db",
        run_name="fraud-detection-aaaaaaaaaaaa",
        run_fingerprint="a" * 64,
        artifact_uri=None,
    )


def make_session(client: FakeClient | None = None) -> tuple[TrackingSession, FakeClient]:
    resolved = client or FakeClient()
    return TrackingSession(record=make_record(), _client=resolved), resolved


def test_log_params_validates_names_and_serialises_values() -> None:
    session, client = make_session()

    session.log_params({"seed": 1, "flag": True, "nested": {"a": 1}})

    assert client.params == [
        ("run-1", "seed", 1),
        ("run-1", "flag", "true"),
        ("run-1", "nested", '{"a":1}'),
    ]

    with pytest.raises(TrackingError, match="parameter names"):
        session.log_params({"": 1})


def test_log_params_wraps_client_failures() -> None:
    client = FakeClient()
    client.fail_on = "log_param"
    session, _ = make_session(client)

    with pytest.raises(TrackingError, match="could not log parameter"):
        session.log_params({"seed": 1})


def test_log_metrics_validates_numeric_finite_values_and_wraps_failures() -> None:
    session, client = make_session()

    session.log_metrics({"pr_auc": 0.75})

    assert client.metrics == [("run-1", "pr_auc", 0.75)]

    with pytest.raises(TrackingError, match="metric names"):
        session.log_metrics({"": 1.0})
    with pytest.raises(TrackingError, match="must be numeric"):
        session.log_metrics(cast(dict[str, float], {"bad": "not-a-number"}))
    with pytest.raises(TrackingError, match="must be finite"):
        session.log_metrics({"bad": math.nan})

    client.fail_on = "log_metric"
    with pytest.raises(TrackingError, match="could not log metric"):
        session.log_metrics({"pr_auc": 0.75})


def test_log_artifacts_requires_real_files_and_wraps_failures(tmp_path: Path) -> None:
    session, client = make_session()
    artifact = tmp_path / "metrics.json"
    artifact.write_text("{}", encoding="utf-8")

    session.log_artifacts([artifact], artifact_path="run")

    assert client.artifacts == [("run-1", str(artifact), "run")]

    with pytest.raises(TrackingError, match="does not exist"):
        session.log_artifacts([tmp_path / "missing.json"])

    client.fail_on = "log_artifact"
    with pytest.raises(TrackingError, match="could not log artifact"):
        session.log_artifacts([artifact])


def test_complete_marks_the_run_once_and_closes_the_session() -> None:
    session, client = make_session()

    session.complete()
    session.complete()

    assert client.tags == [("run-1", "training.status", "completed")]
    assert client.terminated == [("run-1", "FINISHED")]
    with pytest.raises(TrackingError, match="already finished"):
        session.log_params({"seed": 1})


def test_complete_reports_a_terminal_state_failure() -> None:
    client = FakeClient()
    client.fail_on = "set_terminated"
    session, _ = make_session(client)

    with pytest.raises(TrackingError, match="mark MLflow run"):
        session.complete()


def test_fail_is_best_effort_and_does_not_mask_the_original_error() -> None:
    session, client = make_session()
    error = RuntimeError("x" * 2000)

    session.fail(error)

    assert client.tags[0] == ("run-1", "training.status", "failed")
    assert client.tags[1] == ("run-1", "failure.type", "RuntimeError")
    assert client.tags[2][1] == "failure.message"
    assert len(client.tags[2][2]) == 1000
    assert client.terminated == [("run-1", "FAILED")]

    session.fail(RuntimeError("again"))
    assert len(client.terminated) == 1


def test_fail_swallows_tracking_failures() -> None:
    client = FakeClient()
    client.fail_on = "set_tag"
    session, _ = make_session(client)

    session.fail(RuntimeError("training failed"))

    assert client.terminated == [("run-1", "FAILED")]


def test_start_tracking_validates_names_before_importing_mlflow() -> None:
    with pytest.raises(TrackingError, match="run_name"):
        start_tracking(
            make_config(),
            run_name="   ",
            run_fingerprint="a" * 64,
        )
    with pytest.raises(TrackingError, match="run_fingerprint"):
        start_tracking(
            make_config(),
            run_name="run",
            run_fingerprint="",
        )


def test_track_experiment_completes_on_success_and_fails_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, client = make_session()
    monkeypatch.setattr("fraud_risk.tracking.start_tracking", lambda *args, **kwargs: session)

    with track_experiment(
        make_config(),
        run_name="run",
        run_fingerprint="a" * 64,
    ) as active:
        assert active is session

    assert client.terminated == [("run-1", "FINISHED")]

    second_session, second_client = make_session()
    monkeypatch.setattr(
        "fraud_risk.tracking.start_tracking",
        lambda *args, **kwargs: second_session,
    )

    with (
        pytest.raises(RuntimeError, match="training exploded"),
        track_experiment(
            make_config(),
            run_name="run",
            run_fingerprint="b" * 64,
        ),
    ):
        raise RuntimeError("training exploded")

    assert second_client.terminated == [("run-1", "FAILED")]


def make_config() -> TrackingConfig:
    return TrackingConfig(tracking_uri="sqlite:///mlflow.db", experiment_name="fraud-detection")
