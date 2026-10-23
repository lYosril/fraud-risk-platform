"""Unit tests for the monitoring command-line entry point.

The database engine, session factory, repository and report builder are replaced with tiny
fakes. These tests pin exit codes and failure containment; the real SQL path is covered by the
PostgreSQL integration tests.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

from fraud_risk.config import Settings
from fraud_risk.monitoring import cli
from fraud_risk.monitoring.drift import DriftStatus

NOW = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)


class FakeSession:
    def __enter__(self) -> FakeSession:
        return self

    def __exit__(self, *args: object) -> None:
        return None


class FakeEngine:
    def __init__(self) -> None:
        self.disposed = False

    def dispose(self) -> None:
        self.disposed = True


class FakeRepository:
    instances: ClassVar[list[FakeRepository]] = []

    def __init__(self, session: object) -> None:
        self.session = session
        self.saved: list[object] = []
        FakeRepository.instances.append(self)

    def window_aggregate(self, *, start: datetime, end: datetime) -> object:
        return {"start": start, "end": end}

    def analysis_sample(
        self,
        *,
        start: datetime,
        end: datetime,
        limit: int,
        feature_columns: list[str],
    ) -> object:
        return {"limit": limit, "features": feature_columns}

    def save_report(self, report: object) -> None:
        self.saved.append(report)


def make_settings(tmp_path: Path) -> Settings:
    return Settings(  # type: ignore[call-arg]  # `_env_file` is runtime-only
        _env_file=None,
        app_env="test",
        monitoring_report_dir=tmp_path,
        monitoring_reference_path=tmp_path / "reference.json",
    )


def make_reference() -> SimpleNamespace:
    return SimpleNamespace(features=(SimpleNamespace(name="V1"),))


def make_report() -> SimpleNamespace:
    return SimpleNamespace(
        report_id="report-1",
        status=DriftStatus.OK,
        prediction_count=3,
        validation_failure_count=0,
    )


def install_fakes(monkeypatch: pytest.MonkeyPatch, *, write_error: bool) -> FakeEngine:
    engine = FakeEngine()
    FakeRepository.instances.clear()
    monkeypatch.setattr(cli, "load_reference_profile", lambda _path: make_reference())
    monkeypatch.setattr(cli, "create_db_engine", lambda _settings: engine)
    monkeypatch.setattr(cli, "build_session_factory", lambda _engine: FakeSession)
    monkeypatch.setattr(cli, "MonitoringRepository", FakeRepository)
    monkeypatch.setattr(cli, "build_monitoring_report", lambda **_kwargs: make_report())
    monkeypatch.setattr(
        cli,
        "write_report",
        fail_write if write_error else lambda _report, directory: Path(directory) / "report.json",
    )
    return engine


def fail_write(_report: object, _directory: Path) -> Path:
    raise OSError("disk full")


def test_monitoring_cli_writes_and_persists_a_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = install_fakes(monkeypatch, write_error=False)

    exit_code = cli.main(
        [
            "run",
            "--reference",
            str(tmp_path / "reference.json"),
            "--window-end",
            "2026-10-23T12:00:00Z",
            "--report-dir",
            str(tmp_path),
        ],
        settings=make_settings(tmp_path),
    )

    assert exit_code == cli.EXIT_OK
    assert engine.disposed is True
    assert len(FakeRepository.instances) == 1
    assert len(FakeRepository.instances[0].saved) == 1


def test_monitoring_cli_no_persist_skips_the_database_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = install_fakes(monkeypatch, write_error=False)

    exit_code = cli.main(
        [
            "run",
            "--window-end",
            "2026-10-23T12:00:00Z",
            "--report-dir",
            str(tmp_path),
            "--no-persist",
        ],
        settings=make_settings(tmp_path),
    )

    assert exit_code == cli.EXIT_OK
    assert engine.disposed is True
    assert FakeRepository.instances[0].saved == []


def test_monitoring_cli_contains_a_report_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A scheduled report must exit non-zero instead of raising an uncaught OSError."""
    engine = install_fakes(monkeypatch, write_error=True)

    exit_code = cli.main(
        [
            "run",
            "--window-end",
            "2026-10-23T12:00:00Z",
            "--report-dir",
            str(tmp_path),
        ],
        settings=make_settings(tmp_path),
    )

    assert exit_code == cli.EXIT_INVALID
    assert engine.disposed is True


def test_monitoring_cli_rejects_invalid_window_values_before_opening_a_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = FakeEngine()
    monkeypatch.setattr(cli, "create_db_engine", lambda _settings: engine)

    assert (
        cli.main(
            ["run", "--window-hours", "0", "--report-dir", str(tmp_path)],
            settings=make_settings(tmp_path),
        )
        == cli.EXIT_INVALID
    )
    assert (
        cli.main(
            ["run", "--window-end", "not-a-date", "--report-dir", str(tmp_path)],
            settings=make_settings(tmp_path),
        )
        == cli.EXIT_INVALID
    )
    assert engine.disposed is False


def test_parse_optional_datetime_requires_a_timezone() -> None:
    assert cli._parse_optional_datetime(None) is None
    assert cli._parse_optional_datetime("2026-10-23T12:00:00") is None
    assert cli._parse_optional_datetime("not-a-date") is None
    parsed = cli._parse_optional_datetime("2026-10-23T12:00:00+02:00")
    assert parsed == datetime(2026, 10, 23, 10, 0, tzinfo=UTC)


def test_parser_requires_a_subcommand() -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([])


def test_parser_keeps_explicit_overrides(tmp_path: Path) -> None:
    options = cli.build_parser().parse_args(
        [
            "run",
            "--reference",
            str(tmp_path / "custom.json"),
            "--window-hours",
            "2",
            "--report-dir",
            str(tmp_path / "out"),
        ]
    )
    assert options.reference == tmp_path / "custom.json"
    assert options.window_hours == 2
