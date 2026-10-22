"""Tests for the configuration contract.

Deliberately independent of the machine they run on: the configuration environment variables
are removed and any local ``.env`` file is ignored, so the assertions describe the real
defaults rather than whatever happens to be exported by the developer.

Coverage: defaults and safe local values, environment-variable overrides, invalid settings,
production-startup validation, and the guarantee that importing configuration connects to
nothing.
"""

from __future__ import annotations

import os
import socket

# `subprocess` is imported only to prove, in an isolated interpreter, that importing the
# configuration does not pull in a database driver or MLflow. See the test below.
import subprocess
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import fraud_risk
from fraud_risk.config import PLACEHOLDER_SECRETS, Settings, get_settings

REPO_ROOT = Path(__file__).resolve().parents[1]

# Every environment variable `Settings` can read. Kept in step with the model by
# `test_cleared_variable_list_tracks_the_model` below.
CONFIG_ENV_VARS = (
    "APP_ENV",
    "LOG_LEVEL",
    "API_HOST",
    "API_PORT",
    "API_PREFIX",
    "CORS_ALLOWED_ORIGINS",
    "BATCH_MAX_SIZE",
    "API_MAX_REQUEST_BYTES",
    "API_ADMIN_TOKEN",
    "DASHBOARD_API_URL",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "POSTGRES_DB",
    "POSTGRES_HOST",
    "POSTGRES_PORT",
    "DATABASE_URL",
    "MLFLOW_TRACKING_URI",
    "MLFLOW_EXPERIMENT_NAME",
    "MODEL_NAME",
    "MODEL_ALIAS",
    "MODEL_DIR",
    "MODEL_ARTIFACT_PATH",
    "RANDOM_SEED",
    "FALSE_NEGATIVE_COST",
    "FALSE_POSITIVE_COST",
    "REVIEW_THRESHOLD",
    "BLOCK_THRESHOLD",
    "PERSIST_FEATURE_SNAPSHOT",
)

EXPECTED_DEFAULT_DSN = "postgresql+psycopg://fraud:change-me@localhost:5432/fraud"
PRODUCTION_DSN = "postgresql+psycopg://fraud:REAL@db.internal:5432/fraud"


@pytest.fixture()
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove configuration environment variables so defaults are genuinely exercised."""
    for name in CONFIG_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture()
def load_settings(clean_env: None) -> Callable[..., Settings]:
    """Return a factory that builds ``Settings`` with any local ``.env`` file ignored.

    ``_env_file`` is a runtime-only keyword of pydantic-settings that mypy cannot see on the
    generated ``__init__`` signature, so the ignore is scoped to this single call site.
    """

    def _load(**overrides: Any) -> Settings:
        return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]

    return _load


# ---------------------------------------------------------------------------
# Test-harness self-check
# ---------------------------------------------------------------------------
def test_cleared_variable_list_tracks_the_model() -> None:
    """``CONFIG_ENV_VARS`` must cover every field, or the defaults tests pass vacuously."""
    field_names = {name.upper() for name in Settings.model_fields}

    assert field_names == set(CONFIG_ENV_VARS)


def test_package_version_matches_pyproject() -> None:
    """``fraud_risk.__version__`` and ``project.version`` must not drift apart."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert fraud_risk.__version__ == pyproject["project"]["version"]


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
def test_defaults_are_local_and_secret_free(load_settings: Callable[..., Settings]) -> None:
    """A fresh checkout comes up in local mode."""
    settings = load_settings()

    assert settings.app_env == "local"
    assert settings.log_level == "INFO"
    assert settings.persist_feature_snapshot is False
    assert settings.database_url is None


def test_defaults_point_nowhere_real(load_settings: Callable[..., Settings]) -> None:
    """Development defaults must not reference a plausible production target."""
    settings = load_settings()

    assert settings.postgres_host == "localhost"
    assert settings.mlflow_tracking_uri == "sqlite:///mlflow.db"  # local store, no server
    assert not settings.model_dir.is_absolute()
    assert not settings.model_artifact_path.is_absolute()
    assert settings.dashboard_api_url.startswith("http://localhost")

    # The only credential default is an obvious non-secret, and it is refused in production.
    assert settings.postgres_password in PLACEHOLDER_SECRETS


def test_default_api_binding_and_prefix(load_settings: Callable[..., Settings]) -> None:
    """The API binds to loopback and mounts under a versioned prefix by default."""
    settings = load_settings()

    assert settings.api_host == "127.0.0.1"
    assert settings.api_port == 8000
    assert settings.api_prefix == "/api/v1"
    assert settings.cors_origins == ()
    assert settings.batch_max_size == 100
    assert settings.api_max_request_bytes == 1_048_576


def test_random_seed_defaults_and_is_bounded(load_settings: Callable[..., Settings]) -> None:
    """A run is reproducible by default, and an unusable seed fails at startup."""
    assert load_settings().random_seed == 42

    with pytest.raises(ValidationError):
        load_settings(random_seed=-1)
    with pytest.raises(ValidationError):
        load_settings(random_seed=2**32)


def test_cost_weights_default_and_must_be_positive(
    load_settings: Callable[..., Settings],
) -> None:
    """The provisional training cost model is explicit configuration, never a code constant."""
    settings = load_settings()

    assert settings.false_negative_cost == 100.0
    assert settings.false_positive_cost == 1.0

    with pytest.raises(ValidationError):
        load_settings(false_negative_cost=0)
    with pytest.raises(ValidationError):
        load_settings(false_positive_cost=-1)


def test_required_settings_are_all_present(load_settings: Callable[..., Settings]) -> None:
    """Every setting the architecture requires resolves to a usable value by default."""
    settings = load_settings()

    # Application environment, logging, API prefix, database URL, MLflow tracking URI,
    # model artifact location, dashboard API URL.
    assert settings.app_env
    assert settings.log_level
    assert settings.api_prefix
    assert settings.sqlalchemy_dsn
    assert settings.mlflow_tracking_uri
    assert settings.model_dir
    assert settings.model_artifact_path
    assert settings.dashboard_api_url


# ---------------------------------------------------------------------------
# Environment overrides
# ---------------------------------------------------------------------------
def test_environment_variables_override_defaults(
    load_settings: Callable[..., Settings], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Values supplied through the environment take precedence over every default."""
    monkeypatch.setenv("APP_ENV", "staging")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    # TEST-NET-1 (RFC 5737): a documented, never-routable address. The point is that the
    # override is honoured, not that anything is bound, so this stays off 0.0.0.0.
    monkeypatch.setenv("API_HOST", "192.0.2.10")
    monkeypatch.setenv("API_PORT", "9001")
    monkeypatch.setenv("API_PREFIX", "/internal/v2")
    monkeypatch.setenv(
        "CORS_ALLOWED_ORIGINS",
        "http://localhost:8501,https://dashboard.example.internal",
    )
    monkeypatch.setenv("BATCH_MAX_SIZE", "7")
    monkeypatch.setenv("API_MAX_REQUEST_BYTES", "2048")
    monkeypatch.setenv("API_ADMIN_TOKEN", "staging-admin-token")
    monkeypatch.setenv("DASHBOARD_API_URL", "https://api.example.internal")
    monkeypatch.setenv("DATABASE_URL", PRODUCTION_DSN)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow.internal:5000")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "fraud-staging")
    monkeypatch.setenv("MODEL_NAME", "fraud-risk-classifier-staging")
    monkeypatch.setenv("MODEL_ALIAS", "challenger")
    monkeypatch.setenv("MODEL_DIR", "/srv/models")
    monkeypatch.setenv("MODEL_ARTIFACT_PATH", "/srv/models/champion/model.joblib")
    monkeypatch.setenv("FALSE_NEGATIVE_COST", "50")
    monkeypatch.setenv("FALSE_POSITIVE_COST", "2.5")
    monkeypatch.setenv("REVIEW_THRESHOLD", "0.2")
    monkeypatch.setenv("BLOCK_THRESHOLD", "0.8")
    monkeypatch.setenv("PERSIST_FEATURE_SNAPSHOT", "true")

    settings = load_settings()

    assert settings.app_env == "staging"
    assert settings.log_level == "WARNING"
    assert settings.api_host == "192.0.2.10"
    assert settings.api_port == 9001
    assert settings.api_prefix == "/internal/v2"
    assert settings.cors_origins == (
        "http://localhost:8501",
        "https://dashboard.example.internal",
    )
    assert settings.batch_max_size == 7
    assert settings.api_max_request_bytes == 2048
    assert settings.api_admin_token is not None
    assert settings.api_admin_token.get_secret_value() == "staging-admin-token"
    assert settings.dashboard_api_url == "https://api.example.internal"
    assert settings.mlflow_tracking_uri == "http://mlflow.internal:5000"
    assert settings.mlflow_experiment_name == "fraud-staging"
    assert settings.model_name == "fraud-risk-classifier-staging"
    assert settings.model_alias == "challenger"
    assert settings.model_dir == Path("/srv/models")
    assert settings.model_artifact_path == Path("/srv/models/champion/model.joblib")
    assert settings.false_negative_cost == 50.0
    assert settings.false_positive_cost == 2.5
    assert settings.review_threshold == 0.2
    assert settings.block_threshold == 0.8
    assert settings.persist_feature_snapshot is True
    assert settings.sqlalchemy_dsn == PRODUCTION_DSN


def test_postgres_parts_override_builds_the_dsn(
    load_settings: Callable[..., Settings], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The discrete POSTGRES_* variables assemble a DSN when DATABASE_URL is absent."""
    monkeypatch.setenv("POSTGRES_USER", "svc_fraud")
    monkeypatch.setenv("POSTGRES_PASSWORD", "from-vault")
    monkeypatch.setenv("POSTGRES_HOST", "db.internal")
    monkeypatch.setenv("POSTGRES_PORT", "6432")
    monkeypatch.setenv("POSTGRES_DB", "risk")

    settings = load_settings()

    assert (
        settings.sqlalchemy_dsn == "postgresql+psycopg://svc_fraud:from-vault@db.internal:6432/risk"
    )


def test_database_url_override_wins(load_settings: Callable[..., Settings]) -> None:
    """An explicit DATABASE_URL takes precedence over the discrete POSTGRES_* values."""
    settings = load_settings(database_url=PRODUCTION_DSN, postgres_host="ignored")

    assert settings.sqlalchemy_dsn == PRODUCTION_DSN


def test_password_is_percent_encoded_in_dsn(load_settings: Callable[..., Settings]) -> None:
    """Passwords containing URL-significant characters must not corrupt the DSN."""
    # A literal, obviously-fake password is the subject of this test, not a leak.
    settings = load_settings(postgres_password="p@ss/word")  # noqa: S106

    expected = "postgresql+psycopg://fraud:p%40ss%2Fword@localhost:5432/fraud"
    assert settings.sqlalchemy_dsn == expected


# ---------------------------------------------------------------------------
# Invalid settings
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_env", ["prod", "PRODUCTION", "dev", ""])
def test_invalid_app_env_is_rejected(load_settings: Callable[..., Settings], bad_env: str) -> None:
    """Only the four documented environments are accepted."""
    with pytest.raises(ValidationError, match="app_env"):
        load_settings(app_env=bad_env)


def test_invalid_log_level_is_rejected(load_settings: Callable[..., Settings]) -> None:
    """A misspelled level must fail loudly rather than silently disabling logging."""
    with pytest.raises(ValidationError, match="log_level"):
        load_settings(log_level="VERBOSE")


@pytest.mark.parametrize("bad_port", [0, 70000, -1])
def test_invalid_api_port_is_rejected(
    load_settings: Callable[..., Settings], bad_port: int
) -> None:
    """Ports outside the TCP range are configuration errors."""
    with pytest.raises(ValidationError, match="api_port"):
        load_settings(api_port=bad_port)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("batch_max_size", 0),
        ("batch_max_size", 10_001),
        ("api_max_request_bytes", 1023),
        ("api_max_request_bytes", 16_777_217),
    ],
)
def test_batch_and_body_limits_are_bounded(
    load_settings: Callable[..., Settings],
    field: str,
    bad_value: int,
) -> None:
    """A nonsensical batch or body limit must fail at startup, not at request time."""
    with pytest.raises(ValidationError, match=field):
        load_settings(**{field: bad_value})


def test_admin_token_is_normalised_bounded_and_not_rendered(
    load_settings: Callable[..., Settings],
) -> None:
    """A blank token disables admin routes; a weak one fails startup; dumps never leak it."""
    assert load_settings(api_admin_token="   ").api_admin_token is None  # noqa: S106 - synthetic
    padded_token = "  " + "a" * 16 + "  "
    normalised = load_settings(api_admin_token=padded_token)
    assert normalised.api_admin_token is not None
    assert normalised.api_admin_token.get_secret_value() == "a" * 16

    with pytest.raises(ValidationError, match="api_admin_token"):
        load_settings(api_admin_token="too-short")  # noqa: S106 - synthetic

    settings = load_settings(api_admin_token="a" * 16)
    assert "a" * 16 not in repr(settings)
    assert "a" * 16 not in str(settings.model_dump())


def test_api_prefix_must_be_absolute(load_settings: Callable[..., Settings]) -> None:
    """A relative mount point would silently change every route."""
    with pytest.raises(ValidationError, match="must start with"):
        load_settings(api_prefix="api/v1")


def test_api_prefix_must_not_end_with_a_slash(load_settings: Callable[..., Settings]) -> None:
    """A trailing slash produces doubled separators once routes are mounted."""
    with pytest.raises(ValidationError, match="must not end with"):
        load_settings(api_prefix="/api/v1/")


@pytest.mark.parametrize("root_prefix", ["", "/"])
def test_root_api_prefix_is_canonicalised(
    load_settings: Callable[..., Settings], root_prefix: str
) -> None:
    """Both spellings of "serve at the root" normalise to the empty prefix."""
    assert load_settings(api_prefix=root_prefix).api_prefix == ""


@pytest.mark.parametrize(
    "bad_origin",
    [
        "dashboard.example",
        "https://dashboard.example/",
        "https://dashboard.example/path",
        "https://user:pass@dashboard.example",
        "https://dashboard.example?query=1",
        "*,https://dashboard.example",
    ],
)
def test_cors_origins_must_be_exact_http_origins(
    load_settings: Callable[..., Settings], bad_origin: str
) -> None:
    """CORS entries are browser origins, not URLs, globs or credentials."""
    with pytest.raises(ValidationError, match="cors_allowed_origins"):
        load_settings(cors_allowed_origins=bad_origin)


def test_cors_wildcard_is_allowed_only_in_local_and_test(
    load_settings: Callable[..., Settings],
) -> None:
    """A wildcard is a development convenience, never a staging/production policy."""
    assert load_settings(cors_allowed_origins="*").cors_origins == ("*",)
    assert load_settings(app_env="test", cors_allowed_origins="*").cors_origins == ("*",)

    for app_env in ("staging", "production"):
        with pytest.raises(ValidationError, match="only allowed in local/test"):
            load_settings(
                app_env=app_env,
                database_url=PRODUCTION_DSN,
                cors_allowed_origins="*",
            )


def test_production_accepts_explicit_cors_origins(
    load_settings: Callable[..., Settings],
) -> None:
    """Production can allow a known dashboard origin without opening the API to everyone."""
    settings = load_settings(
        app_env="production",
        database_url=PRODUCTION_DSN,
        cors_allowed_origins="https://dashboard.example.internal",
    )

    assert settings.cors_origins == ("https://dashboard.example.internal",)


@pytest.mark.parametrize(
    "bad_dsn",
    ["sqlite:///./fraud.db", "mysql+mysqlconnector://u:p@h/db", "postgres://u:p@h/db"],
)
def test_non_postgresql_database_url_is_rejected(
    load_settings: Callable[..., Settings], bad_dsn: str
) -> None:
    """This project is PostgreSQL-only; a SQLite DSN must not slip through."""
    with pytest.raises(ValidationError, match="must be a PostgreSQL SQLAlchemy URL"):
        load_settings(database_url=bad_dsn)


def test_thresholds_define_a_non_empty_review_band(load_settings: Callable[..., Settings]) -> None:
    """The three-way policy is meaningless if the review band is empty or inverted."""
    with pytest.raises(ValidationError, match="review_threshold must be strictly below"):
        load_settings(review_threshold=0.9, block_threshold=0.9)


def test_thresholds_must_be_probabilities(load_settings: Callable[..., Settings]) -> None:
    """Scores are probabilities; a threshold outside [0, 1] is a configuration error."""
    with pytest.raises(ValidationError):
        load_settings(block_threshold=1.5)


# ---------------------------------------------------------------------------
# Production startup validation
# ---------------------------------------------------------------------------
def test_production_requires_an_explicit_database_target(
    load_settings: Callable[..., Settings],
) -> None:
    """Production must not silently inherit the localhost development database."""
    with pytest.raises(ValidationError, match="production requires an explicit database target"):
        load_settings(app_env="production")


def test_production_rejects_placeholder_password(
    load_settings: Callable[..., Settings], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The committed .env.example placeholder must never reach production."""
    monkeypatch.setenv("POSTGRES_PASSWORD", "change-me")

    with pytest.raises(ValidationError, match="still a placeholder value"):
        load_settings(app_env="production")


@pytest.mark.parametrize("placeholder", sorted(PLACEHOLDER_SECRETS))
def test_production_rejects_every_known_placeholder(
    load_settings: Callable[..., Settings], placeholder: str
) -> None:
    """Every value published in .env.example is refused in production."""
    with pytest.raises(ValidationError, match="still a placeholder value"):
        load_settings(app_env="production", postgres_password=placeholder)


def test_production_rejects_an_empty_password(
    load_settings: Callable[..., Settings],
) -> None:
    """An explicitly supplied but empty secret is a misconfiguration, not a credential."""
    with pytest.raises(ValidationError, match="must not be empty in production"):
        load_settings(app_env="production", postgres_password="   ")  # noqa: S106


def test_production_accepts_a_secret_from_the_environment(
    load_settings: Callable[..., Settings], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicitly supplied real secret starts cleanly."""
    monkeypatch.setenv("POSTGRES_PASSWORD", "s3cret-from-a-secret-manager")

    settings = load_settings(app_env="production")

    assert settings.app_env == "production"


def test_production_accepts_an_explicit_database_url(
    load_settings: Callable[..., Settings],
) -> None:
    """DATABASE_URL carries its own credentials, so no separate password is required."""
    settings = load_settings(app_env="production", database_url=PRODUCTION_DSN)

    assert settings.sqlalchemy_dsn == PRODUCTION_DSN


def test_non_production_allows_the_development_placeholder(
    load_settings: Callable[..., Settings],
) -> None:
    """Outside production the documented development default stays usable."""
    for app_env in ("local", "test", "staging"):
        assert load_settings(app_env=app_env).postgres_password in PLACEHOLDER_SECRETS


# ---------------------------------------------------------------------------
# No implicit I/O
# ---------------------------------------------------------------------------
def test_constructing_settings_opens_no_network_connection(
    load_settings: Callable[..., Settings], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resolving configuration is pure string work: nothing may dial out."""

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("configuration attempted to open a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)

    settings = load_settings()

    assert settings.sqlalchemy_dsn.startswith("postgresql+psycopg://")
    assert settings.mlflow_tracking_uri == "sqlite:///mlflow.db"


def test_importing_config_pulls_in_no_driver_or_tracking_client(tmp_path: Path) -> None:
    """Importing configuration must not drag in psycopg, SQLAlchemy, Alembic or MLflow.

    Run in a subprocess with a temporary working directory: ``tests/test_smoke_imports.py``
    imports all of these into the current process, so an in-process ``sys.modules`` check
    would be meaningless, and a temporary ``cwd`` means a developer's local ``.env`` cannot
    influence the result.
    """
    script = (
        "import sys\n"
        "import fraud_risk.config as config\n"
        "config.get_settings()\n"
        "leaked = sorted(\n"
        "    name\n"
        "    for name in ('alembic', 'mlflow', 'psycopg', 'psycopg2', 'sqlalchemy')\n"
        "    if name in sys.modules\n"
        ")\n"
        "print('|'.join(leaked))\n"
    )
    env = {key: value for key, value in os.environ.items() if key.upper() not in CONFIG_ENV_VARS}
    env["PYTHONPATH"] = str(REPO_ROOT / "src")

    # Fixed argv, no shell, absolute interpreter path.
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        cwd=tmp_path,
        env=env,
    )

    assert result.stdout.strip() == "", f"importing config pulled in: {result.stdout.strip()}"


# ---------------------------------------------------------------------------
# Process-wide instance
# ---------------------------------------------------------------------------
def test_get_settings_is_cached(clean_env: None) -> None:
    """Configuration is read once per process."""
    get_settings.cache_clear()
    try:
        assert get_settings() is get_settings()
    finally:
        get_settings.cache_clear()
