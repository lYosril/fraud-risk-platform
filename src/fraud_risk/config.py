"""Application settings.

The single place where environment variables are read. Everything environment-specific —
API binding and prefix, database target, MLflow tracking URI, model artifact location,
dashboard API URL, decision thresholds — is resolved here, so that no credential, host or
policy value is hardcoded anywhere else.

Three guarantees this module deliberately provides:

1. **Importing it connects to nothing.** Building :class:`Settings` performs no I/O: it does
   not import a database driver, open a socket, or contact an MLflow tracking server.
   Connection setup belongs to the component that needs it (``db.session`` builds the engine,
   the model registry resolves artifacts). This keeps configuration importable from scripts,
   notebooks and tests without a reachable database.
2. **Production cannot inherit a development default.** Constructing ``Settings`` with
   ``app_env="production"`` fails unless a real database target and credentials are supplied.
   ``Settings`` is built during application startup, so this *is* the startup validation.
3. **Secrets come from the environment only.** No secret has a usable default. The committed
   placeholder in ``.env.example`` is refused in production.

A local ``.env`` file (git-ignored) may supply values; copy ``.env.example`` to ``.env``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import quote_plus

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

AppEnv = Literal["local", "test", "staging", "production"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

#: Values published in the committed `.env.example`. Acceptable for local development,
#: refused in production so that a placeholder can never silently reach a real deployment.
PLACEHOLDER_SECRETS = frozenset(
    {"change-me", "changeme", "replace-me", "password", "secret", "todo"}
)

#: This project is PostgreSQL-only (docs/ARCHITECTURE.md §5.3); SQLite is never a substitute.
POSTGRES_SCHEME_PREFIX = "postgresql"


class Settings(BaseSettings):
    """Validated application configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -- Application ---------------------------------------------------------
    app_env: AppEnv = "local"
    log_level: LogLevel = "INFO"

    # -- API -----------------------------------------------------------------
    # Bound to loopback by default: the API is unauthenticated at this stage
    # (docs/ARCHITECTURE.md §13.5). Change this deliberately, not by accident.
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1, le=65535)
    # Mount point for the routers. `""` means the application is served at the root.
    api_prefix: str = "/api/v1"

    # -- Dashboard -----------------------------------------------------------
    # Where the Streamlit dashboard reaches the API. Distinct from `api_host`/`api_port`,
    # which are where *this* process binds: inside a container those are not the same value.
    dashboard_api_url: str = "http://localhost:8000"

    # -- PostgreSQL ----------------------------------------------------------
    postgres_user: str = "fraud"
    # Deliberately-fake local placeholder matching docker-compose. The real value comes from
    # the environment; `repr=False` keeps it out of logs and tracebacks, and production
    # refuses this placeholder outright.
    postgres_password: str = Field(default="change-me", repr=False)
    postgres_db: str = "fraud"
    postgres_host: str = "localhost"
    postgres_port: int = Field(default=5432, ge=1, le=65535)
    # Explicit override; when set it wins over the discrete POSTGRES_* values.
    database_url: str | None = None

    # -- MLflow --------------------------------------------------------------
    # Defaults to a local file store so no tracking server is needed for development.
    mlflow_tracking_uri: str = "./mlruns"
    mlflow_experiment_name: str = "fraud-detection"

    # -- Model artifacts -----------------------------------------------------
    model_name: str = "fraud-risk-classifier"
    # Registry alias resolved at inference time; promoted deliberately (never automatically).
    model_alias: str = "champion"
    # Directory holding published artifact bundles.
    model_dir: Path = Path("./artifacts/models")

    # -- Decision policy (three-way, docs/ARCHITECTURE.md §6) ----------------
    review_threshold: float = Field(default=0.30, ge=0.0, le=1.0)
    block_threshold: float = Field(default=0.70, ge=0.0, le=1.0)

    # -- Inference / persistence --------------------------------------------
    # Persisting input vectors enables drift analysis but costs storage and privacy
    # surface (docs/ARCHITECTURE.md §13.4). Off by default.
    persist_feature_snapshot: bool = False

    # -- Field validation ----------------------------------------------------
    @field_validator("api_prefix")
    @classmethod
    def _validate_api_prefix(cls, value: str) -> str:
        """Require an absolute mount point, and canonicalise the root to ``""``.

        A trailing slash would produce doubled separators once routes are mounted
        (``//predict``), so it is rejected rather than silently trimmed.
        """
        if value in {"", "/"}:
            return ""
        if not value.startswith("/"):
            msg = f"api_prefix must start with '/' (got {value!r}); use '' for the root"
            raise ValueError(msg)
        if value.endswith("/"):
            msg = f"api_prefix must not end with '/' (got {value!r})"
            raise ValueError(msg)
        return value

    @field_validator("database_url")
    @classmethod
    def _validate_database_url(cls, value: str | None) -> str | None:
        """Reject a non-PostgreSQL DSN instead of failing later at connect time."""
        if value is None:
            return None
        if not value.startswith(POSTGRES_SCHEME_PREFIX):
            msg = (
                "database_url must be a PostgreSQL SQLAlchemy URL starting with "
                f"'{POSTGRES_SCHEME_PREFIX}' (got {value.split(':', 1)[0]!r} scheme)"
            )
            raise ValueError(msg)
        return value

    # -- Cross-field validation ---------------------------------------------
    @model_validator(mode="after")
    def _validate_threshold_order(self) -> Settings:
        """Reject a decision policy where the review band is empty or inverted."""
        if self.review_threshold >= self.block_threshold:
            msg = (
                "review_threshold must be strictly below block_threshold "
                f"(got {self.review_threshold} >= {self.block_threshold})"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _validate_production_requirements(self) -> Settings:
        """Refuse to start in production on development defaults.

        An explicit ``DATABASE_URL`` carries its own credentials, so it is accepted as-is.
        Otherwise the discrete ``POSTGRES_*`` values are used, and the password must have
        been supplied from the environment and must not be a committed placeholder.
        """
        if self.app_env != "production":
            return self

        if self.database_url is not None:
            return self

        if "postgres_password" not in self.model_fields_set:
            msg = (
                "production requires an explicit database target: set DATABASE_URL, or set "
                "POSTGRES_PASSWORD from the environment. The development placeholder must "
                "never reach production (see .env.example)."
            )
            raise ValueError(msg)

        password = self.postgres_password.strip()

        if not password:
            msg = (
                "POSTGRES_PASSWORD must not be empty in production; provide the real secret "
                "through the environment"
            )
            raise ValueError(msg)

        if password.lower() in PLACEHOLDER_SECRETS:
            msg = (
                "POSTGRES_PASSWORD is still a placeholder value from .env.example; "
                "provide the real secret through the environment"
            )
            raise ValueError(msg)

        return self

    # -- Derived values ------------------------------------------------------
    @property
    def sqlalchemy_dsn(self) -> str:
        """SQLAlchemy DSN, preferring an explicit ``DATABASE_URL`` override."""
        if self.database_url:
            return self.database_url
        password = quote_plus(self.postgres_password)
        return (
            f"postgresql+psycopg://{self.postgres_user}:{password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance, read once per process.

    Reading configuration here performs no I/O beyond optionally reading the local ``.env``
    file. Tests that need different values should construct ``Settings(...)`` directly rather
    than mutating this cached instance.
    """
    return Settings()
