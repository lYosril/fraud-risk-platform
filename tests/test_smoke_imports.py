"""Smoke test: this interpreter can serve the architecture's declared dependencies.

Deliberately shallow. It answers three narrow questions and nothing else:

1. Does ``pyproject.toml`` declare every package the agreed architecture depends on?
2. Are the *installed* versions inside the ranges declared there?
3. Can those packages actually be imported, including the specific classes the pipeline,
   API and database layers are built on?

Passing this says the environment is provisioned. It says **nothing** about the correctness
of any application module — no scoring, no persistence, no endpoint behaviour is exercised.

The import tests are marked ``slow`` (the ML and dashboard stack takes several seconds to
import), so a light environment can run ``pytest -q -m "not slow"``.
"""

from __future__ import annotations

import importlib
import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest
from packaging.requirements import Requirement

REPO_ROOT = Path(__file__).resolve().parents[1]

# Every package the architecture requires, by *distribution* name (which is not always the
# import name: `scikit-learn` -> `sklearn`, `SQLAlchemy` -> `sqlalchemy`, ...).
ARCHITECTURE_PACKAGES = frozenset(
    {
        "alembic",
        "fastapi",
        "httpx",
        "mlflow",
        "numpy",
        "pandas",
        "plotly",
        "psycopg",
        "pydantic",
        "pytest",
        "scikit-learn",
        "shap",
        "SQLAlchemy",
        "streamlit",
        "xgboost",
    }
)

# Import name -> distribution name.
IMPORT_NAMES: dict[str, str] = {
    "alembic": "alembic",
    "fastapi": "fastapi",
    "httpx": "httpx",
    "mlflow": "mlflow",
    "numpy": "numpy",
    "pandas": "pandas",
    "plotly": "plotly",
    "psycopg": "psycopg",
    "pydantic": "pydantic",
    "pydantic_settings": "pydantic-settings",
    "pytest": "pytest",
    "shap": "shap",
    "sklearn": "scikit-learn",
    "sqlalchemy": "SQLAlchemy",
    "streamlit": "streamlit",
    "xgboost": "xgboost",
}

# The specific surfaces each layer is built on. A renamed or missing symbol here is a real
# breakage that a bare `import xgboost` would not catch.
REQUIRED_SYMBOLS = (
    "sklearn.compose.ColumnTransformer",  # features.engineering
    "sklearn.pipeline.Pipeline",  # features.engineering
    "sklearn.calibration.CalibratedClassifierCV",  # models.training (D5)
    "xgboost.XGBClassifier",  # models.training
    "shap.TreeExplainer",  # models.inference (explanations)
    "sqlalchemy.orm.DeclarativeBase",  # db.base
    "alembic.config.Config",  # db migrations
    "fastapi.FastAPI",  # api.main
    "httpx.ASGITransport",  # api tests; direct ASGI transport, no TestClient bridge
    "plotly.express.line",  # dashboard
)


def _declared_requirements() -> list[Requirement]:
    """Every requirement declared in ``pyproject.toml``, across all extras."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]

    raw = list(project["dependencies"])
    for extra_requirements in project.get("optional-dependencies", {}).values():
        raw.extend(extra_requirements)

    return [Requirement(entry) for entry in raw]


def test_architecture_packages_are_declared() -> None:
    """Every package the architecture needs is declared, rather than assumed present."""
    declared = {requirement.name.lower() for requirement in _declared_requirements()}
    required = {name.lower() for name in ARCHITECTURE_PACKAGES}

    assert required <= declared, f"pyproject.toml does not declare: {sorted(required - declared)}"


def test_installed_versions_satisfy_declared_ranges() -> None:
    """Installed versions stay inside the declared ranges, not merely "importable".

    Distributions that are not installed are skipped, so this also runs in a light
    environment; it fails if *nothing* is installed, to avoid passing vacuously.
    """
    checked: list[str] = []
    violations: list[str] = []

    for requirement in _declared_requirements():
        try:
            installed = version(requirement.name)
        except PackageNotFoundError:
            continue

        checked.append(f"{requirement.name}=={installed}")
        if installed not in requirement.specifier:
            violations.append(f"{requirement.name}=={installed} {requirement.specifier}")

    assert not violations, (
        "installed versions fall outside the ranges declared in pyproject.toml: "
        + "; ".join(violations)
    )
    assert checked, "no declared dependency is installed; this environment is not provisioned"


@pytest.mark.slow
@pytest.mark.parametrize(("import_name", "distribution"), sorted(IMPORT_NAMES.items()))
def test_core_package_imports(import_name: str, distribution: str) -> None:
    """Each core package imports, and reports a version."""
    module = importlib.import_module(import_name)

    assert module is not None
    assert version(distribution)


@pytest.mark.slow
@pytest.mark.parametrize("dotted_path", REQUIRED_SYMBOLS)
def test_required_symbols_are_importable(dotted_path: str) -> None:
    """The classes the pipeline, API and database layers depend on are importable."""
    module_path, _, symbol = dotted_path.rpartition(".")

    module = importlib.import_module(module_path)

    assert hasattr(module, symbol), f"{dotted_path} is missing from {module_path}"
