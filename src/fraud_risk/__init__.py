"""Fraud Detection & Risk Intelligence Platform.

An end-to-end pipeline that scores credit-card transactions for fraud risk, explains each
score, serves predictions over HTTP, persists prediction history and monitors for drift.

Most subpackages are intentionally empty scaffolding at this stage. See
``docs/ARCHITECTURE.md`` for the component map and ``docs/ROADMAP.md`` for what is
implemented versus planned.
"""

from __future__ import annotations

__all__ = ["__version__"]

# Keep in step with `project.version` in pyproject.toml; enforced by tests/test_config.py.
__version__ = "0.1.0"
