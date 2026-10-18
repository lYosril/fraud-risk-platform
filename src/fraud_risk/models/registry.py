"""Model artifact management.

Responsibility
    Define the artifact bundle layout (fitted transformer + calibrated model + metadata) and
    publish/resolve it: write a new version, resolve a named alias such as ``champion``,
    list available versions. MLflow registry aliases back this, with a filesystem store for
    the local proof of concept.

Rules
    - A bundle is self-contained: it must not depend on the training process, the working
      directory or an environment variable to be readable.
    - Loading a missing or corrupt bundle raises a clear error the API can turn into 503.
    - Promotion is an explicit, reviewed step; never automatic.

Status
    Scaffolding only. See ``docs/ROADMAP.md`` (M5).
"""
