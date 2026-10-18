# Fraud Detection & Risk Intelligence Platform

An end-to-end machine learning platform that scores credit-card transactions for fraud risk,
**explains every decision**, serves predictions over a REST API, persists the full prediction
history in PostgreSQL, and monitors the model for drift.

This is a portfolio project built to demonstrate production-shaped engineering: clear module
boundaries, reproducible experiments, explicit evaluation, and an honest record of what works
and what does not.

---

## ⚠️ Current status: scaffolding only

**Milestone M0 (repository foundation) is complete, and the Python 3.11 development
environment is provisioned. Nothing is trained, served or monitored yet.**

- There is **no trained model** in this repository, and **no model metrics are reported
  anywhere**. Any number that appears in a design document is illustrative of a *metric
  definition*, never a result.
- `docs/ARCHITECTURE.md` describes the intended system. It is a design, not a description of
  working code.
- The only implemented logic is `fraud_risk.config.Settings`, which is covered by tests.
- All other modules are documented placeholders (see the status table below).
- The dependency set is installed and importable, and `tests/test_smoke_imports.py` checks
  that it stays that way — but no module that consumes those packages is implemented.

No transaction data is included. No secrets, datasets or model artifacts are committed.

---

## Project goals

1. **Detect** fraudulent transactions in a heavily imbalanced dataset with a model whose
   performance is measured honestly (PR-AUC first, never accuracy).
2. **Explain** each prediction with SHAP contributions, while being upfront about the limits
   of explaining PCA-derived features.
3. **Serve** predictions through a typed, versioned FastAPI service that persists every score
   it returns.
4. **Persist** prediction history, delayed analyst feedback, and drift reports in PostgreSQL
   with reviewable Alembic migrations.
5. **Monitor** data drift, score drift and (once labels arrive) performance decay.
6. **Visualise** the risk picture in a Streamlit dashboard.
7. **Stay maintainable**: one dependency file, one configuration entry point, one shared
   feature transform, tests that run in CI.

---

## Planned architecture

```
raw data (data/raw, git-ignored)
        │
        ▼
  data ingestion & validation ──► features (shared fitted transform) ──► model training
                                                                              │  MLflow
                                                                              ▼
  client ──HTTP──► FastAPI ──► inference (+ SHAP) ──► PostgreSQL ◄── Streamlit dashboard
                                                        ▲
                                              monitoring / drift reports
```

The core idea is a **single shared core** (`src/fraud_risk/`): the batch training path and the
online inference path both go through the same feature transformer, shipped inside the model
artifact bundle. Train/serve skew becomes structurally impossible instead of a thing we hope
does not happen.

| # | Area | Package | Status |
|---|------|---------|--------|
| 0 | Configuration | `fraud_risk/config.py` | **Implemented** (settings + tests) |
| 1 | Data ingestion & validation | `fraud_risk/data/` | Scaffolded |
| 2 | EDA & feature engineering | `fraud_risk/features/` | Scaffolded |
| 3 | Training & evaluation | `fraud_risk/models/training.py`, `evaluation.py` | Scaffolded |
| 4 | Artifacts & inference | `fraud_risk/models/registry.py`, `inference.py` | Scaffolded |
| 5 | FastAPI application | `fraud_risk/api/` | Scaffolded |
| 6 | Database layer | `fraud_risk/db/` | Scaffolded |
| 7 | Streamlit dashboard | `fraud_risk/dashboard/` | Scaffolded |
| 8 | Monitoring | `fraud_risk/monitoring/` | Scaffolded |
| 9 | Tests, CI, containers | `tests/`, `.github/`, `docker-compose.yml` | Scaffolded |

Full component responsibilities and the design rationale (including alternatives considered)
are in **[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)**. Milestone-by-milestone deliverables
and exit criteria are in **[`docs/ROADMAP.md`](docs/ROADMAP.md)**.

---

## Repository layout

```
fraud-risk-platform/
├── .env.example              # placeholder configuration only (commit this, not .env)
├── .gitattributes            # line-ending normalisation
├── .gitignore
├── .github/workflows/ci.yml  # lint + type-check + tests (not executed yet)
├── .python-version           # 3.11
├── AGENTS.md                 # conventions for future work in this repo
├── docker-compose.yml        # PostgreSQL only (app services arrive in M9)
├── pyproject.toml            # the single dependency + tooling definition
├── README.md
├── artifacts/                # model artifact store (git-ignored, .gitkeep only)
├── data/
│   ├── raw/                  # source dataset  (git-ignored)
│   ├── interim/              # intermediate data (git-ignored)
│   ├── processed/            # model-ready data (git-ignored)
│   └── sample/               # tiny synthetic/public-sample fixtures (tracked)
├── docs/
│   ├── ARCHITECTURE.md
│   └── ROADMAP.md
├── notebooks/                # exploration only; never imported by src/
├── src/fraud_risk/
│   ├── __init__.py           # __version__
│   ├── config.py             # Settings — the only reader of environment variables
│   ├── data/                 # ingestion, validation
│   ├── features/             # eda, engineering (the shared fitted transform)
│   ├── models/               # training, evaluation, registry, inference
│   ├── db/                   # base, models, session (+ Alembic migrations in M2)
│   ├── api/                  # main (app factory), schemas
│   ├── dashboard/            # Streamlit app
│   └── monitoring/           # drift
└── tests/
    ├── test_config.py          # configuration contract
    └── test_smoke_imports.py   # environment smoke test (declared / installed / importable)
```

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | **3.11.x** | The supported interpreter (`.python-version`). |
| pip | bundled with Python | The only package manager used. This project deliberately does **not** use uv, Poetry or conda. |
| Docker + Compose | recent | For local PostgreSQL and the Linux environment check. Not needed for the unit tests. |
| Git | 2.30+ | |

No dataset is required to validate this repository. Milestone M1 (see the roadmap) selects
the dataset and documents it in a data card.

### Getting Python 3.11

**Windows** — install from [python.org](https://www.python.org/downloads/) (tick *Add
python.exe to PATH*) or from the Microsoft Store, then confirm the launcher sees it:

```powershell
py --list-paths          # should list -V:3.11
```

**Debian / Ubuntu** — `venv` is a separate package, and this is the single most common
Linux setup failure:

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv
```

**Fedora / RHEL** — `sudo dnf install python3.11`.
**macOS** — `brew install python@3.11`.
Otherwise, `pyenv install 3.11.9` works anywhere but is not required.

> **Always invoke pip as `python -m pip`**, not bare `pip`. On every platform a stray `pip`
> on `PATH` can belong to a different interpreter than the one your virtualenv activated,
> and that silently installs into the wrong place.

---

## Setup

### 1. Clone and enter the repository

```bash
git clone <your-remote-url> fraud-risk-platform
cd fraud-risk-platform
```

### 2. Verify the interpreter before doing anything else

**PowerShell**

```powershell
py -3.11 -c "import sys; print(sys.version)"
py -3.11 -m pip --version
```

**Bash**

```bash
python3.11 -c "import sys; print(sys.version)"
python3.11 -m pip --version
```

The version string must start with `3.11`. If `py -3.11` or `python3.11` is not found, fix
that first — installing into 3.12+ produces an environment that resolves different package
versions and does not match the supported target.

### 3. Create and activate a virtual environment

**PowerShell**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# If activation is refused by the execution policy, allow it for this session only:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

**Bash**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Confirm the venv is the active interpreter (`python -c "import sys; print(sys.executable)"`
must point inside `.venv`) before continuing.

### 4. Install the project and dev tooling

**PowerShell and Bash are identical here** (the venv is already active):

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

`-e` installs the project in editable mode; `[dev]` adds pytest, httpx, ruff, mypy and
packaging. Extras are additive: `pip install -e ".[dev,monitoring]"` also installs Evidently.

Package versions are resolved from the compatible-release ranges in `pyproject.toml` rather
than pinned to "whatever is newest" — see *Dependency layout* in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) §14. The resolver legitimately picks
different versions on different interpreters (for example numpy 2.4.x on 3.11 where 2.5.x
requires a newer Python), which is the intended behaviour.

### 5. Create your local configuration

**PowerShell**

```powershell
Copy-Item .env.example .env
```

**Bash**

```bash
cp .env.example .env
```

`.env` is git-ignored and contains placeholders only.

### 6. (Optional) Start PostgreSQL

```bash
docker compose up -d postgres
docker compose ps
```

Not required for the unit tests — the database is first used in milestone M2.

---

## Validation commands

Run these to verify the initial repository.

### Fast path — no third-party ML dependencies required

```bash
py -3.11 -m venv .venv                 # PowerShell (then activate, see above)
python3.11 -m venv .venv               # Bash       (then activate, see above)

python -m pip install --upgrade pip
python -m pip install pydantic-settings packaging pytest ruff mypy

python -m compileall -q src tests   # every module compiles
ruff format --check .               # formatting is canonical
ruff check .                        # lint + security rules
mypy                                # strict typing
pytest -q -m "not slow"             # unit tests; skips the environment smoke test
```

### Full path — the real dependency set

```bash
python -m pip install -e ".[dev]"
ruff format --check . && ruff check . && mypy && pytest -q
```

### What to look for

- `ruff format --check .` and `ruff check .` report no findings.
- `mypy` reports `Success: no issues found` (strict mode).
- `pytest -q` passes two suites:
  - `tests/test_config.py` — the configuration contract (version parity, secret-free
    defaults, DSN construction and percent-encoding, threshold validation, caching).
  - `tests/test_smoke_imports.py` — the environment smoke test: every package the
    architecture requires is *declared*, *installed inside its declared version range*, and
    *importable*, including the specific classes each layer is built on
    (`ColumnTransformer`, `XGBClassifier`, `TreeExplainer`, `DeclarativeBase`,
    `TestClient`, ...). It is marked `slow` because importing the ML and dashboard stack
    takes several seconds.

### Verification record

Last run on the supported interpreter (Python 3.11.9, Windows) plus the Linux check. Reproduce
it with the commands above — do not take it on trust.

| Check | Windows 3.11.9 | Linux 3.11.17 (container) |
|-------|----------------|---------------------------|
| `ruff format --check .` | 30 files already formatted | 30 files already formatted |
| `ruff check .` | All checks passed | All checks passed |
| `mypy` (`strict = true`) | Success, 26 source files | Success, 26 source files |
| `pytest -q` | **34 passed** (4 third-party warnings), 67.5 s | **34 passed** (same warnings), 5.4 s |
| `python -m pip install -e ".[dev]"` | 130 packages, exit 0 | resolve + install + build succeeded |

Tool versions: ruff 0.17.1, mypy 2.4.0, pytest 9.1.1, pip 26.2.1. Linux was verified with
`python:3.11-slim` mounted at `/work`; the container script used `set -e` and ran to its final
completion marker, so every step in it succeeded.

Earlier, on the pre-3.11 scaffold: `git check-ignore` / `git add -A --dry-run` confirmed that
datasets, model artifacts, `.env`, `mlruns/` and `.venv/` are ignored while the `.gitkeep`
placeholders are tracked (git 2.46.2), and `docker compose config --quiet` exited 0
(Docker 29.8.2).

**Light path also verified:** in a separate venv containing only
`pydantic-settings packaging pytest ruff mypy`, `pytest -q -m "not slow"` gives
`9 passed, 25 deselected`. Deliberately *not* deselecting the slow tests in that environment
gives `22 failed` — the smoke test detects an unprovisioned environment instead of passing
vacuously.

**Known warnings (third-party, not failures):**

- `shap` emits three `PendingDeprecationWarning`s from `shap/plots/colors/_colors.py` at import
  time.
- `fastapi.testclient` warns: *"Using `httpx` with `starlette.testclient` is deprecated;
  install `httpx2` instead."* Tracked below as a compatibility item to resolve at M6.

### Not yet verifiable

Recorded honestly rather than implied to work:

| Check | Why it cannot be run yet |
|-------|--------------------------|
| `pytest -m integration` | No integration tests exist; they arrive with M2/M6. |
| `docker compose up` application stack | Only PostgreSQL is defined; app services arrive in M9. |
| CI being green | `.github/workflows/ci.yml` has never run on GitHub. |
| Anything about model quality | No model has been trained, and no performance numbers are claimed. |

---

## Design principles

- **No leakage.** Labels never reach an inference path; transforms are fitted on training data only.
- **One transform, two paths.** Training and serving share the fitted transformer (anti-skew by construction).
- **Imbalance-aware evaluation.** PR-AUC is primary; accuracy is never reported.
- **Three-way decisions.** `allow` / `review` / `block`, with thresholds treated as business
  policy that is stored alongside every prediction.
- **Explainability with limits.** SHAP per prediction, plus an explicit acknowledgement that
  `V1..V28` are PCA components and therefore not human-interpretable.
- **Config from the environment, only.** One settings object; no hardcoded credentials.
- **Evidence over assertion.** Every reported metric comes from a real logged run.

## Data, privacy and safety

- Only public/anonymised datasets or small synthetic fixtures. **No real cardholder data.**
- Account references are pseudo-anonymised; a PAN is never handled.
- The API is unauthenticated at this stage and must be bound to loopback or a private network.
- `.env`, datasets, model artifacts and database volumes are git-ignored.

## Documentation

| Document | Contents |
|----------|----------|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Components, data flow, data contracts, modelling and monitoring design, key decisions, open questions |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | Milestones M0–M9 with deliverables and exit criteria |
| [`AGENTS.md`](AGENTS.md) | Conventions and rules for future contributors (human or automated) |

## License

Not yet chosen. Until a license is added, all rights are reserved by the author.
