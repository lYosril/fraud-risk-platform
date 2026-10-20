# Fraud Detection & Risk Intelligence Platform

An end-to-end machine learning platform that scores credit-card transactions for fraud risk,
**explains every decision**, serves predictions over a REST API, persists the full prediction
history in PostgreSQL, and monitors the model for drift.

This is a portfolio project built to demonstrate production-shaped engineering: clear module
boundaries, reproducible experiments, explicit evaluation, and an honest record of what works
and what does not.

---

## ⚠️ Current status: foundation, data workflow, EDA and leakage-safe splitting complete; nothing trained or served

**Milestone M0 (repository foundation) is complete, the Python 3.11 development environment
is provisioned, the dataset workflow (acquisition, presence check and contract validation) is
implemented, and the exploratory analysis, split and preprocessing stages are implemented.
Nothing is trained, served or monitored yet.**

- There is **no trained model** in this repository, and **no model metrics are reported
  anywhere**. Any number that appears in a design document is illustrative of a *metric
  definition*, never a result.
- `docs/ARCHITECTURE.md` describes the intended system. It is a design, not a description of
  working code.
- Implemented and tested logic covers `fraud_risk.config.Settings` (configuration), the
  `fraud_risk.data` package: dataset presence and shape (`fraud_risk.data.dataset`), typed
  loading (`fraud_risk.data.ingestion`), the schema contract and summary report
  (`fraud_risk.data.validation`), the `python -m fraud_risk.data` command with its optional
  Kaggle CLI acquisition — `fraud_risk.features.eda` plus the `notebooks/eda.py` script that
  renders its report to the git-ignored `reports/eda/` — and the training-time entry points
  `fraud_risk.features.engineering` (the shared `ColumnTransformer`) and
  `fraud_risk.models.splitting` (temporal or stratified train/validation/test split plus a
  compact, value-free split record).
- All other modules are documented placeholders (see the status table below).
- The dependency set is installed and importable, and `tests/test_smoke_imports.py` checks
  that it stays that way — but no module that consumes those packages is implemented.

No transaction data is included. The reference dataset is never committed: see
[`data/README.md`](data/README.md) for the workflow and
[`docs/DATA_CARD.md`](docs/DATA_CARD.md) for provenance and licensing.

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
| 1 | Data ingestion & validation | `fraud_risk/data/` | Loading + contract validation **implemented**; fingerprint pending (M1) |
| 2 | EDA & feature engineering | `fraud_risk/features/` | EDA **implemented** (+ `notebooks/eda.py`); shared transformer **implemented** (not yet serialised) |
| 3 | Training & evaluation | `fraud_risk/models/splitting.py`, `training.py`, `evaluation.py` | Splitting **implemented**; training/evaluation scaffolded |
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
│   ├── README.md             # how to obtain the dataset and where to put it
│   ├── raw/                  # source dataset  (git-ignored)
│   ├── interim/              # intermediate data (git-ignored)
│   ├── processed/            # model-ready data (git-ignored)
│   └── sample/               # tiny synthetic fixtures (tracked)
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DATA_CARD.md          # provenance, licence, schema, class imbalance
│   └── ROADMAP.md
├── notebooks/                # exploration only; never imported by src/
│   └── eda.py                # reproducible EDA run: figures + summary under reports/
├── reports/                  # generated analysis output (git-ignored)
├── src/fraud_risk/
│   ├── __init__.py           # __version__
│   ├── config.py             # Settings — the only reader of environment variables
│   ├── data/                 # dataset.py (spec/presence), ingestion.py (typed load),
│   │                         # validation.py (contract + report), errors.py (typed errors)
│   ├── features/             # eda, engineering (the shared fitted transform)
│   ├── models/               # splitting (leakage-safe split), training, evaluation,
│   │                         # registry, inference
│   ├── db/                   # base, models, session (+ Alembic migrations in M2)
│   ├── api/                  # main (app factory), schemas
│   ├── dashboard/            # Streamlit app
│   └── monitoring/           # drift
└── tests/
    ├── test_config.py          # configuration contract
    ├── test_dataset.py         # dataset presence/shape check, acquisition guards
    ├── test_ingestion.py       # typed CSV loading and its failure modes
    ├── test_validation.py      # schema contract, report, CLI validation path
    ├── test_eda.py             # EDA summaries + end-to-end script run
    ├── test_engineering.py     # feature contract + training-only fitting
    ├── test_splitting.py       # partition, determinism, schema, metadata, CLI
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

No dataset is required to validate this repository or to run the test suite. The reference
dataset — the public ULB credit-card fraud data, `creditcard.csv` — is documented in
[`docs/DATA_CARD.md`](docs/DATA_CARD.md); obtaining it is optional (step 7).

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

### 7. (Optional) Obtain the dataset

Only needed once a model is trained (M3 onward); the test suite never requires it.

1. Download `creditcard.csv` from
   [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) in a browser, accepting
   the dataset terms if prompted. This repository does not scrape the page or bypass
   authentication.
2. Place it at **`data/raw/creditcard.csv`**.
3. Verify and validate what you have:

   ```bash
   python -m fraud_risk.data
   ```

   The command first checks presence and shape, then loads the file and enforces the data
   contract (columns, nulls, numeric values, `Class` only `0`/`1`), printing a row/feature/
   class summary. Exit code `0` means everything passed; `2` means the file is missing; `1`
   means a file exists but is damaged, has the wrong columns, or violates the contract (the
   offending column and reason are named).

`python -m fraud_risk.data --download` drives the **official** Kaggle CLI instead, but only
when it is installed and credentials are already configured — it never handles credentials
itself. Full instructions (including the synthetic fixture and all exit codes) are in
[`data/README.md`](data/README.md); provenance, licence and known limitations are in
[`docs/DATA_CARD.md`](docs/DATA_CARD.md).

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

# Dataset presence + contract validation. Exits 2 until you add
# data/raw/creditcard.csv (see step 7); the fixture check works immediately:
python -m fraud_risk.data
python -m fraud_risk.data --path data/sample/creditcard.synthetic.csv
```

### Exploratory analysis (optional, requires the dataset)

```bash
# Reference dataset (data/raw/creditcard.csv) -> reports/eda/ (git-ignored)
python notebooks/eda.py

# Any other file, including the tracked fixture, clearly labelled synthetic in the report
python notebooks/eda.py --path data/sample/creditcard.synthetic.csv --output-dir reports/eda-synthetic
```

The run writes `summary.md`, `summary.json`, four CSV tables and six PNG figures. It refuses
to start when the input is missing, and it labels the input as the reference dataset, the
synthetic fixture or a custom file - it never presents a fixture as a dataset result.
`fraud_risk.features.eda` computes every number; the script only renders it, which keeps
plotting libraries out of the package. **No transformation is fitted** at this stage, so
nothing computed here can leak into preprocessing.

### Splitting and preprocessing (requires the dataset, or use the fixture)

```bash
# Chronological 60/20/20 split (the default for time-ordered data) -> reports/splits/
python -m fraud_risk.models.splitting

# Class-proportional random split, for comparison only: do not report a final score from it
python -m fraud_risk.models.splitting --strategy stratified --output reports/splits/split_metadata.stratified.json

# The tracked fixture, clearly not the reference dataset
python -m fraud_risk.models.splitting --path data/sample/creditcard.synthetic.csv --seed 0 --output reports/splits/fixture.json
```

The command validates the file first, splits it, fits the shared `ColumnTransformer` on the
**training rows only**, and writes a ~2.5 KB JSON record: strategy, seed, fractions, sizes,
class counts, the feature schema, the excluded columns, and a SHA-256 digest of each split's
row positions. No feature value is written and no copy of the dataset is stored. `Time` orders
the temporal split but is not a model feature, `Amount` is left unscaled for tree models and
standardised on training statistics for linear ones, and no resampler runs before the split.
The reasoning is in `docs/ARCHITECTURE.md` section 6.

### What to look for

- `ruff format --check .` and `ruff check .` report no findings.
- `mypy` reports `Success: no issues found` (strict mode).
- `pytest -q` passes eight suites:
  - `tests/test_config.py` — the configuration contract (version parity, secret-free
    defaults, DSN construction and percent-encoding, threshold validation, caching).
  - `tests/test_dataset.py` — the dataset contract: the expected 31-column header,
    missing/unexpected columns, empty and header-only files, malformed-row detection, the
    guarantee that no cell value is echoed, CLI exit codes, and the acquisition guards (no
    CLI, no credentials, a failed download, and a reported success with no file on disk).
    It uses only synthetic files and never contacts Kaggle.
  - `tests/test_ingestion.py` — typed loading: missing/empty/non-UTF-8 files, duplicate
    headers that pandas would silently rename, ragged rows that pandas would silently shift,
    and the guarantee that no value is coerced on the way in.
  - `tests/test_validation.py` — the schema contract: duplicate columns, empty frames,
    missing columns, null counts per column, non-numeric/infinite/boolean columns, invalid
    labels, the summary report, the CLI failure path, and a subprocess check that importing
    the modules loads no data.
  - `tests/test_eda.py` — the exploratory stage: computed dimensions, class balance,
    duplicates, amount/time summaries, correlation ranking and the heavy-tail statistic; the
    typed validation gate on invalid input; a no-mutation guarantee; strict-JSON output; the
    synthetic and custom origin banners; and two end-to-end subprocess runs of
    `notebooks/eda.py` (fixture succeeds with six non-trivial PNGs, missing dataset exits 2).
  - `tests/test_engineering.py` — the shared transform: the declared feature schema, the label
    refused as a feature, non-numeric and missing values rejected, passthrough for tree models,
    and standardisation statistics proved to come from the training rows rather than the whole
    frame.
  - `tests/test_splitting.py` — the split contract: every row in exactly one subset, a
    chronological ordering, timestamp boundaries never divided, the stratified strategy's class
    rates, seed determinism, identical schemas, label separation, configuration validation, a
    value-free metadata record, and an end-to-end `python -m fraud_risk.models.splitting` run.
  - `tests/test_smoke_imports.py` — the environment smoke test: every package the
    architecture requires is *declared*, *installed inside its declared version range*, and
    *importable*, including the specific classes each layer is built on
    (`ColumnTransformer`, `XGBClassifier`, `TreeExplainer`, `DeclarativeBase`,
    `TestClient`, ...). It is marked `slow` because importing the ML and dashboard stack
    takes several seconds.

### Verification record

Last full check on the supported interpreter (Python 3.11.9, Windows). The Linux column is the
last `python:3.11-slim` run (Python 3.11.17) and **predates the EDA and splitting stages**:
Docker was not running when those changes were verified, so it was not re-run. Reproduce
everything with the commands above — do not take it on trust.

| Check | Windows 3.11.9 (current) | Linux 3.11.17 (previous state) |
|-------|--------------------------|-------------------------------|
| `ruff format --check .` | 42 files already formatted | 38 files already formatted |
| `ruff check .` | All checks passed | All checks passed |
| `mypy` (`strict = true`) | Success, 36 source files | Success, 32 source files |
| `pytest -q` | **212 passed** (4 third-party warnings) | **138 passed** (same warnings) |

Tool versions: ruff 0.17.1, mypy 2.4.0, pytest 9.1.1, pandas-stubs 2.3.3.260113, pip 26.2.1.
The container run used a clean venv and `pip install -e ".[dev]"`.

**Dataset check verified end to end:** `python -m fraud_risk.data --path
data/sample/creditcard.synthetic.csv` runs both stages — `OK`, 8 data rows, 31 columns, then
`Validation: OK` with 8 rows, 30 features, 0 missing values, 0 duplicate rows, `class 0 = 6`
and `class 1 = 2` — and exits 0. A CSV whose `Class` contains a `2` exits 1 with
`InvalidTargetError` naming the column, and the output never contains a `Validation: OK` line.
With no dataset present, `python -m fraud_risk.data` reports `MISSING` and exits 2. The real
dataset was **not** downloaded for this check, so nothing above is a claim about it.

**Reference dataset also checked locally (never committed):** with `data/raw/creditcard.csv`
present, `python -m fraud_risk.data` exited 0 with `Validation: OK`, and `python notebooks/eda.py`
regenerated `reports/eda/` from that file in this session. The run computed 284,807 rows,
30 features, 0 missing values, 1,081 duplicate rows and 492 fraudulent rows (99.8273% /
0.1727%). These are **dataset facts**, not model metrics, and they are not reproducible in CI,
which has no dataset. Nothing from `reports/` or `data/raw/` is committed.

**Split and preprocessing stage also run on the reference dataset** (never committed).
`python -m fraud_risk.models.splitting` (temporal, seed 42) produced 170,888 / 56,961 / 56,958
rows with 360 / 57 / 75 frauds (0.2107% / 0.1001% / 0.1317%): the class rate is **not**
constant across the two days, so a chronological split cannot preserve it. That is a property
of the data, recorded rather than smoothed away, and it is why the stratified comparison exists.
The same command with `--strategy stratified` produced 170,884 / 56,961 / 56,962 rows and
295 / 98 / 99 frauds (0.1726% / 0.1720% / 0.1738%). Both records are in the git-ignored
`reports/splits/` (2.5 KB each, no feature values). The shared preprocessor was fitted on the
training subset and transformed the validation subset to a 56,961 x 29 matrix.

Earlier, on the pre-3.11 scaffold: `git check-ignore` / `git add -A --dry-run` confirmed that
datasets, model artifacts, `.env`, `mlruns/` and `.venv/` are ignored while the `.gitkeep`
placeholders are tracked (git 2.46.2), and `docker compose config --quiet` exited 0
(Docker 29.8.2). `reports/` is ignored as well — `git check-ignore -v` matches
`.gitignore:91:reports/` for a generated report and figure, and a full EDA run adds nothing to
`git status`.

**Light path verified for the pre-splitting tree:** in a separate venv containing only
`pydantic-settings packaging pytest pandas` (all inside the declared ranges; no ML, API or
dashboard stack), `pytest -q -m "not slow"` gave `130 passed, 2 skipped, 25 deselected` — the
configuration and data suites need none of the ML stack, and the two EDA script tests skipped
because `matplotlib` is a dev-extra dependency. That run **predates** the engineering/splitting
tests, which import scikit-learn and were therefore not re-run there. Deliberately *not*
deselecting the slow tests in that environment gave `20 failed, 135 passed, 2 skipped`: the
smoke test detects an unprovisioned environment instead of passing vacuously.

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
