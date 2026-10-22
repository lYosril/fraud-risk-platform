# Fraud Detection & Risk Intelligence Platform

An end-to-end machine learning platform that scores credit-card transactions for fraud risk,
**explains every decision**, serves predictions over a REST API, persists the full prediction
history in PostgreSQL, and monitors the model for drift.

This is a portfolio project built to demonstrate production-shaped engineering: clear module
boundaries, reproducible experiments, explicit evaluation, and an honest record of what works
and what does not.

---

## âš ï¸ Current status: foundation, data workflow, EDA, splitting, model comparison, frozen final evaluation, SHAP explainability, reusable inference and the versioned FastAPI backend complete; single/batch scoring, prediction persistence, admin-protected history/review, dashboard and monitoring pending

**Milestone M0 (repository foundation) is complete, the Python 3.11 development environment
is provisioned, the dataset workflow (acquisition, presence check and contract validation) is
implemented, and the exploratory analysis, split, preprocessing, model-comparison, frozen
final-evaluation, SHAP explanation, reusable inference-service and HTTP API stages are
implemented. A winning model artifact, a frozen binary operating decision, an explainability
report and a versioned health/readiness/model-metadata/single-prediction/batch-prediction API
exist locally, and successful predictions are persisted to PostgreSQL with admin-protected
read-back and review routes. The dashboard and monitoring are not implemented yet.**

- **A real three-family comparison and a frozen final evaluation have been run** on the local
  reference dataset (logistic regression, random forest, XGBoost). Their actual metrics are
  written to the git-ignored `artifacts/comparison/` and `artifacts/final_evaluation/`
  directories and quoted in the verification record below. They describe those runs and that
  protocol only - they are not a promise about a future model. The training CLI now records one
  real run in a local SQLite-backed MLflow store (see the tracking section and the verification
  record); no model is registered or deployed by that step. The test split has now been scored
  more than once, so it is not a pristine holdout; future tuning must use validation or a fresh
  holdout.
- `docs/ARCHITECTURE.md` describes the intended system. It is a design, not a description of
  working code.
- Implemented and tested logic covers `fraud_risk.config.Settings` (configuration), the
  `fraud_risk.data` package: dataset presence and shape (`fraud_risk.data.dataset`), typed
  loading (`fraud_risk.data.ingestion`), the schema contract and summary report
  (`fraud_risk.data.validation`), the `python -m fraud_risk.data` command with its optional
  Kaggle CLI acquisition â€” `fraud_risk.features.eda` plus the `notebooks/eda.py` script that
  renders its report to the git-ignored `reports/eda/` â€” the training-time entry points
  `fraud_risk.features.engineering` (the shared `ColumnTransformer`),
  `fraud_risk.models.splitting` (temporal or stratified train/validation/test split plus a
  compact, value-free split record), `fraud_risk.models.training` (the calibrated
  logistic-regression / random-forest / XGBoost comparison, its local artifact bundle and
  provenance), `fraud_risk.models.evaluation` (per-class metrics, the majority-class baseline,
  the explicit probability/decision split and the validation-only expected-cost, minimum-recall
  and alert-capacity policies), `fraud_risk.models.final_evaluation` (freeze the model and
  threshold, then score test once), `fraud_risk.models.bundle` (the validated artifact-bundle
  contract shared by final evaluation and explanations), `fraud_risk.models.explainability`
  (SHAP TreeExplainer on the selected calibrated XGBoost, global mean-absolute importance and
  per-transaction top-k contributions, rendered by `notebooks/explainability.py`) and
  `fraud_risk.tracking` (the optional MLflow boundary: configurable tracking URI, deterministic
  run fingerprint, parameter/tag/metric/artifact logging and explicit failed-run status; model
  registration is a separate reviewed step and is not performed here).
- All other modules are documented placeholders (see the status table below).
- The dependency set is installed and importable, and `tests/test_smoke_imports.py` checks
  that it stays that way.

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
        â”‚
        â–¼
  data ingestion & validation â”€â”€â–º features (shared fitted transform) â”€â”€â–º model training
                                                                              â”‚  MLflow
                                                                              â–¼
  client â”€â”€HTTPâ”€â”€â–º FastAPI â”€â”€â–º inference (+ SHAP) â”€â”€â–º PostgreSQL â—„â”€â”€ Streamlit dashboard
                                                        â–²
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
| 2 | EDA & feature engineering | `fraud_risk/features/` | EDA **implemented** (+ `notebooks/eda.py`); shared transformer **implemented** and serialised inside the model bundle |
| 3 | Training & evaluation | `fraud_risk/models/` (`splitting.py`, `training.py`, `evaluation.py`, `final_evaluation.py`, `explainability.py`) + `fraud_risk/tracking.py` | Splitting, three-family comparison, per-class metrics, cost/min-recall/alert-capacity policies, frozen final evaluation, XGBoost SHAP explanations and MLflow run tracking **implemented**; stratified CV and calibration comparison pending (M4) |
| 4 | Artifacts & inference | `fraud_risk/models/registry.py`, `inference.py` | Reusable inference service **implemented** (lazy validated load, schema/order checks, score, optional SHAP); registry aliases **scaffolded** (M5) |
| 5 | FastAPI application | `fraud_risk/api/` | Versioned health, readiness, model-metadata, single-prediction, batch-prediction, history and review routes **implemented** with Pydantic contracts, per-item batch errors, request/batch limits, CORS configuration, safe error mapping, admin bearer-token auth and access logging; dashboard and monitoring **pending** (M6) |
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
â”œâ”€â”€ .env.example              # placeholder configuration only (commit this, not .env)
â”œâ”€â”€ .gitattributes            # line-ending normalisation
â”œâ”€â”€ .gitignore
â”œâ”€â”€ .github/workflows/ci.yml  # lint + type-check + tests (not executed yet)
â”œâ”€â”€ .python-version           # 3.11
â”œâ”€â”€ AGENTS.md                 # conventions for future work in this repo
â”œâ”€â”€ docker-compose.yml        # PostgreSQL only (app services arrive in M9)
â”œâ”€â”€ pyproject.toml            # the single dependency + tooling definition
â”œâ”€â”€ README.md
â”œâ”€â”€ artifacts/                # model artifact store (git-ignored, .gitkeep only)
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ README.md             # how to obtain the dataset and where to put it
â”‚   â”œâ”€â”€ raw/                  # source dataset  (git-ignored)
â”‚   â”œâ”€â”€ interim/              # intermediate data (git-ignored)
â”‚   â”œâ”€â”€ processed/            # model-ready data (git-ignored)
â”‚   â””â”€â”€ sample/               # tiny synthetic fixtures (tracked)
â”œâ”€â”€ docs/
â”‚   â”œâ”€â”€ ARCHITECTURE.md
â”‚   â”œâ”€â”€ DATA_CARD.md          # provenance, licence, schema, class imbalance
â”‚   â””â”€â”€ ROADMAP.md
â”œâ”€â”€ notebooks/                # exploration only; never imported by src/
â”‚   â””â”€â”€ eda.py                # reproducible EDA run: figures + summary under reports/
â”œâ”€â”€ reports/                  # generated analysis output (git-ignored)
â”œâ”€â”€ src/fraud_risk/
â”‚   â”œâ”€â”€ __init__.py           # __version__
â”‚   â”œâ”€â”€ config.py             # Settings â€” the only reader of environment variables
â”‚   â”œâ”€â”€ data/                 # dataset.py (spec/presence), ingestion.py (typed load),
â”‚   â”‚                         # validation.py (contract + report), errors.py (typed errors)
â”‚   â”œâ”€â”€ features/             # eda, engineering (the shared fitted transform)
â”‚   â”œâ”€â”€ models/               # splitting (leakage-safe split), training, evaluation,
â”‚   â”‚                         # final_evaluation, registry, inference
â”‚   â”œâ”€â”€ db/                   # base, models, session (+ Alembic migrations in M2)
â”‚   â”œâ”€â”€ api/                  # main (app factory), routes, dependencies, schemas
â”‚   â”œâ”€â”€ dashboard/            # Streamlit app
â”‚   â””â”€â”€ monitoring/           # drift
â””â”€â”€ tests/
    â”œâ”€â”€ test_config.py          # configuration contract
    â”œâ”€â”€ test_dataset.py         # dataset presence/shape check, acquisition guards
    â”œâ”€â”€ test_ingestion.py       # typed CSV loading and its failure modes
    â”œâ”€â”€ test_validation.py      # schema contract, report, CLI validation path
    â”œâ”€â”€ test_eda.py             # EDA summaries + end-to-end script run
    â”œâ”€â”€ test_engineering.py     # feature contract + training-only fitting
    â”œâ”€â”€ test_splitting.py       # partition, determinism, schema, metadata, CLI
    â”œâ”€â”€ test_evaluation.py      # metric definitions, per-class, majority baseline, cost threshold
    â”œâ”€â”€ test_training.py        # three-family comparison, artifacts, provenance, test isolation
    â”œâ”€â”€ test_final_evaluation.py # frozen decision, one test scoring, policy/CLI checks
    â”œâ”€â”€ test_explainability.py   # SHAP schema, additivity, reports
    â”œâ”€â”€ test_inference.py         # bundle loading, input order, failures, API schemas
    â”œâ”€â”€ test_api.py               # HTTPX ASGI tests: routes, auth, readiness, CORS, docs
    â”œâ”€â”€ test_tracking.py          # MLflow config, redaction, run fingerprint
    â””â”€â”€ test_smoke_imports.py     # environment smoke test (declared / installed / importable)
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
dataset â€” the public ULB credit-card fraud data, `creditcard.csv` â€” is documented in
[`docs/DATA_CARD.md`](docs/DATA_CARD.md); obtaining it is optional (step 7).

### Getting Python 3.11

**Windows** â€” install from [python.org](https://www.python.org/downloads/) (tick *Add
python.exe to PATH*) or from the Microsoft Store, then confirm the launcher sees it:

```powershell
py --list-paths          # should list -V:3.11
```

**Debian / Ubuntu** â€” `venv` is a separate package, and this is the single most common
Linux setup failure:

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv
```

**Fedora / RHEL** â€” `sudo dnf install python3.11`.
**macOS** â€” `brew install python@3.11`.
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
that first â€” installing into 3.12+ produces an environment that resolves different package
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
than pinned to "whatever is newest" â€” see *Dependency layout* in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) Â§14. The resolver legitimately picks
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

Not required for the unit tests â€” the database is first used in milestone M2.

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
when it is installed and credentials are already configured â€” it never handles credentials
itself. Full instructions (including the synthetic fixture and all exit codes) are in
[`data/README.md`](data/README.md); provenance, licence and known limitations are in
[`docs/DATA_CARD.md`](docs/DATA_CARD.md).

---

## Validation commands

Run these to verify the initial repository.

### Fast path â€” no third-party ML dependencies required

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

### Full path â€” the real dependency set

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

### Training and comparing the models (requires the dataset)

```bash
# All three families, temporal split, validation selection, one test evaluation
python -m fraud_risk.models.training

# A single family, with an explicit artifact directory
python -m fraud_risk.models.training --families logistic_regression \
    --output-dir artifacts/comparison-logistic
```

The tracked 8-row `data/sample/creditcard.synthetic.csv` is a schema/CLI fixture, not a
training set: its temporal validation split contains no fraud, so the CLI exits 1 by design.
The test suite covers a full three-family comparison and the frozen final evaluation on
generated 300-row fixtures instead; it never reads the real dataset.

The protocol is identical for every family:

- fit on `train` only, with a sigmoid calibrator fitted on internal training folds;
- handle imbalance by training-only reweighting - `class_weight` for logistic regression and
  random forest, `scale_pos_weight` from the training labels for XGBoost - with **no
  resampling anywhere**;
- choose each candidate's threshold on `validation` to minimise expected cost under
  `FALSE_NEGATIVE_COST` / `FALSE_POSITIVE_COST` (default 100:1 cost units, a provisional
  business assumption, not something the model tunes);
- select the candidate with the lowest validation expected cost; ties break on validation
  PR-AUC, and accuracy never participates;
- freeze the winner and its threshold, then score `test` **once**.

The grid is bounded: four logistic settings, two random forests (200 trees, depth 12) and two
XGBoost settings (300 rounds, depth 6), each with three-fold calibration. `--n-jobs` controls
tree parallelism and every candidate's fit time is recorded. On the reference machine the full
comparison took 114 s.

`artifacts/comparison/` (git-ignored) receives:

| File | Contents |
|------|----------|
| `model.joblib` | the winning calibrated pipeline, exact threshold, feature columns, imbalance method and cost policy |
| `comparison.json` | every candidate's parameters, imbalance setting, fit time, validation PR-AUC, cost-optimal threshold, expected cost and metrics |
| `metrics.json` | validation and test at the frozen threshold: precision, recall, F1, confusion matrix, PR-AUC, ROC-AUC, FPR, per-class metrics and the cost impact of false negatives versus false positives, next to the majority-class baseline |
| `run.json` | seed, feature schema, split record, grids, computational limits, imbalance derivation, library versions, git revision, timestamp, `run_fingerprint` and the MLflow run identity (or an explicit disabled record) |

Accuracy appears only inside `accuracy_comparison`, next to the majority-class baseline. On
the reference run the selected model is *less* accurate than "never fraud" (99.7876% versus
99.8683%) while finding 80% of the frauds - which is exactly why it is not the criterion.

### Tracking training runs locally with MLflow

Tracking is enabled by default in the training CLI. The default local store is SQLite
(`sqlite:///mlflow.db`), so no tracking server is required:

```bash
# Default: SQLite store at ./mlflow.db, experiment "fraud-detection"
python -m fraud_risk.models.training

# Inspect the runs in the MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db

# A different experiment or a shared server
python -m fraud_risk.models.training --experiment-name fraud-staging \
    --tracking-uri http://localhost:5000

# Offline run: write the local artifact bundle but skip MLflow entirely
python -m fraud_risk.models.training --no-mlflow
```

`MLFLOW_TRACKING_URI` and `MLFLOW_EXPERIMENT_NAME` in `.env` are the persistent equivalents of
`--tracking-uri` and `--experiment-name`. The run records the training configuration, split
strategy/seed/fractions/position digests, feature schema digest, model family, validation and
test metrics, and the four artifact files - never dataset rows, feature snapshots or
credentials. `run_fingerprint` (a SHA-256 over the seed, split record, schema, configuration and
git revision) is written into `run.json`, attached as an MLflow tag and used in the run name; it
is the reproducible identifier, while MLflow still assigns its own `run_id`. A run that raises
is marked `FAILED` with a failure-type/message tag before the original exception is re-raised.
The trainer does **not** call `mlflow.log_model` or register a model: publication and promotion
remain a separate reviewed step in M5.

MLflow 3.x keeps the plain `./mlruns` filesystem backend in maintenance mode. If you need it,
set `MLFLOW_ALLOW_FILE_STORE=true` in your own environment; the default here is SQLite.

### Freezing the operating threshold and final test evaluation (requires the dataset)

`fraud_risk.models.final_evaluation` does not retrain or re-select the model. It loads
`artifacts/comparison/model.joblib`, recreates the recorded split, selects the threshold on
**validation only**, copies the model and writes the decision **before** reading test labels,
then scores `test` once:

```bash
# Reference policy: expected cost using the bundle's recorded 100:1 weights
python -m fraud_risk.models.final_evaluation

# Alternative operating policies; the model stays frozen, only the cut changes
python -m fraud_risk.models.final_evaluation --policy min-recall --min-recall 0.80
python -m fraud_risk.models.final_evaluation --policy alert-capacity --max-alert-rate 0.005
python -m fraud_risk.models.final_evaluation --policy alert-capacity --max-alerts 300
```

The default split and feature schema come from `run.json` next to the bundle. If that record
is absent, pass both `--strategy` and `--seed`; default 60/20/20 fractions and the bundle's
feature columns are used. `--policy min-recall` maximises validation precision subject to the
target, then prefers fewer alerts. `--policy alert-capacity` maximises validation recall within
the capacity, then prefers higher precision and fewer alerts. Tied scores are never split, so an
alert count can be below capacity. The test volume is reported, never used to re-tune the
capacity.

`artifacts/final_evaluation/` (git-ignored) receives:

| File | Contents |
|------|----------|
| `decision.json` | the frozen decision, written before test: policy, exact threshold, validation metrics and alert count, model SHA-256, cost weights and validation split digest |
| `frozen_model.joblib` | byte-identical copy of the bundle that was evaluated |
| `metrics.json` | validation and test with separate probability summaries and binary decision summaries: precision, recall, F1, PR-AUC, ROC-AUC, FPR, confusion matrix, alert count/rate, per-class metrics, cost impact and majority baseline |
| `run.json` | full split record, policy, model hash, cost weights, library versions, git revision, timestamp and an explicit statement that test was scored after the freeze |

The model was selected under validation expected cost; a min-recall or capacity policy changes
the operating cut but does not re-optimise the model. No threshold is universally optimal.
The comparison stage already scored this test split, so the final report is a **repeat**
evaluation, not an independent holdout; further tuning needs validation or a fresh split.

### Explaining the frozen model with SHAP

`fraud_risk.models.explainability` loads the bundle frozen by the final-evaluation stage and
builds one `shap.TreeExplainer` per calibration fold. It writes global mean-absolute SHAP
importance and a top-k explanation for one transaction; the plotting entry point lives
outside the package because plotting imports are forbidden in `src/`:

```bash
# Default: artifacts/final_evaluation/frozen_model.joblib, validation split, row 0
python notebooks/explainability.py

# Explicit bundle, sample cap and row
python notebooks/explainability.py \
  --model artifacts/final_evaluation/frozen_model.joblib \
  --path data/raw/creditcard.csv \
  --max-rows 2000 --row 0 --top-k 10
```

`reports/explainability/` (git-ignored) receives `summary.md`, strict `summary.json`,
`global_importance.csv/json`, `transaction_explanation.json`, `provenance.json` and two PNG
figures. The report states the limits before the numbers.

**What the explanation is:** an attribution to the fitted XGBoost tree margin **before** the
sigmoid probability and the outer calibration. It is additive within that space
(`base_value + contributions = tree_margin`), and the module checks that reconstruction. It
is **not** additive in the final calibrated probability, not a probability-point change, and
not the decision threshold. `raw_value` is the submitted value; `model_value` is the exact
array after the bundle's fitted preprocessor (equal for the selected passthrough tree model).
Positive values push the tree margin toward fraud; negative values push it away.

**What it is not:** causal evidence. `V1..V28` are anonymous PCA components with no
human-readable business meaning, correlated features share credit, and path-dependent Tree
SHAP uses the fitted trees' cover rather than an interventional background sample. Global
importance is an average absolute contribution, not model quality or a promotion gate.

The implementation deliberately supports the selected **XGBoost** family only. A
random-forest or logistic bundle raises a clear error instead of silently substituting a
different explainer or output space. No probability or decision threshold is included in the
explanation records.

### Scoring one transaction with the reusable inference service

`fraud_risk.models.inference` loads the frozen bundle, checks that its fitted preprocessor
matches the declared feature order, and scores one transaction through that exact pipeline. It
never fits, calibrates or re-trains anything during a request. The default artifact is
`artifacts/final_evaluation/frozen_model.joblib`, generated from `artifacts/comparison/model.joblib`
by the training and final-evaluation commands:

```bash
python -m fraud_risk.models.training
python -m fraud_risk.models.final_evaluation
```

Override the location with `MODEL_ARTIFACT_PATH` or by passing a path to the service:

```python
from fraud_risk.models.inference import FraudInferenceService, InferenceRequest

service = FraudInferenceService()  # reads Settings.model_artifact_path
features = {"V1": -1.2, "V2": 0.4}  # ... V1..V28 in the bundle's declared order
result = service.predict(
    InferenceRequest(transaction_id="txn-123", features=features, amount=42.50),
    explain=True,
)
print(result.probability, result.threshold, result.prediction, result.model_version)
```

`features` must be supplied in the exact order declared by the artifact (`V1..V28` for the
reference model). `Amount` is passed as the separate top-level `amount` field; alternatively,
omit `amount` and include `Amount` in `features` as the last key. Reordered, missing, extra,
non-finite or negative inputs raise `InferenceInputError` rather than being silently realigned.
A missing, corrupt or schema-incompatible artifact raises `InferenceModelError`. The service is
lazy and caches the loaded pipeline, so repeated requests reuse the same fitted object.

`model_version` is the bundle's `run_fingerprint` when present; older bundles without a
fingerprint fall back to the artifact's SHA-256. The response also carries `model_sha256` and
`model_family`. `explanation` is optional and is only produced when requested; for the selected
XGBoost bundle it is the SHAP top-k explanation described above, while an unsupported family
fails loudly instead of returning a different explanation. The binary `prediction` is
`probability >= threshold` using the threshold stored in the bundle.

The FastAPI-ready request and response models live in `fraud_risk.api.schemas`:

```python
from fraud_risk.api.schemas import PredictionRequest, PredictionResponse

request = PredictionRequest(
    transaction_id="txn-123",
    amount=42.50,
    features={"V1": -1.2},  # illustrative; the real contract requires the full schema
    explain=True,
)
result = service.predict(request.to_inference_request(), explain=request.explain)
payload = PredictionResponse.from_result(result).model_dump()
```

The request schema rejects unknown fields, label fields (`is_fraud` / `Class`), a duplicated
`Amount` inside `features`, negative amounts and non-finite values. No feature snapshot or raw
request payload is written to the response.

### Running the FastAPI backend

Start the local server from the repository root:

```bash
python -m uvicorn fraud_risk.api.main:app --host 127.0.0.1 --port 8000
```

The interactive OpenAPI documentation is at `http://127.0.0.1:8000/docs`; the raw schema is at
`http://127.0.0.1:8000/openapi.json`. Both are disabled when `APP_ENV=production`, along with
ReDoc and debug mode.

| Endpoint | Auth | Purpose |
|----------|------|---------|
| `GET /api/v1/health` | public | Process liveness only; does not claim the model is loadable |
| `GET /api/v1/ready` | public | Loads and validates the configured model; `200` when ready, `503` otherwise |
| `GET /api/v1/model` | admin | Model name, version, SHA-256, family, threshold and feature schema |
| `POST /api/v1/predict` | public | Scores one transaction through `FraudInferenceService` and persists the result |
| `POST /api/v1/predict/batch` | public | Scores a bounded list, returning one result or safe error per item |
| `GET /api/v1/predictions/{id}` | admin | Reads one persisted prediction (never the feature snapshot) |
| `PATCH /api/v1/predictions/{id}/review` | admin | Updates the analyst review status and returns the updated record |

Admin routes require `Authorization: Bearer <API_ADMIN_TOKEN>`. The token must be at least 16
characters when set. If it is unset, those routes return `503` rather than becoming anonymous.
A missing or wrong token returns `401` with `WWW-Authenticate: Bearer`.

Both prediction routes return the model version, risk probability, applied threshold, the raw
binary `prediction` and its human-readable `decision` (`legitimate` or `fraud`). The route
handlers are asynchronous; model loading, preprocessing and `predict_proba` run in Starlette's
thread pool so the event loop is not blocked.

Readiness is deliberately separate from health: a running process with a missing, corrupt or
incompatible artifact reports `503` with `model_available: false` instead of pretending to be
ready. Readiness exposes availability only, not model identity or the feature schema. Model
failures are logged server-side and returned to clients as a generic
`{"detail": "model unavailable"}` so filesystem paths and internals are not exposed.
Request-validation failures are normalized to the same `{"detail": "<field>: <reason>"}` shape
and never echo the submitted feature values. Unexpected failures return
`500 {"detail": "internal server error"}` without a traceback. Repository calls run in the
thread pool, writes commit once per unit of work, and database failures roll back to a safe
`503`. Access logs contain method, path, status and duration only: no query strings, headers,
tokens or request bodies.

### Production security settings and known limitations

Set `API_ADMIN_TOKEN` from a secret manager (never from a committed file) to enable model
metadata, history and review. Set `CORS_ALLOWED_ORIGINS` to the exact dashboard origin;
`Authorization` and `PATCH` are allowed for the admin routes. Keep `API_HOST` on a private
interface and terminate TLS at a reverse proxy or load balancer — the application does not
terminate TLS itself. Disable interactive docs in production (`APP_ENV=production` does this
automatically). `PERSIST_FEATURE_SNAPSHOT` remains off by default; enabling it stores raw model
inputs and requires an explicit storage/privacy review.

Known limitations, stated plainly:

- Scoring remains public and returns model version/SHA/family/threshold in each prediction.
  Protect or redact scoring too if model identity is sensitive in your environment.
- The admin surface uses one shared token. There is no per-user identity, role model, token
  rotation, revocation list or rate limiting. Access logs are not an audit trail of who
  reviewed what.
- The API does not itself provide HTTPS, a WAF, request authentication for scoring, or
  protection against a compromised host. Those belong to the deployment platform.
- Tests passing means the documented behaviours are covered; it does not prove the deployment
  is secure. Review the infrastructure, secret injection, network boundary and database
  permissions separately.

**Current scope:** successful predictions are persisted, and `GET /api/v1/predictions/{id}` plus
`PATCH /api/v1/predictions/{id}/review` provide admin-protected read-back and review. Feedback
labels, the dashboard and monitoring remain later milestones.

### Request and response examples

**These examples are illustrative.** They use synthetic transaction ids and the public ULB
credit-card benchmark's anonymised `V1..V28` feature contract; they are **not live banking
transactions**, and the numeric outputs below are placeholders rather than a real evaluation.
The feature list is abbreviated for readability: a real request must contain every feature
declared by the loaded bundle, in that bundle's exact order.

Single transaction request:

```json
{
  "transaction_id": "synthetic-txn-0001",
  "amount": 42.50,
  "features": {"V1": -1.2, "V2": 0.4},
  "explain": false
}
```

Illustrative single response:

```json
{
  "transaction_id": "synthetic-txn-0001",
  "probability": 0.0817,
  "threshold": 0.0042,
  "prediction": 1,
  "decision": "fraud",
  "model_version": "<model-version>",
  "model_sha256": "<sha256>",
  "model_family": "xgboost",
  "explanation": null
}
```

Batch request, including one deliberately invalid item:

```json
{
  "items": [
    {
      "transaction_id": "synthetic-txn-0001",
      "amount": 42.50,
      "features": {"V1": -1.2, "V2": 0.4}
    },
    {
      "transaction_id": "synthetic-txn-0002",
      "amount": -1.0,
      "features": {"V1": 0.0, "V2": 0.0}
    }
  ]
}
```

Illustrative batch response. The valid item is still scored; the invalid item is reported at
the same index and does not corrupt the batch:

```json
{
  "model_version": "<model-version>",
  "model_sha256": "<sha256>",
  "model_family": "xgboost",
  "threshold": 0.0042,
  "total": 2,
  "succeeded": 1,
  "failed": 1,
  "results": [
    {
      "index": 0,
      "transaction_id": "synthetic-txn-0001",
      "status": "ok",
      "prediction": {
        "transaction_id": "synthetic-txn-0001",
        "probability": 0.0817,
        "threshold": 0.0042,
        "prediction": 1,
        "decision": "fraud",
        "model_version": "<model-version>",
        "model_sha256": "<sha256>",
        "model_family": "xgboost",
        "explanation": null
      },
      "error": null
    },
    {
      "index": 1,
      "transaction_id": "synthetic-txn-0002",
      "status": "error",
      "prediction": null,
      "error": "amount: Input should be greater than or equal to 0"
    }
  ]
}
```

Batch and request-size limits are configuration, not hardcoded route constants:

- `BATCH_MAX_SIZE` defaults to `100`; a larger batch returns `413`.
- `API_MAX_REQUEST_BYTES` defaults to `1_048_576`; the limit is enforced before JSON parsing and
  an oversized body returns `413 {"detail": "request body too large"}`.
- An empty `items` list or a non-object item returns `422`.
- A missing/corrupt model returns `503` for the whole batch, never a misleading per-item error.

CORS is configured through `CORS_ALLOWED_ORIGINS` as a comma-separated list of exact origins
(for example `http://localhost:8501`). The default is empty, credentials are never allowed, and a
wildcard is rejected outside `local`/`test`. Keep public scoring bound to loopback or a private
network; the admin routes are separately protected by `API_ADMIN_TOKEN`.

### What to look for

- `ruff format --check .` and `ruff check .` report no findings.
- `mypy` reports `Success: no issues found` (strict mode).
- `pytest -q` passes seventeen suites:
  - `tests/test_config.py` â€” the configuration contract (version parity, secret-free
    defaults, DSN construction and percent-encoding, threshold validation, caching).
  - `tests/test_dataset.py` â€” the dataset contract: the expected 31-column header,
    missing/unexpected columns, empty and header-only files, malformed-row detection, the
    guarantee that no cell value is echoed, CLI exit codes, and the acquisition guards (no
    CLI, no credentials, a failed download, and a reported success with no file on disk).
    It uses only synthetic files and never contacts Kaggle.
  - `tests/test_ingestion.py` â€” typed loading: missing/empty/non-UTF-8 files, duplicate
    headers that pandas would silently rename, ragged rows that pandas would silently shift,
    and the guarantee that no value is coerced on the way in.
  - `tests/test_validation.py` â€” the schema contract: duplicate columns, empty frames,
    missing columns, null counts per column, non-numeric/infinite/boolean columns, invalid
    labels, the summary report, the CLI failure path, and a subprocess check that importing
    the modules loads no data.
  - `tests/test_eda.py` â€” the exploratory stage: computed dimensions, class balance,
    duplicates, amount/time summaries, correlation ranking and the heavy-tail statistic; the
    typed validation gate on invalid input; a no-mutation guarantee; strict-JSON output; the
    synthetic and custom origin banners; and two end-to-end subprocess runs of
    `notebooks/eda.py` (fixture succeeds with six non-trivial PNGs, missing dataset exits 2).
  - `tests/test_engineering.py` â€” the shared transform: the declared feature schema, the label
    refused as a feature, non-numeric and missing values rejected, passthrough for tree models,
    and standardisation statistics proved to come from the training rows rather than the whole
    frame.
  - `tests/test_splitting.py` â€” the split contract: every row in exactly one subset, a
    chronological ordering, timestamp boundaries never divided, the stratified strategy's class
    rates, seed determinism, identical schemas, label separation, configuration validation, a
    value-free metadata record, and an end-to-end `python -m fraud_risk.models.splitting` run.
  - `tests/test_evaluation.py` â€” the metric definitions: hand-checked precision/recall/F1,
    confusion counts, per-class metrics, the FPR denominator, threshold-free AUCs, the
    majority-class trap, cost weights, alert counts, probability/decision separation, the
    cost-optimal threshold rule (including its higher-recall tie-break and the "approve
    everything" case), minimum-recall and alert-capacity policies, tied-score handling, and
    typed rejection of malformed inputs.
  - `tests/test_final_evaluation.py` â€” the frozen stage: the model and decision are written
    before test, test is scored exactly once, corrupting test labels/features cannot change the
    threshold or validation metrics, the frozen copy is byte-identical, policy/CLI validation,
    and the metric/probability/decision records stay consistent.
  - `tests/test_training.py` â€” the comparison: all three families fitted, selection by
    validation expected cost rather than accuracy, a reloadable `model.joblib`, every required
    metric plus per-class and cost-impact records, threshold freezing, reproducibility,
    training-derived imbalance weights, computational limits, a deliberately corrupted test
    split proving it never influences fitting or selection, CLI success/failure paths, and the
    MLflow integration: a real SQLite-backed run with parameters, tags, metrics and artifacts,
    an explicitly disabled-tracking record, and a failed fit marked `FAILED` with failure tags.
  - `tests/test_tracking.py` — the tracking boundary: configuration from settings, blank-value
    rejection, credential/query redaction, and the deterministic, input-sensitive run
    fingerprint.
  - `tests/test_explainability.py` — the SHAP layer: the selected XGBoost bundle loads,
    unsupported families fail loudly, top-k contributions are ranked and additive in the
    explained tree-margin space, displayed values equal the exact preprocessor inputs (also
    for a scaled fixture), global importance is sorted and row-free, the report writer emits
    strict JSON/CSV/Markdown, probability/threshold fields are absent from explanations, and
    an end-to-end subprocess run writes two non-trivial PNGs from a synthetic fixture.
  - `tests/test_inference.py` — the reusable serving path: a fitted bundle is loaded and
    cached, scoring never calls `fit`, probability/threshold/decision/model-version fields stay
    separate, reordered/missing/extra/non-finite/negative inputs are rejected, missing/corrupt/
    incompatible artifacts fail with a model error, an optional XGBoost explanation is
    returned, and the FastAPI-ready request/response schemas reject labels and out-of-range
    values without exposing a feature snapshot.
  - `tests/test_api.py` — the versioned HTTP contract: health, availability-only readiness
    (including a missing artifact), admin bearer-token auth and fail-closed behaviour, model
    metadata, dependency-overridden single and batch prediction, persisted history without the
    feature snapshot, review updates, per-item batch errors, empty/oversized batches,
    request-body limits, validation and safe error mapping including a generic 500, CORS
    allow/deny and admin preflight, production docs removal, factory laziness and a subprocess
    proof that importing the app opens no network connection.
  - `tests/test_db.py` — the database contract: ORM shape and privacy defaults, repository
    commit/rollback policy for create/read/review, request-scoped session rollback/close, and
    offline Alembic rendering. `tests/test_db_integration.py` performs the real PostgreSQL
    round-trip and schema check when `TEST_DATABASE_URL` is set.
  - `tests/test_smoke_imports.py` â€” the environment smoke test: every package the
    architecture requires is *declared*, *installed inside its declared version range*, and
    *importable*, including the specific classes each layer is built on
    (`ColumnTransformer`, `XGBClassifier`, `TreeExplainer`, `DeclarativeBase`,
    `ASGITransport`, ...). It is marked `slow` because importing the ML and dashboard stack
    takes several seconds.

### Verification record

Last full check on the supported interpreter (Python 3.11.9, Windows), with the Compose
PostgreSQL service reachable through `TEST_DATABASE_URL`. The Linux column is the last
`python:3.11-slim` run (Python 3.11.17) and **predates the EDA, splitting, modelling and API
stages**: Docker was not running when those changes were verified, so it was not re-run.
Reproduce everything with the commands above.

| Check | Windows 3.11.9 (current) | Linux 3.11.17 (previous state) |
|-------|--------------------------|-------------------------------|
| `ruff format --check .` | 62 files already formatted | 38 files already formatted |
| `ruff check .` | All checks passed | All checks passed |
| `mypy` (`strict = true`) | Success, 54 source files | Success, 32 source files |
| `pytest -q` | **385 passed** (4 third-party warnings, 58.35 s) | **138 passed** (same warnings) |

The `pytest -q` row includes the two PostgreSQL integration tests; without `TEST_DATABASE_URL`
they are skipped. The suite was run against this worktree; it is not a claim that a deployment
or its infrastructure is secure.

Tool versions: ruff 0.17.1, mypy 2.4.0, pytest 9.1.1, pandas-stubs 2.3.3.260113, pip 26.2.1.
The container run used a clean venv and `pip install -e ".[dev]"`.

**Dataset check verified end to end:** `python -m fraud_risk.data --path
data/sample/creditcard.synthetic.csv` runs both stages â€” `OK`, 8 data rows, 31 columns, then
`Validation: OK` with 8 rows, 30 features, 0 missing values, 0 duplicate rows, `class 0 = 6`
and `class 1 = 2` â€” and exits 0. A CSV whose `Class` contains a `2` exits 1 with
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

**Three-family comparison also run on the reference dataset** (never committed).
`python -m fraud_risk.models.training` (temporal split, seed 42, 114 s wall clock on all
cores) fitted eight candidates and selected on validation expected cost with the provisional
100:1 policy. Every column except `Fit` is a validation metric:

| Candidate (sorted by validation cost) | PR-AUC | Cost units | Threshold | Precision | Recall | F1 | FPR | Fit |
|---------------------------------------|--------|-----------|-----------|-----------|--------|----|-----|-----|
| `xgboost:scale_pos_weight=473.6889` | 0.7931 | **1,016** | 0.000389 | 0.2927 | 0.8421 | 0.4344 | 0.002039 | 6.9 s |
| `random_forest:class_weight=balanced_subsample` | 0.7628 | 1,185 | 0.001836 | 0.3511 | 0.8070 | 0.4894 | 0.001494 | 34.5 s |
| `random_forest:class_weight=None` | 0.7559 | 1,197 | 0.000824 | 0.3217 | 0.8070 | 0.4600 | 0.001705 | 51.6 s |
| `logistic_regression:class_weight=balanced,C=1.0` | 0.7419 | 1,208 | 0.010008 | 0.1843 | 0.8246 | 0.3013 | 0.003655 | 1.8 s |
| `xgboost:scale_pos_weight=1.0` | 0.7737 | 1,210 | 0.000444 | 0.2949 | 0.8070 | 0.4319 | 0.001933 | 8.5 s |
| `logistic_regression:class_weight=None,C=1.0` | 0.5175 | 1,211 | 0.023099 | 0.1822 | 0.8246 | 0.2984 | 0.003708 | 1.0 s |
| `logistic_regression:class_weight=balanced,C=0.1` | 0.7334 | 1,221 | 0.011141 | 0.1754 | 0.8246 | 0.2892 | 0.003884 | 1.5 s |
| `logistic_regression:class_weight=None,C=0.1` | 0.5501 | 1,230 | 0.024187 | 0.1697 | 0.8246 | 0.2814 | 0.004042 | 1.0 s |

PR-AUC and expected cost do not rank identically: the unweighted logistic model has the
weakest ranking (AP 0.5175, still far above the 0.001 prevalence baseline) yet a competitive
cost (1,211), because its threshold flags broadly. The documented criterion is cost; PR-AUC only breaks cost ties. Here the winner
leads both. The winner was XGBoost with `scale_pos_weight=473.6889` (170,528 training
negatives / 360 training positives) and a frozen threshold of
`0.00038929328712573435`:

| Split | Precision | Recall | F1 | PR-AUC | ROC-AUC | FPR | Confusion (TN, FP / FN, TP) |
|-------|-----------|--------|----|--------|---------|-----|-----------------------------|
| validation (selection) | 0.2927 | 0.8421 | 0.4344 | 0.7931 | 0.9822 | 0.002039 | 56,788 / 116 Â· 9 / 48 |
| test (frozen, once) | 0.3614 | 0.8000 | 0.4979 | 0.8002 | 0.9801 | 0.001863 | 56,777 / 106 Â· 15 / 60 |
| test majority class | 0.0000 | 0.0000 | 0.0000 | 0.0013 | 0.5000 | 0.000000 | 56,883 / 0 Â· 75 / 0 |

Per-class (test): fraud precision 0.3614, recall 0.8000, F1 0.4979 (support 75); legitimate
precision 0.9997, recall 0.9981, F1 0.9989 (support 56,883). At the assumed weights the test
errors cost 1,606 units: 15 false negatives x 100 = 1,500 (missed fraud, loss or chargeback,
never reviewed) plus 106 false positives x 1 = 106 (analyst review and customer friction).
Validation cost was 1,016 (9 false negatives, 116 false positives). The absolute threshold is
a ranking cut, not a validated business probability: the reliability/Brier comparison is
still M4 work, and reweighted XGBoost compresses its score scale. These are **one run's
numbers on one temporal split** - the test cost is 58% above validation cost, ordinary
out-of-sample variation with only 57-75 frauds, not a measured gain over another run. The run
was repeated once after an artifact-format change; the test metrics were identical, and no
test result entered selection. The earlier baseline, the comparison stage and the frozen final
evaluation have all scored this split, so it is no longer a pristine holdout; future tuning
must use validation or a fresh holdout. The final-evaluation stage performed exactly one new
test scoring pass, after writing its decision. Seeds are fixed, but tree parallelism changes
floating-point summation order: bit-exact replication of these numbers on another machine or
thread count requires `--n-jobs 1`. The 100:1 weights are provisional
(`ARCHITECTURE.md` section 13.2); changing them re-ranks the grid.

Artifacts: `model.joblib` (1.6 MB), `comparison.json` (20 KB), `metrics.json` (5.3 KB) and
`run.json` (schema 3, including the tracking block), all git-ignored; the bundle was reloaded in
a separate process. The tracking-enabled rerun on the same temporal split completed in 104.3 s
of total candidate fit time and wrote real MLflow run
`758f8670c78744159032d4fcabfff602` to experiment `fraud-detection` in the local SQLite store.
Its status is `FINISHED`, its deterministic fingerprint is
`6572a635b3c8ea41f933f67f90d00db285246e6f4d8dc9b74a067de2fddc8521`, and the run has the four
artifact files plus the parameters, tags and metrics described above. `MlflowClient` listed no
registered models, which is the intended separation: tracking does not register or deploy. This
is **one real tracking run**, not a historical series; the earlier comparison run predates
tracking, so do not read it as a two-run comparison.

**Frozen final evaluation also run on the reference dataset.** The command
`python -m fraud_risk.models.final_evaluation` (default expected-cost policy; 6 s wall clock) recreated
the temporal split, selected the threshold on validation only, wrote `decision.json` and a
byte-identical `frozen_model.joblib`, then scored test once. It reproduced the comparison's
threshold (`0.00038929328712573435`) and test metrics exactly - the expected reproducibility
result, not an independent new sample. The model SHA-256 is
`e43c600bf6560e6334e3f5efc2667b5d19d5f14dd0e71cfd55bba252518743c9`; `run.json` records the
decision file's SHA-256
(`feaeaa6e541fdbb45bee81602e818d5a4586deba3953e8515a391a4534c3f0c4`). Validation had 164
alerts (0.2879%), test 166 (0.2914%). The cost impact remains
1,016 validation units and 1,606 test units; the test split is still the same 56,958 rows
(56,883 legitimate / 75 fraud). `run.json` records `git_revision = ecb5fdb`; the worktree was
dirty at run time, so this is not commit-exact provenance until the changes are committed and
the command is rerun.

**SHAP explainability also run on the frozen model.** `python notebooks/explainability.py`
(default `artifacts/final_evaluation/frozen_model.joblib`, validation split, 2,000 uniformly
spaced rows, row 0, top 10) completed in 12.7 s wall clock in the latest run and wrote the six
report/JSON/CSV files plus two PNGs under `reports/explainability/`. The bundle SHA-256 matched
`e43c600bf6560e6334e3f5efc2667b5d19d5f14dd0e71cfd55bba252518743c9`. The global ranking on that
sample was V4 (mean |SHAP| 2.7243), V14 (2.1458), V12 (1.4679), V11 (1.1810), V10 (1.0867);
the explained row's base value was 3.5814 and averaged tree margin -13.0113, with additivity
residual -3.9e-06. A loaded model explains one transaction in **11.4 ms median** (20 timed
calls, max 12.0 ms; model load excluded). These are tree-margin attributions, not calibrated
probability points and not the frozen threshold. The report is git-ignored; the individual
section contains the top-k raw/model values for one row, which is why it must not be treated
as a data export. The implementation supports the selected XGBoost family only and does not
silently substitute an explainer for a different family.

Alternative operating policies were inspected on **validation only**, as a check that the
policy layer works. They were not scored on test:

| Validation policy | Threshold | Precision | Recall | F1 | Alerts | FPR |
|-------------------|-----------|-----------|--------|----|--------|-----|
| expected cost (reference) | 0.00038929328712573435 | 0.2927 | 0.8421 | 0.4344 | 164 | 0.002039 |
| minimum recall 0.80 | 0.0005363336500146721 | 0.4182 | 0.8070 | 0.5509 | 110 | 0.001125 |
| minimum recall 0.85 | 0.0003464947237638509 | 0.0888 | 0.8596 | 0.1609 | 552 | 0.008839 |
| capacity 0.5% (284 alerts) | 0.00038929328712573435 | 0.2927 | 0.8421 | 0.4344 | 164 | 0.002039 |
| capacity 1% (569 alerts) | 0.0003464947237638509 | 0.0888 | 0.8596 | 0.1609 | 552 | 0.008839 |

The 0.5% capacity is not binding, so it selects the same cut as expected cost. The jump from
0.80 to 0.85 minimum recall shows a precision cliff: the reweighted XGBoost score scale is
compressed, and the next cut that reaches 0.85 recall adds hundreds of false positives. This
is why a policy must be chosen for a stated review objective, and why no threshold is
universally optimal. Choosing among these rows by test performance would be test-set tuning;
the reference final run used the pre-declared expected-cost policy only.

Earlier, on the pre-3.11 scaffold: `git check-ignore` / `git add -A --dry-run` confirmed that
datasets, model artifacts, `.env`, `mlruns/` and `.venv/` are ignored while the `.gitkeep`
placeholders are tracked (git 2.46.2), and `docker compose config --quiet` exited 0
(Docker 29.8.2). `reports/` is ignored as well â€” `git check-ignore -v` matches
`.gitignore:91:reports/` for a generated report and figure, and a full EDA run adds nothing to
`git status`.

**Light path verified for the pre-splitting tree:** in a separate venv containing only
`pydantic-settings packaging pytest pandas` (all inside the declared ranges; no ML, API or
dashboard stack), `pytest -q -m "not slow"` gave `130 passed, 2 skipped, 25 deselected` â€” the
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
| `pytest -m integration` without `TEST_DATABASE_URL` | The two PostgreSQL tests skip by design; run Compose PostgreSQL and set the URL to execute them (done in the verification record above). |
| `docker compose up` application stack | Only PostgreSQL is defined; app services arrive in M9. |
| CI being green | `.github/workflows/ci.yml` has never run on GitHub. |
| General model quality or production readiness | One temporal split has been scored; the test split is no longer a pristine holdout, there is only one real MLflow run (not a two-run history) and no reliability/Brier comparison, and the cost weights are provisional assumptions. The reported numbers describe this run, not a financial decision system. |
| Linux/container re-verification of the modelling stages | Docker was unavailable; the recorded Linux column predates EDA, splitting, training and final evaluation. |

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
- Public scoring must be bound to loopback or a private network; model metadata, prediction
  history and review are protected by the configurable `API_ADMIN_TOKEN` bearer token.
- `.env`, datasets, model artifacts and database volumes are git-ignored.

## Documentation

| Document | Contents |
|----------|----------|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Components, data flow, data contracts, modelling and monitoring design, key decisions, open questions |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | Milestones M0â€“M9 with deliverables and exit criteria |
| [`AGENTS.md`](AGENTS.md) | Conventions and rules for future contributors (human or automated) |

## License

Not yet chosen. Until a license is added, all rights are reserved by the author.
