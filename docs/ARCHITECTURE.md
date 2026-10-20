# Architecture — Fraud Detection & Risk Intelligence Platform

> **Status:** design baseline, not an implementation report.
> Milestone M0 (see `docs/ROADMAP.md`) created the repository skeleton, configuration
> contracts and these documents. No component below is implemented yet, and no
> performance number appears anywhere in this repository.

---

## 1. Purpose and scope

The platform scores credit-card transactions for fraud risk, explains each score,
stores the prediction, and exposes both programmatically and visually.

**In scope**

- Reproducible offline pipeline: ingest → validate → explore → engineer features → train → evaluate.
- Versioned model artifacts with an explicit promotion path.
- A REST API that returns `score`, `decision` and per-prediction explanations.
- Durable prediction history in PostgreSQL with feedback capture.
- An analytics dashboard over the stored history.
- Monitoring for data drift and (once labels arrive) performance decay.

**Out of scope (deliberate non-goals)**

- Real cardholder data. Only public/synthetic datasets and locally generated fixtures.
- Real-time streaming infrastructure (Kafka/Flink). Batch/HTTP inference first; streaming is a
  possible later milestone that must not distort the current design.
- Deep learning / sequence models. Tree ensembles are the right complexity for tabular,
  imbalanced, low-label-volume fraud data and are directly SHAP-explainable.
- Multi-tenant identity, billing, or a full case-management UI.
- Auto-retraining loops. Retraining is manual and reviewed until monitoring has a track record.

---

## 2. System context

```
                 ┌──────────────────────────── OFFLINE (batch, human-triggered) ───────────────────────────┐
                 │                                                                                        │
  public/synthetic        ingestion +                   EDA & feature            training &                │
  transaction dataset ──► validation ──────────────►    engineering  ────────►   evaluation ──► MLflow    │
  (data/raw, never        (schema, ranges,             (src/fraud_risk/         (imbalance,     tracking + │
   committed)              leakage guards)              features/)               calibration)   registry   │
                 │                                                                        │                │
                 └────────────────────────────────────────────────────────────────────────┼───────────────┘
                                                                                          │
                                                                          model artifacts (artifacts/, promoted alias)
                                                                                          │
                 ┌──────────────────────────── ONLINE (request/response) ──────────────────┼───────────────┐
                 │                                                                        ▼                │
  client ──HTTP──► FastAPI (/predict, /health, …) ──► inference (load artifact, score) ──► SHAP explanation │
                 │            │                                        │                                 │
                 │            └────────────► PostgreSQL ◄──────────────┘  persistence: prediction,         │
                 │                            (history, feedback,          decision, threshold,            │
                 │                             drift reports)              explanation, model version        │
                 └────────────────────────────────────────┬───────────────────────────────────────────────┘
                                                          │
                                        Streamlit dashboard ──► monitoring / drift reports
```

Two paths, one shared core. The API and the batch jobs call the *same* feature and inference
code in `src/fraud_risk/`, so training/serving skew is a bug, not a design assumption.

---

## 3. Component map

| # | Area | Package / path | Responsibility | Status |
|---|------|----------------|----------------|--------|
| 1 | Data ingestion & validation | `src/fraud_risk/data/` | Load raw files from approved local paths; enforce schema, dtypes, ranges, nullability; block leakage columns; emit a validation report. | Loading + contract/report implemented; value ranges, fingerprint and inference-side leakage guard pending |
| 2 | EDA & feature engineering | `src/fraud_risk/features/` | Offline analysis (`features.eda` + `notebooks/eda.py`) + reusable, unit-tested transformers (a single `ColumnTransformer` used by training *and* inference). | EDA and the shared transformer implemented; joblib serialisation pending |
| 3 | Training & evaluation | `src/fraud_risk/models/splitting.py`, `training.py`, `evaluation.py` | Splitting, cross-validation, class-imbalance handling, probability calibration, metric computation, business-cost analysis. | Splitting implemented; training/evaluation scaffolded |
| 4 | Artifact management & inference | `src/fraud_risk/models/registry.py`, `inference.py` | Publish/load versioned artifact bundles (model + preprocessing + metadata); in-process caching; score/explanations. | Scaffolded |
| 5 | FastAPI application | `src/fraud_risk/api/` | HTTP surface, request/response schemas, dependency wiring, error handling, request IDs. | Scaffolded |
| 6 | Database layer | `src/fraud_risk/db/` | SQLAlchemy 2.x models, engine/session lifecycle, Alembic migrations. | Scaffolded |
| 7 | Streamlit dashboard | `src/fraud_risk/dashboard/` | Read-only analytics: volume, score/decision distribution, review queue, drift history. Talks to the API/DB, never to model internals. | Scaffolded |
| 8 | Monitoring | `src/fraud_risk/monitoring/` | Drift detection on inputs and scores, delayed-label performance tracking, report persistence. | Scaffolded |
| 9 | Tests & deployment | `tests/`, `.github/workflows/`, `docker-compose.yml` | Unit/integration tests, CI pipeline, local infrastructure. | Scaffolded |

Cross-cutting: `src/fraud_risk/config.py` (the only place environment configuration is read).

**Dependency direction** (imports point downwards only):

```
api / dashboard  ->  models.inference  ->  models.registry  ->  features  ->  data
                     |
                     +-> db (models + session)      monitoring -> db, features
```

`config.py` is imported by everyone and imports nothing from the project.

---

## 4. Data flow

### Offline (training)

1. `data.ingestion` reads a raw file from `data/raw/` into a `pandas.DataFrame` exactly as
   stored — no coercion, no cleaning — and refuses files it cannot parse (missing, empty,
   non-UTF-8, duplicate headers, ragged rows). Nothing is written back to `data/raw/`.
2. `data.validation` runs contract checks (columns, nullability, numeric/finite values,
   binary labels) and raises a typed error on violation. Failures are reported, never
   silently coerced; the summary report is row-free.
3. `features.eda` summarises the validated frame (dimensions, class balance, `Amount`/`Time`
   behaviour, per-feature statistics, feature/target associations). `notebooks/eda.py`
   renders it into the git-ignored `reports/eda/`. This stage fits **nothing**: no scaler,
   bin edge or quantile it computes is reused downstream.
4. `features.engineering` builds the feature matrix via one declarative pipeline. It is fitted
   on training rows only; `models.splitting` decides which rows those are and records the
   choice (strategy, seed, sizes, class rates, position digests) without storing data.
5. `models.training` consumes the split, fits candidate models, calibrates, and logs every run
   (params, metrics, artifacts, dataset fingerprint) to MLflow.
6. `models.evaluation` produces the comparison table, threshold analysis and error analysis.
7. The chosen run is exported as an artifact bundle and given a registry alias
   (`champion`). Promotion is an explicit, reviewed step.

### Online (inference)

1. Client `POST /predict` with a transaction JSON body (Pydantic-validated).
2. The API resolves the aliased model from the artifact store (cached in-process).
3. Features are built with the same transformer object that was fitted during training —
   the fitted transformer travels inside the artifact bundle.
4. Score → decision (three-way policy, §6) → SHAP top-k contributions.
5. One row is persisted to `predictions`, the explanation included as JSONB, then the
   response is returned.

**Persistence is not the caller's job.** The API records what it returned, so the dashboard
and drift monitor always describe real traffic.

---

## 5. Data contracts

### 5.1 Canonical transaction (input)

Reference dataset for development: the public **ULB credit-card fraud dataset**
(`creditcard.csv`), distributed on Kaggle as `mlg-ulb/creditcardfraud`. Its columns are
`Time`, `V1..V28`, `Amount`, `Class`; `Class` is the source target and maps to `is_fraud`
below. Provenance, licence and limitations are in [`DATA_CARD.md`](DATA_CARD.md), and the
operator workflow (manual placement or official Kaggle CLI) is in
[`../data/README.md`](../data/README.md). The canonical record is a superset so a more
realistic source can be swapped in later without touching the API contract:

| Field | Type | Notes |
|-------|------|-------|
| `transaction_id` | `str` (UUID) | Idempotency key; unique per scored transaction. |
| `event_time` | `datetime` (UTC) | Derived from the dataset `Time` column, or client-supplied. |
| `amount` | `float >= 0` | Transaction amount. |
| `currency` | `str` (ISO 4217) | Optional; defaults to the dataset currency. |
| `account_ref` | `str` | **Pseudo-anonymised** card/account reference. Never a PAN. |
| `merchant_category` | `str` | Optional; absent in the reference dataset. |
| `features` | `dict[str, float]` | Anonymised model inputs (`V1..V28` for the reference dataset). |
| `is_fraud` | `int` (0/1) | **Labels only.** Rejected on the inference path. |

The `is_fraud` column is the single most likely source of leakage; ingestion treats its
presence on an inference payload as a validation error.

### 5.2 Prediction record (output + persisted row)

| Field | Type | Notes |
|-------|------|-------|
| `prediction_id` | `uuid` | Primary key. |
| `transaction_id` | `str` | Natural key, unique index (idempotent scoring). |
| `created_at` | `timestamptz` | Server time, UTC. |
| `score` | `double` | Calibrated probability in `[0, 1]`. |
| `decision` | `enum` | `allow` \| `review` \| `block`. |
| `thresholds` | `jsonb` | The `review`/`block` thresholds in force at scoring time. |
| `model_name`, `model_version`, `mlflow_run_id` | `str` | Full traceability from a row back to a run. |
| `explanation` | `jsonb` | Top-k SHAP contributions (`[{feature, value, contribution}]`). |
| `feature_snapshot` | `jsonb` (nullable) | Inputs used, for drift analysis. Configurable, off by default in production. |
| `latency_ms` | `double` | Scoring latency. |

### 5.3 Tables

| Table | Purpose |
|-------|---------|
| `predictions` | One row per scored transaction (above). |
| `feedback_labels` | Analyst-confirmed outcomes, joined to `predictions` later. Delayed labels are the norm, so this is a separate table, not a column update that may never arrive. |
| `drift_reports` | Persisted drift runs (window, metric, value, threshold, breached flag) so the dashboard can show trends rather than a single point. |

Alembic owns the schema; migrations are generated in M2 and are the only way the schema
changes. No `create_all()` outside tests.

---

## 6. Modeling approach

**Splits.** Temporal first (train on earlier, test on later), stratified k-fold inside
training. A random split on `Time`-ordered data inflates scores and is treated as a defect.
`models.splitting` implements both rules. `temporal` sorts by `Time`, cuts 60/20/20 and pushes
each boundary to the next distinct timestamp, so rows sharing a second - often near-duplicates
- cannot straddle two subsets. `stratified` is class-proportional and seed-reproducible; it is
available for comparison and for genuinely independent data, but it must not produce the
headline score on this dataset. Fraud is not uniform over the two days, so the chronological
rule cannot preserve the global class rate; the realised rates are recorded instead of assumed.
The split record holds the strategy, seed, fractions, sizes, class counts, feature schema and a
SHA-256 digest of each subset's row positions, and no row-level values. It is written to the
git-ignored `reports/splits/` and travels with the artifact metadata.

**Features and scaling.** The model sees `V1..V28` and `Amount`. `Time` is deliberately not a
feature: it counts seconds since the first row of *this file*, so it cannot be computed when
scoring one transaction, and under the temporal split it would only be asked to extrapolate
beyond the training range. It remains the split-ordering key and the drift-window key. Tree
models receive raw values (they are scale-invariant, and SHAP values stay in original units);
linear and distance-based models get a per-column `StandardScaler` fitted on training rows
only. No log, power or quantile reshaping is applied until an evaluation justifies it. The
object returned by `features.engineering.build_preprocessor` is the single transform that
ships in the bundle (D3); resampling, if ever justified, belongs inside the training pipeline
after the split and never replaces or reweights validation/test rows.

**Imbalance.** Fraud rates are typically well under 1%. Strategy order:
`scale_pos_weight` / `class_weight` → threshold tuning → (only if justified) resampling.
Accuracy is never reported as a headline metric; a model that predicts "never fraud" is the
baseline that must be beaten.

**Calibration.** Scores are probabilities and the decision policy compares them to thresholds,
so calibration matters. Candidates are compared on a reliability curve / Brier score, and a
calibrator (`CalibratedClassifierCV`, isotonic or sigmoid) is part of the artifact bundle
when it helps.

**Metrics.** Primary: **PR-AUC (average precision)**. Secondary: recall at a fixed
precision target, precision at a fixed recall target, ROC-AUC (reported, never primary),
and a business-cost summary (`cost = FN·cost_fraud + FP·cost_review`) over a threshold sweep.
Every reported number must come from a real run on a real dataset, with the run id recorded.

**Decision policy (three-way).** A single 0.5 cut-off is wrong for fraud operations: it hides
the review queue.

```
score < review_threshold            -> allow
review_threshold <= score < block   -> review   (queued for an analyst)
score >= block_threshold            -> block
```

Thresholds are configuration (`.env` / settings), tuned on the validation split against the
cost function, and stored with every prediction so historical rows stay interpretable after
a policy change.

---

## 7. Explainability

- Per-prediction explanations use **SHAP TreeExplainer** on the fitted tree ensemble
  (`shap.TreeExplainer`, `check_additivity=False` where floating-point noise makes it fail).
- The API returns the top-k absolute contributions plus a base value.
- Explanations are computed **on demand** for a single transaction. Full-dataset SHAP sweeps
  belong in the offline EDA/analysis path, not in a request handler.
- Performance note: explainer construction is expensive, so the explainer is built once per
  loaded artifact and cached alongside the model.

**Honest limitation, recorded on purpose:** the reference dataset's `V1..V28` are the output
of a PCA, so a SHAP value on `V14` is not human-interpretable. Mitigation is a second,
aggregate explanation layer (grouped/permutation importance over `Amount`, hour-of-day, and
`V*` blocks) for analyst-facing text. We do not claim semantic feature stories the data
cannot support.

---

## 8. Drift monitoring

**Chosen approach: a small, first-party drift module, with Evidently as an optional extra.**

Rationale: this module runs on a schedule and must be testable and dependency-light; it needs
a handful of well-understood statistics on a fixed set of columns, not a framework. Evidently
is genuinely useful for its rich report HTML and is therefore offered as an optional extra
(`pip install -e ".[monitoring]"`), used to render reports — but the *decision metric* stays
in our code so it can be unit-tested and its thresholds version-controlled.

What is monitored:

| Signal | Metric | Window |
|--------|--------|--------|
| Input features | Population Stability Index per feature (binned, reference = training distribution) | rolling |
| Input features | Kolmogorov–Smirnov two-sample statistic + p-value | rolling |
| Model score | PSI + mean/quantile shift of `score` | rolling |
| Traffic | Prediction volume, `allow`/`review`/`block` mix | rolling |
| Performance | PR-AUC / recall on joined `feedback_labels` | rolling, delayed |

Alerting is threshold-based and reported in `drift_reports`. A PSI above the configured
warning/alert levels is a *prompt to investigate*, not an automatic retrain trigger.

---

## 9. Configuration and secrets

- One settings object, `fraud_risk.config.Settings` (pydantic-settings), reads environment
  variables, optionally from `.env`. It is the only place `os.environ` is consulted.
- `.env.example` contains **placeholders only** and is committed; `.env` is git-ignored.
- No credential ever has a real default. Defaults are obviously-fake local placeholders.
- `DATABASE_URL` overrides the assembled `postgresql+psycopg://` DSN; the DSN property is
  the single source of truth for both the app and Alembic.
- Streamlit secrets (`secrets.toml`) are git-ignored; the dashboard reads API/DB config from
  environment variables like every other component.
- CI uses service containers / throwaway credentials, never repository secrets for databases.

---

## 10. Deployment topology

**Local (target of M1–M8).** Python venv + `docker compose up -d postgres` + `uvicorn` +
`streamlit run` + MLflow tracking to a local file store (`./mlruns`), all configured by `.env`.

**Containerised (M9).** One image, several commands (api / dashboard / trainer), plus
PostgreSQL. Compose for local integration; the same image is the deployment unit so the
training and serving environments cannot silently diverge.

`docker-compose.yml` currently defines **only** PostgreSQL. Application services are added in
M9 with the Dockerfile; no service definition is committed before the image that backs it.

---

## 11. Testing strategy

| Level | Scope | Notes |
|-------|-------|-------|
| Unit | Transforms, validators, threshold policy, drift statistics, settings | Pure functions, no I/O, fast. |
| Integration | API endpoints, DB writes, migrations | `httpx` ASGI transport against the FastAPI app; PostgreSQL via a Compose service in CI. |
| Contract | Request/response schemas, artifact bundle layout | Guards against silent train/serve skew. |
| Smoke | `/health`, model-load error path | Ensures the API fails loudly when no artifact is available. |

Rules: SQLite is not used to stand in for PostgreSQL in tests (behaviour differs); the model
in integration tests is a tiny deterministic estimator, never a real trained model; flaky
tests are fixed or deleted, never retried.

---

## 12. Key design decisions

| # | Decision | Why | Alternative rejected |
|---|----------|-----|----------------------|
| D1 | `src/` layout, single package `fraud_risk` | Prevents accidental imports of the repo root and forces testing of the installed package. | Flat top-level modules (import shadowing). |
| D2 | One `pyproject.toml` for deps + tooling | One resolver, one lockstep version story, no `requirements*.txt` drift. | Split requirements files. |
| D3 | Fitted transformer ships inside the artifact bundle | Removes train/serve skew by construction. | Re-fitting transforms at serving time. |
| D4 | Three-way decision instead of a single 0.5 cut-off | Fraud operations need an explicit review queue; thresholds are business policy, not a model constant. | Binary threshold. |
| D5 | Calibrated probabilities | Thresholds and cost analysis are only meaningful if scores are probabilities. | Raw `predict_proba` from an uncalibrated booster. |
| D6 | Persist predictions unconditionally in the API | Makes dashboard and drift data real traffic, not a reconstruction. | Persist from the client. |
| D7 | Feedback labels in their own table | Labels arrive late and incompletely; a nullable column on `predictions` invites the assumption they exist. | `actual_label` column. |
| D8 | First-party drift statistics + optional Evidently | Testable, versioned thresholds, light dependency; Evidently kept for rich reports. | Evidently as the decision engine. |
| D9 | MLflow for tracking and registry aliases | Standard, self-hostable, records params/metrics/artifacts per run. | Hand-rolled JSON run logs. |
| D10 | No `create_all()` outside tests; Alembic only | Schema changes stay reviewable and reversible. | `metadata.create_all()` at startup. |
| D11 | Configuration strictly via environment | Nothing environment-specific is committed; secrets cannot leak via code. | Committed config files. |
| D12 | PoC scope excludes streaming | Keeps the first deliverable honest and finishable. | Kafka from day one. |

---

## 13. Open questions (to resolve before the milestone that depends on them)

1. **Dataset.** ✅ **Resolved.** The public ULB credit-card fraud dataset
   (`creditcard.csv`, Kaggle `mlg-ulb/creditcardfraud`) is selected; source, licence, schema,
   target mapping and limitations are documented in [`DATA_CARD.md`](DATA_CARD.md). The
   explainability consequence (PCA-derived features) is recorded in §7.
2. **Cost function weights** (`cost_fraud`, `cost_review`) and the target precision for the
   review queue. *Blocks M4.*
3. **Promotion gate.** Minimum PR-AUC / drift criteria required to move an alias to
   `champion`. *Blocks M5.*
4. **Feature snapshot policy.** Whether to persist raw inputs by default (storage + privacy
   cost vs. drift-analysis value). *Blocks M6.*
5. **Auth on the API.** Out of scope for the PoC; must be decided before any exposure beyond
   localhost. *Blocks M9.*

---

## 14. Dependency layout and rationale

Policy (mirrors `AGENTS.md` §8): one dependency file (`pyproject.toml`), compatible-release
ranges (`>=x,<y`) rather than exact pins, and every package *declared* rather than assumed
present transitively.

**Why the ML stack is a core dependency, not an extra.** §10 chose one image with several
commands so that training and serving cannot silently diverge. That decision forces the
boundary: anything the serving path imports at request time must be in `dependencies`.

| Package | Group | Why it is where it is |
|---------|-------|-----------------------|
| `numpy`, `pandas`, `scikit-learn`, `joblib` | core | Not training-only. The fitted transformer ships inside the artifact bundle (D3), so the inference path imports `sklearn` and `pandas` on every load. |
| `xgboost` | core | The estimator itself. |
| `shap` | core | Per-prediction explanations are a shipped API feature (§7), not an offline report. |
| `mlflow` | core | Run tracking *and* registry-alias resolution at serving time (D9). |
| `fastapi`, `uvicorn[standard]` | core | The HTTP surface. |
| `pydantic`, `pydantic-settings` | core | API schemas (trust boundary) and `Settings` (§9). |
| `sqlalchemy`, `alembic`, `psycopg[binary]` | core | Persistence and migrations (D10). |
| `streamlit`, `plotly` | core | The dashboard command of the same image (§10). |
| `evidently` | extra `monitoring` | Rich report rendering only. The drift *decision* is first-party so thresholds stay testable and version-controlled (D8). |
| `pytest`, `pytest-cov`, `httpx`, `ruff`, `mypy`, `packaging`, `pandas-stubs`, `matplotlib` | extra `dev` | Never imported by `src/`. `httpx` is required by FastAPI's `TestClient`; `packaging` is used by the environment smoke test to verify installed versions against the declared ranges. `pandas-stubs` exists because pandas ships no inline types: without it `mypy --strict` treats every `DataFrame` as `Any`, which would make the data layer's annotations decorative. `matplotlib` renders the EDA figures from `notebooks/eda.py`; it stays out of `dependencies` because no module under `src/` may import a plotting library (AGENTS.md §2), which also keeps the serving image free of it. |

**Invariants:** nothing in `dependencies` is used only by an extra; nothing in an extra is
imported by `src/fraud_risk/`.

**Known trade-off, recorded deliberately.** `shap` pulls in `numba` + `llvmlite`, the heaviest
transitive chain in the tree. A slim API image (ML stack in an extra) would cut image size and
cold-start time, and that is the first thing to revisit if image size becomes a real problem.
It is not done now, because it would reintroduce exactly the train/serve divergence that D3 and
§10 exist to prevent.

**Known limitation.** There is no lock file, so the resolver picks the newest version inside
each range at install time, and it legitimately picks *different* versions on different
interpreters (for example numpy 2.4.x on Python 3.11 where 2.5.x requires a newer Python).
`tests/test_smoke_imports.py` pins the contract in executable form — every installed version
must satisfy its declared range — but two machines can still differ within those ranges.
Adding a lock file is the fix, and is listed in `docs/ROADMAP.md` under Backlog.
