# Architecture — Fraud Detection & Risk Intelligence Platform

> **Status:** partial implementation, not a completed system.
> M0, M1 and M3 are complete. M4 is implemented for the three-family comparison, the frozen
> final evaluation and MLflow run tracking, with stratified CV and the reliability/Brier
> comparison still open. There is one real tracking run, not a two-run history. M5 has the
> shared artifact-bundle loader, the offline and per-request XGBoost SHAP layer, a reusable lazy
> inference service and FastAPI-ready request/response schemas; registry alias resolution and
> the promotion gate are not implemented. M2 has the predictions table, Alembic migration and
> repository read/write/filtered-history/summary paths; feedback-label and drift-report tables
> remain open. M6 is partially implemented: the versioned HTTP surface, dependency wiring, CORS policy,
> safe error mapping and admin-protected history/filtering/summary/review routes exist;
> request-id logging and feedback capture remain open. M7 is partially implemented: the
> Streamlit overview, scoring, filtered history, explanation and analyst-review views read and
> write through the API; drift-history views are still future work. The monitoring package is
> still scaffolded. Real metrics from the local reference runs are recorded in the README
> verification record and in the git-ignored artifacts; they describe those runs only.

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
| 2 | EDA & feature engineering | `src/fraud_risk/features/` | Offline analysis (`features.eda` + `notebooks/eda.py`) + reusable, unit-tested transformers (a single `ColumnTransformer` used by training *and* inference). | EDA and the shared transformer implemented; the fitted transformer ships inside the joblib artifact bundle |
| 3 | Training & evaluation | `src/fraud_risk/models/splitting.py`, `training.py`, `evaluation.py`, `final_evaluation.py`, `bundle.py`, `explainability.py` | Splitting, cross-validation, class-imbalance handling, probability calibration, metric computation, business-cost/operating-policy analysis, frozen final evaluation, artifact-bundle loading and SHAP explanations. | Splitting, the three-family comparison (logistic regression, random forest, XGBoost), calibration, per-class/cost metrics, validation-only cost/min-recall/alert-capacity policies, the frozen final evaluation, XGBoost SHAP explanations and serving inference implemented; CV and reliability/Brier comparison pending |
| 3a | Experiment tracking | `src/fraud_risk/tracking.py` | Optional MLflow boundary: configurable tracking URI, deterministic run fingerprint, parameter/tag/metric/artifact logging, explicit failed-run status. Registration and promotion are deliberately outside it. | Implemented for the training CLI; one real local SQLite-backed run demonstrated. No model registered |
| 4 | Artifact management & inference | `src/fraud_risk/models/registry.py`, `inference.py` | Publish/load versioned artifact bundles (model + preprocessing + metadata); in-process caching; score/explanations. | `models.inference` implemented: lazy validated artifact loading, schema/order and preprocessing compatibility checks, one-row scoring through the fitted pipeline, frozen binary threshold, model version/SHA and optional SHAP. Registry aliases remain scaffolded; `api.schemas` provides Pydantic request/response models |
| 5 | FastAPI application | `src/fraud_risk/api/` | HTTP surface, request/response schemas, dependency wiring, error handling, request IDs. | Application factory, dependency providers, versioned health/readiness/model-metadata/single-prediction/batch-prediction/history-list/summary/review routes, Pydantic contracts, per-item batch errors, request/batch limits, CORS policy, safe error mapping, admin bearer-token auth and access logging implemented; monitoring and request IDs pending (M6) |
| 6 | Database layer | `src/fraud_risk/db/` | SQLAlchemy 2.x models, engine/session lifecycle, Alembic migrations. | Predictions model, session lifecycle, initial Alembic migration and repository create/read/list/summary/review paths implemented; feedback-label and drift-report tables pending (M2) |
| 7 | Streamlit dashboard | `src/fraud_risk/dashboard/` | Read-only analytics: volume, score/decision distribution, review queue, drift history. Talks to the API/DB, never to model internals. | Overview counts, risk/decision/review charts, filtered history, transaction scoring, explanation rendering and an admin-only analyst review workflow implemented against the API; drift-history views pending (M7) |
| 8 | Monitoring | `src/fraud_risk/monitoring/` | Drift detection on inputs and scores, delayed-label performance tracking, report persistence. | Scaffolded |
| 9 | Tests & deployment | `tests/`, `.github/workflows/`, `docker-compose.yml` | Unit/integration tests, CI pipeline, local infrastructure. | Scaffolded |

Cross-cutting: `src/fraud_risk/config.py` (the only place environment configuration is read)
and `src/fraud_risk/tracking.py` (the optional MLflow boundary used by training; it imports
configuration types only under `TYPE_CHECKING` and imports MLflow lazily).

**Dependency direction** (imports point downwards only):

```
api / dashboard  ->  models.inference  ->  models.registry  ->  features  ->  data
                     |
                     +-> db (models + session)      monitoring -> db, features

models.explainability  ->  models.bundle  ->  models.evaluation
models.explainability  ->  features  ->  data
models.final_evaluation  ->  models.bundle
models.inference  ->  models.explainability, models.bundle, models.evaluation
models.training  ->  tracking  (lazy mlflow; no project imports at runtime)
```

`models.bundle` owns the shared artifact contract, so the final-evaluation stage and the
explainability layer both load the same validated bundle without importing each other.
`config.py` is imported by everyone and imports nothing from the project. `tracking.py` is a
leaf boundary: it accepts plain mappings and `TYPE_CHECKING`-only settings types, so it does
not create an import cycle with training or configuration.

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
5. `models.training` consumes the split, fits the candidate grid (logistic regression, random
   forest, XGBoost), calibrates each candidate on training folds, ranks them on validation
   expected cost, and writes a local artifact bundle. When enabled, `tracking` starts an
   MLflow run before fitting and records the configuration, split/schema metadata, selected
   model, validation/test metrics and the artifact files. It records a deterministic
   `run_fingerprint` and marks a run `FAILED` if fitting or artifact handling raises.
6. `models.evaluation` scores the chosen model on validation and, once the threshold is
   frozen, on test: PR-AUC, precision/recall/F1, confusion matrix, ROC-AUC, FPR and per-class
   metrics - always next to the majority-class baseline. Expected cost at the validation
   threshold is the selection criterion; the reliability/Brier calibration comparison and
   richer error analysis remain M4 work.
7. A chosen run is exported as an artifact bundle and, later, given a registry alias
   (`champion`). Training never registers or deploys a model; promotion is an explicit,
   reviewed M5 step.

### Online (inference)

1. Client `POST /api/v1/predict` with a transaction JSON body (Pydantic-validated).
2. The API resolves the configured artifact path (registry aliases are still M5 work) and
   asks the process-wide `models.inference.FraudInferenceService` to load it once.
3. Features are validated against the bundle's exact names and order, then passed to the
   same fitted transformer object that was fitted during training — it travels inside the
   artifact bundle and is never rebuilt at serving time.
4. The service calls the fitted `predict_proba`, applies the bundle's binary threshold and,
   when requested, returns SHAP top-k contributions. The three-way review/block policy is
   still an API-layer concern (§6).
5. The API persists the returned score, threshold, model identity and binary decision through
   a session-scoped repository. The raw feature snapshot is stored only when
   `PERSIST_FEATURE_SNAPSHOT=true`; prediction history is readable by prediction id.

**Implemented HTTP surface (M6, partial).** All routes are mounted under
`Settings.api_prefix` (default `/api/v1`): `GET /health` (liveness only), `GET /ready`
(attempts to load and validate the artifact; `200` ready, `503` not ready), `GET /model`
(admin-only: model name/version/SHA/family/threshold/feature schema), `POST /predict` (one
transaction), `POST /predict/batch` (a bounded list), `GET /predictions` (admin-only bounded,
newest-first history with optional `created_from`/`created_to`, risk-score, decision and
review-status filters), `GET /predictions/summary` (admin-only aggregate counts and fixed-width
risk bins), `GET /predictions/{id}` (admin-only history) and
`PATCH /predictions/{id}/review` (admin-only analyst review). Each successful prediction
carries the model identity, probability, applied threshold, raw binary `prediction` and a
human-readable `decision`. Batch items are validated independently: a malformed transaction
produces a safe error at its own index while valid items are still scored. The batch item count
is capped by `BATCH_MAX_SIZE`, and an ASGI middleware enforces `API_MAX_REQUEST_BYTES` before
JSON parsing. Model loading, scoring and repository calls run in Starlette's thread pool behind
async handlers. Model failures are logged server-side and returned as a non-leaking
`503 {"detail": "model unavailable"}`.

**Predicted versus confirmed fraud.** `decision` is the model's thresholded prediction. It is
never ground truth and never a human review outcome. Every persisted prediction starts with
`review_status="pending"`; an authorized analyst may move it to `confirmed_fraud` or
`confirmed_legitimate`, and those states are terminal. An invalid transition returns `409`
without a commit. The summary's `fraud_predictions` count is derived from the model `decision`,
whereas `confirmed_fraud` is derived only from the analyst review status. No code path copies a
model prediction into a confirmed outcome.

**Access model.** Health, readiness and scoring are public. Readiness reports availability only,
not model identity or the feature schema. Model metadata, prediction history and review
operations require `Authorization: Bearer <API_ADMIN_TOKEN>`. If the token is unset, those
routes fail closed with `503 {"detail": "administrative API is not configured"}`; they never
become anonymous. A missing/wrong credential returns `401` with `WWW-Authenticate: Bearer`.
This is a single shared administrative token, not per-user identity or role-based authorization.
`/docs` and `/openapi.json` are available locally but disabled when `APP_ENV=production`;
ReDoc and debug mode are always disabled. CORS is an exact-origin allow-list from
`CORS_ALLOWED_ORIGINS`, without credentials; `Authorization` and `PATCH` are allowed, and a
wildcard is rejected outside `local`/`test`.

**Persistence is the API's job, not the caller's.** The API records what it returned, so the
dashboard and drift monitor describe real traffic. The production database schema is owned by
Alembic; the application never calls `metadata.create_all()`. Repository writes commit once per
unit of work and roll back on `SQLAlchemyError`; the request-scoped session dependency also
rolls back and always closes. Access logs record method, path, status and duration only; they
never record query strings, headers, bearer tokens or request bodies. Validation details never
echo the submitted feature values, and unexpected failures return
`500 {"detail": "internal server error"}` without a traceback or filesystem path.

**Dashboard (M7, partial).** The Streamlit app calls the admin-protected history/summary,
model-metadata and review routes plus the public prediction route through
`dashboard.api_client`; it never imports model internals or queries PostgreSQL directly.
`dashboard.transformations` holds the deterministic counts/frame logic and `dashboard.charts`
holds the Plotly figures. The overview uses database aggregate counts and fixed-width risk bins,
so its totals do not depend on a paginated page. The analyst-review tab filters persisted rows
by date, risk score, model prediction and review status, and only offers a review form for
pending rows. Tables and metrics explicitly label the model's predicted label separately from
the analyst's review outcome. Model performance remains explicitly unavailable until feedback
labels or an evaluation endpoint exist. On an unreachable API or an invalid response, the
affected section shows an error state rather than a synthetic dataset. The admin token comes
from `Settings.api_admin_token`; review controls are hidden without it, and the API independently
enforces authorization. The token is never embedded in the dashboard or displayed.

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
| `prediction_id` | `uuid` | Primary key, generated by the repository. |
| `transaction_id` | `str` (128) | Indexed natural key. Uniqueness/idempotency is **not** enforced by the database yet. |
| `created_at` | `timestamptz` | Server time, UTC. |
| `model_version`, `model_sha256` | `str` | Model identity at scoring time. |
| `risk_score` | `double` | Calibrated probability in `[0, 1]`. |
| `threshold` | `double` | The binary decision threshold applied at scoring time. |
| `decision` | `enum` | `legitimate` \| `fraud` (the model's binary decision). |
| `review_status` | `enum` | `pending` \| `confirmed_fraud` \| `confirmed_legitimate`. |
| `feature_snapshot` | `jsonb` (nullable) | Inputs used, only when `PERSIST_FEATURE_SNAPSHOT=true`; off by default. Never contains a PAN or account reference. |

The richer three-way review/block policy, per-prediction explanations, latency and feedback
labels are documented design targets but are **not** in the current `predictions` schema. The
three-way thresholds are currently configuration/reporting concepts; the persisted decision is
binary.

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

**Implemented comparison.** `models.training` fits a bounded grid across logistic regression,
random forest and XGBoost on the shared feature pipeline (standardisation for the linear model,
passthrough for the scale-invariant trees), each candidate wrapped in a sigmoid calibrator
fitted on internal folds of the *training* data. Imbalance is handled by training-only
reweighting - `class_weight` (`None` / `"balanced"`, `"balanced_subsample"` for the forest) and
`scale_pos_weight` derived from the training labels - with no resampling anywhere. Every
candidate's threshold is chosen on validation to minimise expected cost under the configured
`FN:FP` weights, and the candidate with the lowest validation cost wins; ties break on
validation PR-AUC. Accuracy never participates. The winner and its exact threshold are frozen
and the test split is scored once. `comparison.json` records every candidate, `metrics.json`
reports precision, recall, F1, the confusion matrix, PR-AUC, ROC-AUC, FPR, per-class metrics
and the cost impact of false negatives versus false positives for the model and the
majority-class baseline on both splits; accuracy appears only in the deliberate
`accuracy_comparison` block. `run.json` (schema 3) carries the seed, feature schema, split
record, grids, computational limits, imbalance derivation, library versions, git revision,
`run_fingerprint` and the MLflow identity or an explicit disabled record. The reliability
curve / Brier calibration comparison, correlation-aware error analysis and stratified CV
remain M4 work.

**Experiment tracking.** `fraud_risk.tracking` is the only module that imports MLflow, and it
does so lazily. `TrackingConfig` takes the URI and experiment name from `Settings`; the CLI
also accepts `--tracking-uri`, `--experiment-name` and `--no-mlflow`. The default local URI is
`sqlite:///mlflow.db`, because MLflow 3.x places the plain filesystem backend behind
`MLFLOW_ALLOW_FILE_STORE`. A run starts after split validation and before fitting, so a failure
during fitting, artifact writing or artifact upload is recorded as `FAILED` with
`failure.type`/`failure.message` tags and the original exception is re-raised. Logged data is
limited to aggregate configuration, split metadata and metric names/values; no rows, feature
snapshots, credentials or DSNs are logged, and tracking URIs are redacted before being written
to local records. Artifacts are the four local files, uploaded as files rather than through
`mlflow.log_model`; **no model is registered and no alias is created by training**.

**Operating policies and final evaluation.** `models.final_evaluation` consumes the winning
comparison bundle and selects the operating threshold on validation under one explicit policy:
expected cost (`CostPolicy`), a minimum recall target (`MinRecallPolicy`), or a limited
review capacity (`AlertCapacityPolicy`, absolute count or share of rows). The model is not
re-selected. Probabilities and binary decisions are separate outputs: `binary_decisions` is
the only threshold comparison, ranking metrics use the probabilities, and
precision/recall/F1/confusion/FPR/alerts use the decisions. The stage copies the bundle to
`frozen_model.joblib`, writes `decision.json` with the model SHA-256, policy, exact threshold,
validation metrics and validation split digest, and only then reads test labels and scores
test once. `metrics.json` reports the probability summary and decision summary separately
for both splits; `run.json` records the full split, policy, cost weights, library versions and
git revision. Thresholds are policy choices, not universally optimal values. The test split
was already scored by the comparison stage, so the final report is a repeat evaluation, not a
pristine holdout.

**Calibration.** Scores are probabilities and the decision policy compares them to thresholds,
so calibration matters. Every candidate is wrapped in a `CalibratedClassifierCV` (sigmoid)
fitted on training folds, and the calibrator ships inside the artifact bundle. Comparing
sigmoid against isotonic and uncalibrated scores on a reliability curve / Brier score is still
M4 work; until then the stored threshold is a validated ranking cut, not a validated
probability.

**Metrics.** The primary reported ranking metric is **PR-AUC (average precision)**; the
model-selection criterion is validation expected cost. Also reported: precision, recall and
F1 at the operating threshold, the confusion matrix, per-class metrics, ROC-AUC (reported,
never primary), FPR, the number and share of alerts, and the expected-cost impact
(`cost = FN·cost_fraud + FP·cost_review`) at the validation-selected threshold. Probabilities
and decisions are serialised separately: the AUCs and score summaries come from probabilities,
while the error metrics and alert counts come from the binary decisions.
Every reported number must come from a real run on a real dataset, with the run id recorded.
`run.json` and the MLflow run (when tracking is enabled) are that record; the deterministic
`run_fingerprint` identifies the reproducible inputs, while MLflow's random `run_id` identifies
the individual execution.

**Decision policy (three-way).** A single 0.5 cut-off is wrong for fraud operations: it hides
the review queue.

```
score < review_threshold            -> allow
review_threshold <= score < block   -> review   (queued for an analyst)
score >= block_threshold            -> block
```

Thresholds are configuration (`.env` / settings), tuned on the validation split against the
cost function, and stored with every prediction so historical rows stay interpretable after
a policy change. The single threshold selected by `models.final_evaluation` is the binary
review cut for that stage; the cost-optimal cut is a *provisional ranking cut* under the
provisional 100:1 weights, and the min-recall / alert-capacity policies are alternative
operating choices. None of them is the three-way review/block policy above, which belongs to
the API and is tuned against the cost function once the business confirms the weights
(section 13.2).

---

## 7. Explainability

**Implemented layer (M5, offline + reusable core).** `fraud_risk.models.explainability`
loads the frozen bundle and explains the selected calibrated XGBoost model with
`shap.TreeExplainer` (`model_output="raw"`, `feature_perturbation="tree_path_dependent"`).

- The bundle pipeline is `preprocess -> CalibratedClassifierCV(XGBClassifier)`. The
  calibrated wrapper contains three tree models fitted on internal training folds. One
  explainer is built per fold and cached with the loaded bundle.
- The explained output is the **pre-calibration XGBoost tree margin**. Per-fold SHAP values
  are averaged, and the module verifies additivity (`base_value + contributions = margin`)
  against the estimator before returning an explanation.
- The final calibrated probability and the decision threshold are deliberately separate.
  They are not derived from SHAP values and do not appear in the explanation record. The
  explanation is **not** an additive decomposition of `predict_proba`, because sigmoid
  probability calibration and fold averaging happen after the explained output.
- `raw_value` is the caller's input; `model_value` is the exact array produced by the
  bundle's fitted preprocessor. They differ for a scaled preprocessor and are identical for
  the selected passthrough tree model.
- Global importance is mean absolute contribution over a deterministic sample. Per-request
  explanations return the top-k absolute contributions plus a base value, the remaining
  contribution and the additivity residual.
- `notebooks/explainability.py` renders `reports/explainability/` (git-ignored): Markdown,
  strict JSON, a CSV ranking, one transaction record and two PNGs. Plotting imports stay
  outside `src/` (AGENTS.md section 2).
- The implementation supports the **selected XGBoost family only**. A random-forest or
  logistic bundle raises a clear error rather than silently substituting a different
  explainer or output space. Random forest would need a probability-space contract and
  logistic regression a `LinearExplainer`; neither is claimed here.

**Honest limitations, recorded on purpose:** the reference dataset's `V1..V28` are the output
of a PCA, so a SHAP value on `V14` is not human-interpretable. Correlated features share
credit, tree-path-dependent SHAP uses the fitted trees' cover rather than an interventional
background, and a SHAP value is an attribution to this fitted model, not a causal effect.
Global importance is not a measure of model quality or a promotion gate. Mitigation for
analyst-facing text remains a second aggregate layer grouped over `Amount`, hour-of-day and
`V*` blocks; we do not claim semantic feature stories the data cannot support.

### 7.1 Reusable inference service

`models.inference.FraudInferenceService` is the only scoring entry point outside the API
wiring. It is constructed with an artifact path, or reads `Settings.model_artifact_path`
(default `artifacts/final_evaluation/frozen_model.joblib`), and performs no I/O until the
first request.

- **Load and cache.** `load_model_bundle` validates the artifact schema, threshold, feature
  columns and `predict_proba`; the service then checks that the fitted `preprocess` step's
  input and output names exactly match the bundle's feature order and that the classifier
  expects that many features. A missing, corrupt or incompatible artifact raises
  `InferenceModelError`; the API can map that to 503.
- **Validate once per request.** The request carries a single transaction's features and an
  optional top-level amount. The service rejects missing, extra, reordered, non-finite or
  negative values before touching the estimator. It builds a one-row frame in the artifact's
  exact column order, so a reordered JSON object can never be silently realigned.
- **Score without fitting.** Only the loaded pipeline's `predict_proba` is called. The service
  never fits, calibrates or rebuilds preprocessing; the threshold comes from the bundle and the
  binary decision is `probability >= threshold`.
- **Identity and explanation.** The response carries `model_version` (the bundle
  `run_fingerprint`, or the artifact SHA-256 for older bundles), `model_sha256`, `model_family`,
  the probability, threshold and binary prediction. `explain=True` builds/caches the SHAP layer
  for the selected XGBoost bundle; unsupported families fail loudly rather than returning a
  different output.
- **API-facing types.** `api.schemas.PredictionRequest` / `PredictionResponse` are Pydantic v2
  models used by the versioned FastAPI routes. They reject label fields and unknown fields,
  constrain the amount and probability ranges, and expose no feature snapshot.

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
- `MODEL_ARTIFACT_PATH` selects the single bundle the inference service loads; its default is
  the locally frozen final-evaluation artifact. `MODEL_DIR` is reserved for the registry
  layout, which is not implemented yet.
- Streamlit secrets (`secrets.toml`) are git-ignored; the dashboard reads API/DB config from
  environment variables like every other component.
- CI uses service containers / throwaway credentials, never repository secrets for databases.

---

## 10. Deployment topology

**Local (target of M1-M8).** Python venv + `docker compose up -d postgres` + `uvicorn` +
`streamlit run` + MLflow tracking to a local SQLite store (`sqlite:///mlflow.db`), all
configured by `.env`. `mlflow ui --backend-store-uri sqlite:///mlflow.db` inspects the runs;
the plain `./mlruns` filesystem backend is not the default because MLflow 3.x maintenance-mode
gates it.

**Containerised (M9).** One image, several commands (api / dashboard / trainer), plus
PostgreSQL. Compose for local integration; the same image is the deployment unit so the
training and serving environments cannot silently diverge.

`docker-compose.yml` currently defines **only** PostgreSQL. Application services are added in
M9 with the Dockerfile; no service definition is committed before the image that backs it.

---

## 11. Testing strategy

| Level | Scope | Notes |
|-------|-------|-------|
| Unit | Transforms, validators, threshold policy, SHAP schema/additivity, drift statistics, settings | Pure functions, no I/O, fast. |
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
| D9 | MLflow for tracking and registry aliases | Standard, self-hostable, records params/metrics/artifacts per run. Training writes a run and a deterministic fingerprint but never registers a model; the registry alias is a separate reviewed M5 step. | Hand-rolled JSON run logs. |
| D10 | No `create_all()` outside tests; Alembic only | Schema changes stay reviewable and reversible. | `metadata.create_all()` at startup. |
| D11 | Configuration strictly via environment | Nothing environment-specific is committed; secrets cannot leak via code. | Committed config files. |
| D12 | PoC scope excludes streaming | Keeps the first deliverable honest and finishable. | Kafka from day one. |
| D13 | Explain the pre-calibration XGBoost tree margin, not `predict_proba` | TreeExplainer gives an exact additive decomposition for the tree's raw margin. Calibration and fold averaging happen after that output, so calling the contributions probability points would be false. | KernelSHAP over the calibrated pipeline without a recorded background sample; probability-space attributions with no additivity check. |

---

## 13. Open questions (to resolve before the milestone that depends on them)

1. **Dataset.** ✅ **Resolved.** The public ULB credit-card fraud dataset
   (`creditcard.csv`, Kaggle `mlg-ulb/creditcardfraud`) is selected; source, licence, schema,
   target mapping and limitations are documented in [`DATA_CARD.md`](DATA_CARD.md). The
   explainability consequence (PCA-derived features) is recorded in §7.
2. **Cost function weights** (`cost_fraud`, `cost_review`) and the target precision for the
   review queue. **Provisionally resolved for M4:** the comparison uses `FN:FP = 100:1`
   abstract cost units (`FALSE_NEGATIVE_COST` / `FALSE_POSITIVE_COST`), recorded with every
   run and used to pick both the threshold and the winning candidate. These are assumptions,
   not measured business costs; changing them re-ranks the grid. The final-evaluation stage
   also supports a minimum-recall target and an absolute or rate-based alert capacity, so a
   precision target is no longer the only way to state the review constraint. The target
   precision itself, and a confirmed business cost ratio, are still open.
3. **Promotion gate.** Minimum PR-AUC / drift criteria required to move an alias to
   `champion`. *Blocks M5.*
4. **Feature snapshot policy.** Whether to persist raw inputs by default (storage + privacy
   cost vs. drift-analysis value). *Blocks M6.*
5. **Auth on the API.** ✅ **Resolved for the current scope.** Health, readiness and scoring
   are public; model metadata, prediction history and review require a single administrative
   bearer token from `API_ADMIN_TOKEN`. The routes fail closed when it is unset. This is not a
   complete identity/authorization system: there is no per-user identity, role model, token
   rotation, rate limiting or audit trail beyond access logs. Those remain required before
   exposing scoring to untrusted clients or the review workflow to multiple analysts.

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
| `pytest`, `pytest-cov`, `httpx`, `ruff`, `mypy`, `packaging`, `pandas-stubs`, `matplotlib` | extra `dev` | Never imported by `src/`. `httpx` drives the API tests through `httpx.ASGITransport`; `packaging` is used by the environment smoke test to verify installed versions against the declared ranges. `pandas-stubs` exists because pandas ships no inline types: without it `mypy --strict` treats every `DataFrame` as `Any`, which would make the data layer's annotations decorative. `matplotlib` renders the EDA figures from `notebooks/eda.py`; it stays out of `dependencies` because no module under `src/` may import a plotting library (AGENTS.md §2), which also keeps the serving image free of it. |

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
