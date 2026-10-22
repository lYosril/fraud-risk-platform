# Implementation Roadmap

Milestones are ordered so that each one is independently demonstrable and no milestone
depends on an unfinished later one. **M0, M1 and M3 are complete. M4 is implemented for the
comparison, frozen final-evaluation path and one real MLflow-tracked run, but still lacks a
two-run history, stratified CV and the reliability/Brier comparison. M5 is in progress: the
bundle contract, XGBoost SHAP layer and reusable inference service are implemented; registry
alias resolution and the promotion gate are not. M6 is partially implemented: the versioned
health/readiness/model-metadata/single-prediction/batch-prediction/history-list/summary/review
routes, per-item batch errors, request/batch limits, CORS policy, safe error mapping, prediction
persistence and admin bearer-token auth are in place, while request-id propagation and feedback
capture remain open. M2 is partially implemented (predictions table, migration and repository;
feedback/drift tables open). M7 is partially implemented (overview, scoring, filtered history,
explanation and analyst-review views; drift history open). M8–M9 are not started.**

Test each milestone's exit criteria before starting the next one. No milestone is "done"
because code exists â€” it is done when its exit criteria pass and the evidence is recorded.

---

## M0 â€” Repository foundation âœ… *complete*

**Deliverables**

- Repository skeleton, `src/fraud_risk/` package with documented module boundaries.
- `pyproject.toml` (dependencies, extras, ruff/mypy/pytest/coverage config).
- `.gitignore`, `.gitattributes`, `.env.example`, `.python-version`.
- `docs/ARCHITECTURE.md`, `docs/ROADMAP.md`, `README.md`, `AGENTS.md`.
- `docker-compose.yml` (PostgreSQL only), `.github/workflows/ci.yml`.
- `fraud_risk.config.Settings` and a small `tests/` suite covering it.
- Provisioned Python 3.11 environment (documented for PowerShell and Bash) and
  `tests/test_smoke_imports.py`, which checks the architecture's packages are declared,
  installed inside their declared ranges, and importable.

**Exit criteria**

- [x] `python -m compileall src tests` succeeds.
- [x] `ruff format --check .` and `ruff check .` report no findings.
- [x] `mypy` reports `Success` under `strict = true`.
- [x] `pytest -q` passes (`34 passed` on both Windows 3.11.9 and Linux 3.11.17).
- [x] The declared dependency set resolves and the editable build works
      (`pip install --dry-run -e ".[dev]"`, then installed: 130 packages, exit 0).
- [x] `.gitignore` behaviour verified: `.venv/`, `.env`, `mlruns/`, `data/raw/*` and
      `artifacts/*` are ignored while the `.gitkeep` placeholders are tracked.
- [x] The documented light setup path works (`9 passed, 25 deselected` with only
      `pydantic-settings packaging pytest ruff mypy` installed).
- [ ] CI workflow executed at least once on GitHub â€” *not run yet; the workflow file has
      never been pushed.*

---

## M1 â€” Data ingestion and validation

**Deliverables**

- **Dataset decided and documented** (ARCHITECTURE Â§13.1 resolved): data card in
  `docs/DATA_CARD.md`, operator workflow in `data/README.md`, and
  `python -m fraud_risk.data` for presence, header, column and row-width checking.
- `data.ingestion` â€” **loading done**: `load_csv` returns the file as stored (no coercion,
  no cleaning) and raises typed errors for missing, empty, non-UTF-8, duplicate-header and
  ragged files. The dataset fingerprint (hash + row count + time range) is still pending.
- `data.validation` â€” **contract done**: required columns, duplicate column names, empty
  data, per-column null counts, numeric/finite non-target columns, and `Class` restricted
  to `0`/`1`, plus a report of row/feature/missing/duplicate/class counts and proportions.
  Typed errors live in `data.errors`. Value-range checks are not implemented yet; the API
  request schema rejects `is_fraud`/`Class`, while a dedicated dataset-level inference guard
  remains pending.
- CLI entry point â€” **done**: `python -m fraud_risk.data` now runs the contract check after
  the presence/shape check, so a failure is reproducible from the command line.
- **Unit tests done** for every rule above, including the failure cases
  (`tests/test_ingestion.py`, `tests/test_validation.py`).

**Exit criteria**

- Validating a known-good fixture passes; a corrupted fixture fails with an actionable message.
  *Met: the tracked fixture validates with exit 0; corrupt fixtures exit 1 with a typed error
  naming the column and reason.*
- The validation report is written to a git-ignored location and is reproducible from the hash.
  *Not met: the report is printed, not persisted, and the fingerprint is not implemented.*

**Depends on:** M0.

---

## M2 â€” Persistence layer

**Deliverables**

- `db.base` (`DeclarativeBase` + naming convention for constraints), `db.models`
  (`predictions`, `feedback_labels`, `drift_reports`), `db.session` (engine + session factory).
- Alembic initialised; first migration creating the three tables, plus a downgrade path.
- Repository functions for the write/read paths the API actually needs (no premature DAO layer).

**Exit criteria**

- `alembic upgrade head` against the Compose PostgreSQL creates the schema; `downgrade base`
  removes it cleanly.
- Round-trip test: insert a prediction row, read it back with JSONB fields intact.
- Verified against PostgreSQL, not SQLite.

**Depends on:** M0.

---

## M3 â€” Exploratory analysis and feature engineering

**Deliverables**

- EDA â€” **done**: `notebooks/eda.py` loads the file through the ingestion/validation layer,
  takes every number from `features.eda`, and writes `summary.md`, `summary.json`, four CSV
  tables and six figures to the git-ignored `reports/eda/`. A script rather than a notebook so
  the tests execute it end to end. It covers shape/schema, duplicates, missing values, class
  balance, `Amount`, `Time`, per-feature statistics, feature/target associations, and the
  limitations and leakage risks. A non-reference input is labelled synthetic or custom in every
  artefact. No transform is fitted at this stage.
- `features.engineering`: one reusable `ColumnTransformer` producing the model matrix â€”
  **done**: `V1..V28` + `Amount`, `Time` and `Class` excluded with the reasons in the schema,
  raw values for tree models and a training-fitted `StandardScaler` for linear ones.
- Leakage controls that are unit-tested, not just documented â€” **done** for the split and the
  transform: disjointness, timestamp-boundary integrity, label separation, and training-only
  fitting statistics.
- Fitted transformer serialisation (joblib) embedded in the artifact bundle (D3) â€” **done**:
  the fitted `preprocess` step travels with the model and is the only transform used by
  `models.inference`; a dedicated cross-process equality test is still desirable.
- Early precision/recall expectations move to M4: they require a trained, calibrated model,
  and no number may be invented before one exists.

**Exit criteria**

- The transform is fitted once and applied identically in a second process, producing
  identical output for identical input (train/serve skew test).
  *Partially met: fitted on training rows only, serialised inside the bundle and reused
  unchanged at serving time, with the training-only statistics unit-tested. A dedicated
  cross-process equality test remains.*
- The EDA run is reproducible from a clean install: `python notebooks/eda.py` regenerates
  every artefact from the input file and labels a non-reference input as synthetic or custom.
  *Met for both the tracked fixture and the local reference file; reproducible from the
  file, not from a persisted hash (the dataset fingerprint is still an M1 item).*

**Depends on:** M1.

---

## M4 â€” Model training and evaluation

**Deliverables**

- `models.splitting` â€” **done** (delivered with the preprocessing stage): temporal 60/20/20
  split (default for the time-ordered file, boundaries pushed to the next distinct second) and
  a seed-reproducible stratified alternative, plus a ~2.5 KB split record with no row-level
  values.
- `models.training` â€” **done**: a bounded grid of calibrated logistic regression, random
  forest and
  XGBoost candidates on the shared feature pipeline; training-only class reweighting
  (`class_weight` / `scale_pos_weight`); each threshold chosen on validation expected cost;
  the winner selected on that cost (ties by validation PR-AUC) and the test split scored once;
  local `model.joblib` / `comparison.json` / `metrics.json` / `run.json` artifacts. Remaining:
  stratified CV, the reliability/Brier calibration comparison, and a second comparable MLflow
  run (one real run exists; it is not yet a history).
- `fraud_risk.tracking` â€” **done**: configurable MLflow URI/experiment, a deterministic
  `run_fingerprint` over seed/split/schema/config/git revision, params/tags/metrics/artifacts,
  and explicit `FAILED` status with failure tags when training raises. The four local files are
  logged as artifacts; no model is logged, registered or deployed by training.
- `models.evaluation` â€” **done**: precision, recall, F1, confusion matrix, PR-AUC,
  ROC-AUC, false-positive rate, alert count/rate, per-class metrics, the majority-class
  baseline, the validation-only cost-optimal threshold rule and the explicit separation of
  probabilities from binary decisions, with the cost impact of false negatives versus false
  positives. Also **done**: validation-only `MinRecallPolicy` and `AlertCapacityPolicy`
  (absolute alert count or share of rows), with deterministic tie-breaking and tied-score
  groups never split. Remaining: reliability curve / Brier score, the calibration comparison,
  richer error analysis and the three-way review/block operating points.
- `models.final_evaluation` â€” **done**: consumes the comparison bundle, re-selects the
  operating threshold on validation under expected cost, minimum recall or alert capacity,
  copies the model and writes `decision.json` before test, then scores test once and writes
  `metrics.json` / `run.json`. Probabilities and decisions are reported separately, and the
  model SHA-256, policy, split, cost weights and library versions are recorded. This is a
  repeat test evaluation because the comparison stage already scored the test split; the
  split is therefore not a pristine holdout.
- MLflow logging of params, metrics, artifacts, git revision and the split definition â€”
  **done** for the training CLI. A dataset content fingerprint is still pending with the M1
  data layer, so split-position digests and row counts are recorded instead of inventing one.

**Exit criteria**

- At least two real runs are visible in the MLflow UI with comparable metrics.
- Reported metrics come from actual runs; the run id is recorded next to every number.
- The threshold policy (ARCHITECTURE Â§6) is tuned on validation data and frozen with the model.

The local comparison and final-evaluation artifacts satisfy the threshold-freezing criterion
for a binary operating cut. The MLflow criterion is **partially met**: one real training run
(`758f8670c78744159032d4fcabfff602`, experiment `fraud-detection`, status `FINISHED`) is in the
local SQLite store, but there is no second comparable run and therefore no run history in the
sense the criterion requires. The test split has been scored by the comparison stage and again
by the final evaluation, so it cannot be used for further honest tuning. No model is
registered by the tracking integration.

**Depends on:** M2 (for the MLflow/Postgres option) and M3.

---

## M5 â€” Artifact management and inference

**Deliverables**

- Artifact bundle format: fitted transformer + calibrated model + metadata (schema version,
  feature names, thresholds, metrics, git revision, training timestamp).
- `models.registry`: publish a bundle, resolve a named alias (`champion`), list versions.
  MLflow registry aliases back this, with a filesystem store for the local PoC.
- `models.explainability` — **done for the selected XGBoost bundle**: cached
  `shap.TreeExplainer` per calibration fold, additivity-checked pre-calibration tree-margin
  contributions, global mean-absolute importance, top-k per-transaction explanations with
  exact raw/model input values, strict JSON/CSV/Markdown reports and a plotting script under
  the git-ignored `reports/explainability/`. Unsupported model families fail loudly. The
  per-request path is now wired through `models.inference`; the registry remains pending.
- `models.inference` â€” **done**: lazy artifact loading and caching, exact feature
  schema/order validation, fitted-preprocessor compatibility checks, one-row `predict_proba`,
  bundle threshold, model version/SHA and optional SHAP explanation. Scoring never fits or
  retrains. Missing/corrupt/incompatible artifacts raise a clear model error.
- `api.schemas` â€” **done**: Pydantic v2 request/response and error contracts are used by the
  versioned FastAPI routes, rejecting label/unknown fields and constraining input/output
  ranges. Read-back/feedback schemas remain M6 work.
- `models.registry`, persistence and the documented promotion gate remain pending.

**Exit criteria**

- Loading the configured bundle and scoring a fixture transaction returns a calibrated score:
  **met** by the synthetic inference tests; the service also loads the local real artifact.
- Loading a *missing* bundle raises a clear error the API can turn into a 503: **met**
  (`InferenceModelError`; the implemented API maps it to a non-leaking 503).
- Explanations are produced for a single transaction well within an interactive latency budget:
  **met by the existing SHAP layer and wired through the new service on the selected XGBoost
  bundle**; the measured 11.4 ms median for one explanation excludes HTTP, persistence and the
  transaction's own feature construction.

**Status:** the inference service and API contracts are implemented and tested. Registry alias
resolution, persistence and the promotion gate remain pending.

**Depends on:** M4.

---

## M6 â€” FastAPI application

**Deliverables**

- `api.main` â€” **done for the current surface**: application factory, process-wide settings and
  one lazy inference service, versioned router, CORS policy, request-body limit middleware,
  access logging, exception mapping, and production removal of `/docs`/`/openapi.json`. Lifespan
  engine disposal is done; request-id propagation remains pending.
- `api.schemas` â€” **done**: Pydantic v2 health, availability-only readiness,
  model-metadata, single-prediction, batch-request/batch-result, persisted-record, review and
  error models; validation rejects label fields, unknown fields and non-numeric feature values.
- Routes under `Settings.api_prefix` â€” **done**: `GET /health` (liveness only),
  `GET /ready` (loads/validates the model; `200`/`503`, availability only), `GET /model`
  (admin-only identity and feature contract), `POST /predict` (one transaction),
  `POST /predict/batch` (bounded list with per-item success/error results),
  `GET /predictions` (admin-only bounded, filterable history page), `GET /predictions/summary`
  (admin-only aggregate counts and risk bins), `GET /predictions/{id}` and
  `PATCH /predictions/{id}/review` (admin-only read-back/review). Review updates enforce a
  pending-to-terminal transition and reject invalid updates with `409`.
  Model loading, scoring and repository calls run in Starlette's thread pool behind async
  handlers.
- Persistence of every prediction; documented, non-leaking error responses. **Done**:
  successful predictions are written through a request-scoped SQLAlchemy session, writes roll
  back on failure, and read/review routes return the persisted record without exposing the
  opt-in feature snapshot.
- Administrative authentication. **Done for the current scope**: `API_ADMIN_TOKEN` protects
  model metadata, history and review with a constant-time bearer-token check. Protected routes
  fail closed with `503` when the token is unset.
- Resolve the inherited `starlette.testclient` / `httpx` deprecation first. **Resolved for the
  API suite** by testing through `httpx.ASGITransport` and dependency overrides, with no
  `TestClient` dependence and no deprecation warning.

**Exit criteria**

- Integration tests run clean, with no `httpx`/`httpx2` deprecation warning. **Met for
  `tests/test_api.py`.**
- `uvicorn` serves the OpenAPI docs; every endpoint has a documented success and failure case.
  **OpenAPI schema coverage is tested; local startup, `/docs`, single/batch examples, auth and
  the limits are documented.**
- Integration tests via `httpx.ASGITransport` cover happy path, validation failure, unknown id,
  and the no-model-loaded path. **Single and batch happy paths, per-item validation failure,
  empty/oversized batches, body limits, missing-model 503, CORS preflight, admin auth failure,
  unknown-id 404, history/review and production-doc removal are covered.**
- A scored transaction is readable back from PostgreSQL and matches the response body.
  **Implemented and covered by the PostgreSQL integration suite; re-run it against Compose
  PostgreSQL before declaring the milestone complete.**

**Status:** the versioned HTTP surface, bounded batch scoring, prediction persistence,
admin-protected read-back/review, safe error mapping and access logging are implemented and
tested. Request-id propagation, feedback capture, per-user authorization, token rotation and
rate limiting remain open, so the milestone is not complete.

**Depends on:** M5, M2.

---

## M7 â€” Streamlit dashboard

**Deliverables**

- `dashboard.app` reading from the API/DB (never importing model internals).
- Views: traffic over time, score distribution with threshold markers, decision mix,
  review-queue table, per-prediction explanation drill-down, drift history.
- Plotly charts; API base URL from settings.

**Implemented so far:** `dashboard.api_client` (typed HTTP client, admin bearer token from
`Settings`, filter query serialization, review PATCH, safe failure states),
`dashboard.transformations` (counts, risk bins, frames, predicted-label vs analyst-outcome
display), `dashboard.charts` (Plotly risk/decision/review/volume/explanation figures) and
`dashboard.app` (overview, filtered history, analyst-review workflow, transaction scoring and
explanation view). The API added `GET /predictions` filtering, `GET /predictions/summary` and
terminal review transitions for real stored traffic. The per-prediction drill-down route/UI and
drift history remain open; therefore M7 is not complete.

**Exit criteria**

- Runs with `streamlit run` against a locally seeded database and shows real stored rows.
- No hardcoded connection strings; the dashboard starts (with a clear message) when the API
  is unreachable.

**Depends on:** M6 (seeded data).

---

## M8 â€” Monitoring

**Deliverables**

- `monitoring.drift`: PSI (per feature, reference = training distribution), KS statistic,
  score-distribution shift, volume/decision-mix tracking, and delayed-label performance.
- Persist each run to `drift_reports`; configurable warning/alert thresholds.
- Optional Evidently HTML report via the `monitoring` extra.
- A documented runbook: what each alert means and what to check first.

**Exit criteria**

- Unit tests with hand-constructed distributions where the expected PSI is known analytically.
- Injecting a shifted dataset produces a breach that is visible in `drift_reports` and in the
  dashboard; injecting an unshifted one does not.
- Thresholds are version-controlled, not hardcoded in the analysis script.

**Depends on:** M6, M7.

---

## M9 â€” CI/CD and containerised deployment

**Deliverables**

- Multi-stage `Dockerfile` (one image, several commands) and application services in
  `docker-compose.yml` (api, dashboard, optional mlflow, postgres).
- CI: lint â†’ type-check â†’ unit tests â†’ integration tests against a PostgreSQL service
  container â†’ build image. Coverage reported, not used as a gate until it is meaningful.
- Deployment notes (reverse proxy, secrets injection, migration step on release).

**Exit criteria**

- `docker compose up --build` brings up a working stack from a clean clone (README-documented).
- CI is green on a real commit â€” not just locally.
- Secrets come from the environment at runtime; the image contains none.

**Depends on:** M6, M7, M8.

---

## Backlog (explicitly not scheduled)

- Adopt a lock file (for example `uv.lock` or a `pip-compile`d constraints file) so CI and
  production resolve byte-identical dependency versions.
- Per-user authentication and role-based authorization, token rotation/revocation and request
  rate limiting for the API (the current admin bearer token is a single shared secret; see
  ARCHITECTURE §13.5).
- Model card per promoted version.
- Batch scoring endpoint.
- Automatic drift-triggered retraining (only after M8 has historical evidence).
- Streaming ingestion.
- Kubernetes manifests / cloud IaC.
