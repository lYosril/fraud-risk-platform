# AGENTS.md — working conventions for this repository

Instructions for any automated agent (or human) making changes here. Read
`docs/ARCHITECTURE.md` and `docs/ROADMAP.md` first: they define the intended design and the
current implementation status. **If this file and the code disagree, the code wins — and the
disagreement is a bug in the code or in this file. Fix one of them.**

---

## 1. Ground rules

1. **Nothing is "tested" until it has been run.** Never write "verified", "works", "passes"
   or "tested" unless you executed that exact command in this session. If a check could not
   be run (missing dependency, no database, no network), say so explicitly and say why.
2. **Never present synthetic, mocked or placeholder output as a real result.** No fabricated
   metrics, no invented PR-AUC, no "example output" that looks like a real evaluation. Label
   illustrative numbers as illustrative, or omit them.
3. **Never commit secrets.** No credentials, tokens, `.env` files, connection strings with
   real passwords, or private keys. `.env.example` holds placeholders only.
4. **Never commit transaction data or model artifacts.** `data/raw`, `data/interim`,
   `data/processed`, `artifacts/` and `mlruns/` are git-ignored on purpose. Only tiny
   synthetic or public-sample fixtures may live in `data/sample/`.
5. **Never hardcode configuration.** Anything environment-specific goes through
   `fraud_risk.config.Settings`.
6. **Do not invent scope.** Implement the milestone you were asked for. If something is
   missing, note it in `docs/ROADMAP.md` rather than building it unasked.
7. **Fail loudly.** Validation errors raise; the API returns a documented error shape. No
   bare `except Exception: pass` and no silent coercion of bad input.

---

## 2. Layout and module boundaries

```
src/fraud_risk/
  config.py        Settings — the ONLY place environment variables are read
  data/            ingestion + validation
  features/        EDA helpers + the fitted feature transformer
  models/          training, evaluation, registry (artifacts), inference
  db/              SQLAlchemy models, session/engine, Alembic migrations
  api/             FastAPI app, routes, Pydantic schemas
  dashboard/       Streamlit + Plotly (reads via the API/DB, never model internals)
  monitoring/      drift statistics, reports, delayed-label performance
tests/             mirrors the package layout
docs/              architecture + roadmap
notebooks/         exploration only; never imported by src
```

Dependency direction is strictly one-way
(`api`/`dashboard` → `models.inference` → `models.registry` → `features` → `data`, with `db`
and `monitoring` at the leaves). **Importing upwards, or a cycle, is a defect.** Never import
plotting or Streamlit code from `src/fraud_risk/data`, `features`, `models` or `api`.

Adding a module, package or dependency means: update `docs/ARCHITECTURE.md` §3 if a
responsibility moved, and update the status column if a milestone advanced.

---

## 3. Python style

- Target **Python 3.11**. Do not use syntax newer than 3.11 (no 3.12/3.13-only features).
- `from __future__ import annotations` at the top of modules that will contain logic.
- **Type hints on every public function and method**, including `-> None`. `mypy --strict`
  must stay clean; add a narrowly-scoped override rather than loosening global settings.
- Line length 100, formatted with `ruff format`. Do not hand-align to a different width.
- Prefer `pathlib.Path` over `os.path`. Use timezone-aware datetimes (`DTZ` is enforced).
- Use structured logging (`logging.getLogger(__name__)`), never `print` in `src/` outside the
  dashboard.
- Dataclasses for plain data holders; Pydantic only at trust boundaries (API schemas,
  settings, artifact metadata).
- **Avoid unnecessary abstraction.** No interface with one implementation, no repository
  layer for a single query, no plugin system before the second plugin. Three similar lines
  beat a premature helper.
- Frame/chunk large transformations explicitly; never load a full production dataset into
  memory inside a request handler.

## 4. ML-specific rules

- **No leakage, ever.** The label column (`is_fraud`) must not reach any inference path, and
  every fitted transform must be fitted on training data only. A leaked feature is worse than
  a missing one.
- **One shared transform.** Training and serving must use the same fitted transformer object,
  shipped inside the artifact bundle. Re-implementing preprocessing at serving time is
  forbidden.
- **Split before you fit, and split temporally** where the data is time-ordered. Random splits
  on time-ordered data inflate metrics and are treated as a bug.
- **Metrics:** PR-AUC (average precision) is primary. ROC-AUC is reported but never the
  headline. **Never report accuracy** on this problem, and never a metric you did not compute.
- **Record provenance:** every run logs params, metrics, the dataset fingerprint, the split
  definition, the git revision and the library versions.
- **Calibrate before thresholding.** A threshold compared against an uncalibrated score is
  meaningless.
- Promotion of a model alias is a deliberate reviewable step, never automatic.

## 5. Testing

- Every new behaviour gets a test in the same change. Bug fixes get a regression test.
- Fast unit tests are the default. Anything needing PostgreSQL, the API or a model artifact is
  marked `@pytest.mark.integration`; anything slow gets `@pytest.mark.slow`.
- Do **not** substitute SQLite for PostgreSQL in tests — behaviour differs, and that difference
  hides real bugs.
- Use small deterministic fixtures, not a real trained model.
- Flaky tests are fixed or deleted. Never add retries, sleeps, or `xfail` to paper over flakiness.

## 6. Commands

```bash
# environment
python -m venv .venv
source .venv/bin/activate          # PowerShell: .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

# checks (run all four before declaring a change complete)
ruff check .
ruff format --check .
mypy
pytest -q

# local infrastructure
docker compose up -d postgres
```

A change is complete only when `ruff check`, `ruff format --check`, `mypy` and `pytest` pass
on the exact code you are about to hand over — or when you state plainly which of them you
could not run and why.

## 7. Commits and branches

- Conventional Commits: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`, `build:`, `ci:`.
- Scope by component where useful: `feat(api): add /predict endpoint`.
- One logical change per commit. Update `docs/ROADMAP.md` checkboxes in the same commit that
  satisfies them — not before.
- Never commit directly to a long-lived branch with a red check.
- Do not rewrite history that others may have based work on.

## 8. Dependencies

- One dependency file: `pyproject.toml`. Never add `requirements.txt` alongside it.
- Use compatible-release ranges (`>=x,<y`), never unbounded `>=` on a library we rely on.
- Avoid a new dependency when the standard library or an existing dependency suffices. Adding
  a heavy ML/data dependency is a design decision — record the rationale in
  `docs/ARCHITECTURE.md` §12.
- Keep optional extras optional: nothing in `dependencies` should be used only by an extra.

## 9. Security and data hygiene

- The API is unauthenticated by design at this stage and must only be bound to localhost/private
  networks. Say so if you expose it. See `docs/ARCHITECTURE.md` §13.5.
- Never log full request payloads, card/account identifiers, or feature snapshots.
- Pseudo-anonymise account references; never handle a PAN.
- Keep query parameterisation via SQLAlchemy — no string-built SQL.
