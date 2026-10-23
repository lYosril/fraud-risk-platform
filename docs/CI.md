# Continuous integration

The workflow is [`.github/workflows/ci.yml`](../.github/workflows/ci.yml). It is deliberately
limited to static checks and tests. It does not deploy the application, publish images, call
Kaggle or any external API, or use a production database.

## Triggers and permissions

- Every `pull_request`.
- Every push to the primary branch, `main`.
- `permissions: contents: read` at workflow level.
- No repository secrets are referenced and no credentials are stored in the workflow.
- `concurrency` cancels superseded runs on the same ref.

## Required checks

Configure these two job names as required status checks for `main` in GitHub branch protection:

| Required check | Job ID | What it runs |
| --- | --- | --- |
| `Lint, type-check and unit tests` | `quality` | Python 3.11 from `.python-version`, `pip install -e ".[dev]"`, `ruff check .`, `ruff format --check .`, `mypy`, `python -m pytest -q -m "not integration"` |
| `PostgreSQL integration tests` | `integration` | The same Python/install setup, then `python -m pytest -q -m integration` against an ephemeral PostgreSQL service |

The integration job runs only after the quality job succeeds (`needs: quality`).

## Integration service

The integration job uses the official `postgres:16-alpine` service container. It is job-local
and ephemeral:

- database `fraud_test`, user `fraud`;
- `POSTGRES_HOST_AUTH_METHOD=trust`, so no password or production credential is placed in the
  workflow; the service is reachable only from that job's runner;
- a `pg_isready` health check gates the test step;
- the test fixture runs `alembic upgrade head` against `TEST_DATABASE_URL` before the
  integration tests.

No model artifact, Kaggle credential, external API or production database is required.

## Dependency installation and caching

The repository's single dependency definition is `pyproject.toml`; CI installs it with pip:

```bash
python -m pip install -e ".[dev]"
```

`actions/setup-python` caches pip downloads keyed on `pyproject.toml`. There is no separate
requirements file or lock file to maintain.

## Local equivalents

Run the same checks from a Python 3.11 environment:

```bash
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
mypy
python -m pytest -q -m "not integration"
```

For the integration job, start the local PostgreSQL service and point `TEST_DATABASE_URL` at
the dedicated test database (never the `fraud` database):

```powershell
docker compose up -d postgres
$env:TEST_DATABASE_URL="postgresql+psycopg://fraud:change-me@127.0.0.1:55432/fraud_test_monitoring"
python -m pytest -q -m integration
```

In Bash, use `export TEST_DATABASE_URL=...` instead. Use the port your local Compose
environment exposes; this worktree used `55432`.

## Status

The workflow file has been syntax-checked locally, and the commands from both jobs have been
run locally, including the integration tests against a temporary `postgres:16-alpine` container
using trust authentication. The workflow itself has not yet run on GitHub. Do not mark the CI
milestone complete until a real pull request or push to `main` is green.
