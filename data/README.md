# `data/` — local datasets (never committed)

Datasets are **not** part of this repository. Everything here except this README, the
`.gitkeep` placeholders and `data/sample/` is git-ignored, and the published dataset file is
ignored by name as well, so it cannot be committed even if it is dropped in the wrong place.

| Directory | Holds | Git-ignored |
|-----------|-------|-------------|
| `data/raw/` | The dataset exactly as published. Read-only for our code. | Yes |
| `data/interim/` | Derived, intermediate tables. | Yes |
| `data/processed/` | Model-ready output of the feature pipeline. | Yes |
| `data/sample/` | Tiny **synthetic** fixtures used by the tests. | No (tracked) |

Only `data/raw/` is used today. The other directories are reserved for milestone M1 and later.

---

## The reference dataset

**Credit Card Fraud Detection** — Machine Learning Group, Université Libre de Bruxelles (ULB),
published on Kaggle: <https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>

- File: **`creditcard.csv`** (the Kaggle distribution also offers a `.zip` of the same CSV).
- 31 columns: `Time`, `V1`…`V28`, `Amount`, `Class`.
- Target: **`Class`** — `1` = fraud, `0` = legitimate.
- Provenance, licence, schema, published class-imbalance figures and known limitations are in
  **[`docs/DATA_CARD.md`](../docs/DATA_CARD.md)**. Read that before using the data for anything.

### Obtain it — option A: manual download (recommended)

1. Open <https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud> in a browser.
2. Sign in and accept the dataset terms if the page asks you to. That is a Kaggle account
   requirement; nothing in this repository automates it.
3. Download `creditcard.csv` (or download the archive and extract `creditcard.csv`).
4. Place it exactly here:

   ```
   data/raw/creditcard.csv
   ```

**This project does not scrape the Kaggle website and does not bypass authentication.** There
is no fallback that "finds" the data elsewhere: if the file is absent, the tooling says so.

### Obtain it — option B: official Kaggle CLI (only if already configured)

If — and only if — the official Kaggle CLI is installed and your credentials are already
configured, the helper can drive it for you:

```bash
python -m pip install kaggle          # optional; not a dependency of this project

python -m fraud_risk.data --download  # downloads into data/raw/ and unzips it
```

Credentials are read by the Kaggle CLI itself, from the standard locations:

- `~/.kaggle/kaggle.json` (`%USERPROFILE%\.kaggle\kaggle.json` on Windows), or
- the `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables.

The helper only checks that credentials **exist**; it never reads, logs or passes them on a
command line. If the CLI or the credentials are missing it exits with instructions instead of
guessing. An existing `creditcard.csv` is never overwritten by `--download`.

> Never commit `kaggle.json`. It is a credential; this repository ignores it via the
> `secrets/` and `.env` rules, and it should live outside the repository regardless.

### Verify and validate what you have

```bash
python -m fraud_risk.data                          # checks data/raw/creditcard.csv
python -m fraud_risk.data --path path/to/other.csv # checks any local file
```

The command runs in two stages and never prints a cell value, so its output is safe to paste
into a bug report:

1. **Presence and shape** — does the file exist, is it readable, does it have the documented
   header and column set, how many data rows does it have.
2. **Contract validation** — loads the file and checks the columns, nulls, numeric values and
   labels (see the table below), then prints a summary: rows, features, missing values,
   duplicate rows, and the count and proportion of each class.

| Exit code | Meaning |
|-----------|---------|
| `0` | The file is present, readable and passes every contract check. |
| `1` | A file exists but is damaged, unreadable, has the wrong columns, or violates the data contract. The offending column and reason are named. |
| `2` | The file is missing. |
| `3` | `--download` was requested but could not be completed. |

Example output for the synthetic fixture (8 rows, 2 of them fraudulent):

```
Validation: OK
  path       : data\sample\creditcard.synthetic.csv
  rows       : 8
  features   : 30 (+ target Class)
  missing    : 0 values
  duplicates : 0 rows
  class 0    : 6 (75.0000%)
  class 1    : 2 (25.0000%)
```

What is checked, and what is deliberately not:

| Checked | Not checked (by design) |
|---------|-------------------------|
| The 31 documented columns are present, each exactly once. | Value ranges or statistical plausibility. |
| At least one data row. | Whether both classes are present (reported, not enforced). |
| No null/NaN values anywhere in those 31 columns. | Duplicate rows — they are counted and reported, never deleted. |
| Every non-target column is numeric and finite (no `inf`). | Rows are never dropped and values are never imputed. |
| `Class` contains only `0` and `1`. | The dataset fingerprint (still pending in M1). |
| Rows are not silently shifted by a malformed CSV. | Leakage on an inference path (M6). |

Validation only observes. Cleaning — imputation, dropping, scaling — belongs to the feature
pipeline, must be fitted on training data only, and is not done here.

Beyond the documented 31 columns, a file with *extra* columns is accepted and those columns are
ignored by the contract.

---

## `data/sample/creditcard.synthetic.csv`

A deliberately tiny fixture: **8 rows, 31 columns, 2 rows labelled fraudulent**. Its values are
generated by a fixed formula. It is **not** a sample of the real dataset, is not derived from
it, and carries no statistical meaning whatsoever. It exists so that the presence check, the
contract validation and their tests have a file with the right shape to work against — and so
that `python -m fraud_risk.data --path ...` can be demonstrated without the real data.

Tests build their own malformed variants in temporary directories; nothing in `data/sample/`
is needed to run the suite.
