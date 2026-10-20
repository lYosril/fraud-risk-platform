# Data card — Credit Card Fraud Detection (ULB)

This card describes the dataset this platform is designed around. It is a **design and
provenance document**, not a report of results: no statistic on this page was computed in this
repository. The dataset itself is never committed — see [`data/README.md`](../data/README.md).

---

## 1. Identity

| Field | Value |
|-------|-------|
| Name | Credit Card Fraud Detection |
| Publisher | Machine Learning Group, Université Libre de Bruxelles (ULB) |
| Distribution | Kaggle — <https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud> |
| File | `creditcard.csv` (also distributed inside a `.zip`) |
| Rows | 284,807 transactions *(published)* |
| Columns | 31: `Time`, `V1`…`V28`, `Amount`, `Class` |
| Target | `Class` — `1` = fraud, `0` = legitimate |
| Nature | Anonymised, PCA-transformed card transactions |

An OpenML mirror exists (dataset 1597). It is a separate distribution with its own terms; use
the Kaggle copy this project documents unless you have a reason not to.

## 2. Source and access

The data is obtained **by the operator**, not by the build:

1. Manually from the Kaggle dataset page, placed at `data/raw/creditcard.csv`; or
2. Through the **official Kaggle CLI**, if it is already installed and authenticated.

Nothing in this repository scrapes the dataset page, harvests a token, or bypasses an
authentication step. Repository tooling never reads Kaggle credentials: it only checks whether
they exist and lets the official CLI use them.

## 3. Licence and usage

The Kaggle dataset page lists the licence as **DbCL v1.0 (Database Contents License)**. Confirm
the current terms on the dataset page before redistributing or using the data commercially —
the licence is the publisher's to change, and this document is not legal advice.

Usage considerations that follow from the dataset's nature:

- **Do not redistribute.** This repository does not, and must not. The file is git-ignored by
  path *and* by name.
- **Cite the source.** The ULB Machine Learning Group; link the Kaggle dataset in any write-up.
- **Respect anonymisation.** `V1`…`V28` are published as PCA components precisely to make the
  original transaction context unrecoverable. Do not attempt to re-identify cardholders.
- **Educational / research scope.** This is a portfolio project; it is not a production
  fraud-decision system and the licence does not make it one.

## 4. Collection and provenance

Published description (paraphrased, not independently verified here):

- Transactions made by **European cardholders in September 2013**.
- A span of **two days**, i.e. roughly 48 hours of activity.
- `V1`…`V28` are the output of a **PCA transformation**; the publisher withheld the original
  features for confidentiality. Only `Time` and `Amount` are untransformed.
- `Time` is the **number of seconds elapsed since the first transaction in the file**. It is an
  offset, *not* a wall-clock timestamp and not a calendar date.
- No currency is documented for `Amount`; do not assume one.

## 5. Schema

| Column | Type | Meaning |
|--------|------|---------|
| `Time` | integer, seconds | Seconds since the first transaction in the file. |
| `V1` … `V28` | float | PCA components. Anonymised; no business meaning. |
| `Amount` | float | Transaction amount. Currency undocumented. |
| `Class` | integer, `0`/`1` | **Target.** `1` = fraud, `0` = legitimate. |

Column order as published: `Time`, `V1`…`V28`, `Amount`, `Class` (31 columns). The presence
check compares columns **by name**, so a reordered but otherwise correct file is accepted.

### Mapping to the canonical contract

`docs/ARCHITECTURE.md` §5.1 defines a canonical transaction record that is a superset of this
file. The mapping is:

| Canonical field | Here |
|-----------------|------|
| `is_fraud` | `Class` |
| `amount` | `Amount` |
| `event_time` | Derived from `Time` + a chosen epoch start. Never a real timestamp. |
| `features` | `V1`…`V28` |
| `currency`, `merchant_category`, `account_ref` | Absent. |

`is_fraud` is **labels-only**: it must never reach an inference path. That guard belongs to
`data.validation` (M1) and is not implemented yet.

## 6. Class imbalance

Published figures: **492 fraudulent transactions out of 284,807** — **0.172 %** positive.

Consequences that shape the rest of the design:

- **Accuracy is meaningless** here: predicting "legitimate" for everything scores 99.83 %.
  PR-AUC (average precision) is the headline metric; ROC-AUC is reported but never the headline.
- Resampling, class weighting and threshold selection must be fitted on training data only.
- **Calibration precedes thresholding**: the three-way `allow`/`review`/`block` policy is applied
  to calibrated probabilities, not to raw scores.
- A stratified or temporally-aware split is required; a random split on time-ordered data would
  inflate the metrics.

## 7. Known limitations

- **PCA components are not human-interpretable.** SHAP attributions on `V1`…`V28` can rank
  feature importance but cannot be explained in business terms. This limitation is recorded
  deliberately in `docs/ARCHITECTURE.md` §7 and should be stated wherever explanations are shown.
- **Two days, one market, 2013.** No seasonality, no long-term drift, no geographic or product
  diversity. Drift monitoring on this data is a mechanism demonstration, not evidence about a
  live population.
- **No operational context**: no merchant category, device, channel, customer history or
  account references. The canonical record keeps room for these so a richer source can replace
  the reference dataset without changing the API contract.
- **`Amount` has no documented currency**, so amount-based features are unitless until a source
  defines one.
- **Duplicate rows** have been reported by third parties. Check for them during training-time
  analysis (M3) and record the decision; do not drop rows silently.
- **Anonymisation is not a guarantee.** The dataset is old; treat it as sensitive regardless.

## 8. Intended use in this project

- Development and demonstration of the full pipeline: validation → features → training →
  calibration → serving → monitoring.
- Benchmarks that are comparable with published work on the same file.

**Not** intended: production fraud decisions, claims about a specific bank's loss rates, or any
statement that a metric measured here transfers to another population.

## 9. How this repository handles the data

| Concern | Handling |
|---------|----------|
| Committing the data | `.gitignore` blocks `data/raw/*`, `data/processed/*`, `data/interim/*`, and the name `creditcard.csv`/`creditcard.zip` anywhere. |
| Locating the file | `fraud_risk.data.dataset` declares the filename, the 31 columns and the target; `python -m fraud_risk.data` reports presence, size, row count, header and column set. |
| Printing the data | The check never emits a cell value — only counts and column names. |
| Acquiring the file | Manual download (default) or the official Kaggle CLI when credentials are already configured. Explicit failures, never a fabricated success. |
| Tests | A tiny **synthetic** fixture (`data/sample/creditcard.synthetic.csv`, 8 rows) and temporary malformed files. The real dataset is never downloaded or asserted against in the test suite. |

## 10. Verification status

Recorded honestly, per `AGENTS.md`:

- **Verified in this repository:** the expected column names, the target name and the presence
  check are exercised by tests against synthetic files.
- **Not verified here:** no copy of the real dataset is present, so none of the published
  figures above (row count, fraud count, imbalance) was recomputed. They are reproduced as
  published and are labelled as such.
- **Not implemented yet:** dtype/range/nullability validation, the dataset fingerprint, and the
  leakage guard. Those are M1 deliverables (`docs/ROADMAP.md`).
