# Delayed-label performance reporting

This document describes the routine performance workflow. It is deliberately separate from
both drift monitoring (`docs/MONITORING.md`) and the one-time frozen final evaluation under
`artifacts/final_evaluation/`.

## What it answers, and what it does not

It answers: *for these verified outcomes, at this recorded threshold, how did the scores that
actually served each row perform overall, per model version, and per calendar period?*

It does **not** infer labels from `predictions.review_status`, join a `feedback_labels` table
(not implemented), re-score rows with a newer model, or read the final test metrics. Analyst
review state is an incomplete, non-random operational signal; treating it as ground truth would
bias every metric. A verified outcome must be supplied explicitly in the evaluation extract.

## Input contract

The CSV must contain exactly these columns, in any order:

| Group | Columns |
|-------|---------|
| Model features | `V1` ... `V28`, `Amount` |
| Verified target | `is_fraud` (only `0` or `1`) |
| Served score | `risk_score` (finite, `[0, 1]`) |
| Model identity | `model_version` (non-empty, at most 128 characters) |
| Evaluation timestamp | `scored_at` (ISO-8601 with an explicit timezone offset) |

Extra columns are rejected rather than ignored. In particular:

- `review_status` is rejected with a message explaining that it is workflow state, not a
  confirmed outcome.
- The raw `Class` column is rejected; export the verified target as `is_fraud`.
- Duplicate column names, missing features, nulls, non-finite features, negative `Amount`,
  labels other than `0`/`1`, out-of-range scores, blank model versions and timezone-naive
  timestamps all fail loudly.

`risk_score` is the probability-like score stored at prediction time. It is never recomputed, so
one extract can contain several model versions and the report describes the scores that actually
served those rows.

## Command

```powershell
python -m fraud_risk.monitoring performance `
  --path data/evaluation/labeled_extract.csv `
  --threshold 0.5 `
  --label-source verified_chargebacks `
  --labels-verified `
  --period month
```

| Option | Meaning |
|--------|---------|
| `--path` | Required scored, labeled CSV. |
| `--threshold` | Required decision threshold in `[0, 1]`; a row is positive when `risk_score >= threshold`. It is recorded, not recalibrated. |
| `--label-source` | Required description of where the verified outcome came from. |
| `--labels-verified` | Assert that `is_fraud` is a confirmed outcome. Without it, the report is explicitly `partial` and provisional. |
| `--period` | `day`, `week` or `month` (default `Settings.performance_period`, itself defaulting to `month`). |
| `--window-start` / `--window-end` | Optional inclusive UTC bounds on `scored_at`. |
| `--report-dir` | Output directory (default `Settings.performance_report_dir`, `./reports/performance`). |
| `--no-persist` | Write JSON only; do not insert a `performance_reports` row. |

The command writes timestamped JSON plus `latest.json` and, unless `--no-persist` is passed,
persists one aggregate row. Set `DATABASE_URL` or `POSTGRES_*` to the database the API uses.

## Metrics and undefined states

The report records the threshold and these definitions:

| Metric | Definition | Undefined when |
|--------|------------|----------------|
| Precision | `TP / (TP + FP)` | No row was predicted positive at the threshold |
| Recall | `TP / (TP + FN)` | The group has no actual positive |
| F1 | `2PR / (P + R)` | Precision or recall is undefined |
| PR-AUC | Average precision from the positive-class scores | The group has only one class |
| ROC-AUC | Area under the ROC curve from the scores | The group has only one class |
| False-positive rate | `FP / (FP + TN)` | The group has no actual negative |
| Confusion matrix | `TN`, `FP`, `FN`, `TP` counts at the threshold | Never; counts are always present |

Every metric is serialized as:

```json
{
  "value": null,
  "defined": false,
  "reason": "no actual positives in this group"
}
```

A defined metric has a numeric `value`, `"defined": true` and a `null` reason. An undefined
metric has a `null` value and a non-empty reason. It is never coerced to zero and is never
silently omitted. A zero F1 is reported as a defined `0.0` when both precision and recall are
defined and zero; that is a computed value, not a missing one.

## Grouping and provenance

Metrics are computed for:

- the overall selected rows;
- each `model_version` across the selected periods;
- each `(model_version, period)` segment.

Periods are UTC calendar periods. A segment's `start` is inclusive and its `end` is exclusive;
the report states `end_inclusive: false`. Small segments are still reported with their row and
class counts so a reader can discount unstable estimates instead of guessing.

The report records:

- evaluation timestamp and report ID;
- observed data window plus any requested window bounds;
- dataset SHA-256 and row counts, including rows excluded by the window;
- threshold, period, label source and `labels_verified`;
- the exact metric definitions and caveats;
- the score source, explicitly "supplied at prediction time; not recomputed";
- a label-provenance block with `analyst_review_status_used: false`.

## Persistence, API and dashboard

The aggregate report is stored in `performance_reports` with summary columns for window,
threshold, period, label source, verification flag, row/version/segment counts and the complete
JSON payload. No feature row, transaction identifier or raw label is stored in that table.

`GET /api/v1/performance/summary` is admin-protected and returns either:

```json
{"available": false, "reason": "No performance report has been generated.", "report": null}
```

or `{"available": true, "reason": null, "report": {...}}`. The dashboard Performance tab renders
the overall metrics, the per-version table, the per-segment table, warnings and caveats. It shows
`undefined` rather than substituting a number.

## Late labels and interpretation limits

Fraud outcomes arrive late, often after a chargeback or investigation window. This creates
several limits that the report states rather than hides:

1. **Label completeness.** A recent window may contain few or no verified outcomes. Missing
   labels are not negatives; the report only includes rows that were supplied with a label.
2. **Label selection bias.** Analyst-reviewed rows are not a random sample of traffic. Using
   `review_status` as a target would over-represent alerted transactions and distort precision,
   recall and PR-AUC. That is why it is rejected.
3. **Score vintage.** A row's `risk_score` came from the model version that served it. A later
   model's performance cannot be inferred from those scores; compare versions by their own
   segments.
4. **Class imbalance and small segments.** PR-AUC is the primary headline metric. ROC-AUC is
   reported but can look optimistic under heavy imbalance. A segment with only one class has an
   undefined PR-AUC/ROC-AUC, and a segment with very few rows has wide uncertainty even when the
   metric is defined.
5. **Threshold dependence.** Precision, recall, F1 and the false-positive rate depend on the
   recorded threshold. PR-AUC and ROC-AUC do not. The threshold is not calibrated by this
   workflow.
6. **No causal claim.** A change in performance across periods may reflect label lag, case mix,
   policy changes or data quality, not model decay. Investigate drift and label provenance
   together.
7. **Not a promotion gate by itself.** The final held-out test evaluation remains the frozen
   reference under `artifacts/final_evaluation/`. Routine reports must not be used to tune on the
   same labels or to silently re-select a model. Promotion remains a deliberate review step.

## Related documents

- [`MONITORING.md`](MONITORING.md): drift signals, thresholds and triage.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) §5.3 and §8.1: tables and design rationale.
- [`ROADMAP.md`](ROADMAP.md) M8: implementation status and exit criteria.
