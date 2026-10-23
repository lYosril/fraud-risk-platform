# Monitoring runbook

This document describes the implemented M8 monitoring path. It covers what is measured, how
the reference is built, how thresholds were chosen, how to read a report, and what the system
deliberately does **not** claim.

The implementation is first-party (`fraud_risk.monitoring`), not Evidently. Evidently remains
an optional `monitoring` extra for richer HTML rendering, but it is not required for the alert
decision and is not installed by the default dependency set.

---

## 1. What is measured

| Area | Signal | Reference | Availability |
|------|--------|-----------|--------------|
| Feature distribution | Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) per feature | Training split | Only when `PERSIST_FEATURE_SNAPSHOT=true` and snapshots exist |
| Score distribution | PSI, KS, mean shift and p50/p90/p99 of the risk score | Training-split score distribution | Whenever predictions exist |
| Missing values | Per-feature missing rate and overall missing-cell rate | Training reference missing rate (normally 0) | Only with feature snapshots |
| Input validation | Count and rate of rejected requests | No reference distribution; threshold applies to the rate | When validation-failure capture is enabled |
| Traffic | Prediction volume, decision mix, model-version usage, analyst review coverage | None (operational counts) | Whenever predictions exist |
| Performance | Delayed-label PR-AUC / recall | Requires complete delayed labels | **Unavailable** in the current platform |

A feature or score shift is evidence that the input or output distribution changed. It is **not**
evidence that model performance degraded. The report keeps these sections separate and always
sets `performance.available=false` unless complete delayed labels exist.

Analyst review outcomes (`confirmed_fraud`, `confirmed_legitimate`) are reported only as an
incomplete, non-random reviewed subset. They are not a substitute for delayed ground truth, and
the platform does not compute an overall PR-AUC, recall or accuracy from them.

---

## 2. Reference profile

The reference is built **from the training split only**. It contains aggregate statistics, not
transaction rows. The builder lives in the offline model layer (`fraud_risk.models.reference`)
and feeds the leaf `fraud_risk.monitoring.reference` serializer; the monitoring report path does
not import model code.

The profile contains:

- Quantile bin edges (10 bins by default) and the reference share in each bin.
- A 101-point quantile sample for the approximate KS statistic.
- Count, mean, missing rate, and the model identity/threshold that produced the score reference.
- Dataset source, split strategy, split seed and number of scored rows.

Build it with the real dataset and the frozen artifact:

```powershell
python -m fraud_risk.models.reference `
  --path data/raw/creditcard.csv `
  --model-artifact artifacts/final_evaluation/frozen_model.joblib `
  --output reports/monitoring/reference_profile.json
```

Defaults come from `Settings` (`MONITORING_REFERENCE_PATH`, `MODEL_ARTIFACT_PATH`). The default
split is temporal with `RANDOM_SEED` (42), matching the repository training default. Use
`--strategy` / `--seed` only when reproducing a model trained with a different split; the values
are recorded in the profile.

`reference_id` is a SHA-256 digest of the distribution content. Editing the JSON invalidates the
ID and loading fails loudly. Both `reports/` and the reference file are git-ignored.

`--max-rows N` scores a deterministic evenly spaced subset of the training split. `0` (the
default) uses every training row. The sampling method is recorded in the profile.

---

## 3. Running a report

```powershell
python -m fraud_risk.monitoring run --window-hours 24
```

The command:

1. Loads and validates the reference profile.
2. Reads exact aggregates for the inclusive window from PostgreSQL: prediction volume, decision
   mix, model-version usage and analyst review-state counts.
3. Loads at most `MONITORING_SAMPLE_LIMIT` newest prediction rows for score/feature
   distributions. Aggregate counts are never derived from this sample.
4. Sums count-only validation-failure events in the same window.
5. Writes `reports/monitoring/drift_<UTC timestamp>_<id>.json` and `latest.json`.
6. Inserts one row in `drift_reports` unless `--no-persist` was passed.

The report is available through the admin route `GET /api/v1/monitoring/summary` and the
dashboard **Monitoring** tab. If no report has been generated, the API returns
`available=false` with a reason; it does not synthesise a report.

### Scheduling

Run the command from the same scheduler that owns other batch jobs (cron, Task Scheduler,
Kubernetes CronJob). The repository does not ship an in-process scheduler. A report should be
generated at least once per configured window; more frequent runs simply cover overlapping
windows. Nothing in the design retrains a model automatically.

---

## 4. Thresholds and rationale

Thresholds live in `Settings`, are validated at startup, and are copied into every JSON report.
`warning` must be strictly below `alert` for every pair.

| Signal | Warning | Alert | Rationale |
|--------|---------|-------|-----------|
| Feature PSI | 0.10 | 0.25 | Common PSI convention: below 0.10 is usually treated as no meaningful shift, 0.10–0.25 as moderate, above 0.25 as major. |
| Score PSI | 0.10 | 0.25 | Same convention, applied to the model output rather than an input. |
| KS statistic | 0.10 | 0.20 | A diagnostic band, not a calibrated decision rule. KS grows with sample size, so the p-value is reported but not used for the alert. |
| Missing rate | 1% | 5% | The API rejects non-finite values, so any stored missingness is unexpected. These bands are deliberately conservative. |
| Validation failure rate | 2% | 5% | Share of rejected requests among rejected + successful scoring attempts. A starting operational band to tune against real client behaviour. |

These are **starting thresholds**, not proved optima. They are configuration so an operator can
change them without editing analysis code, and the values used are recorded in the report. Review
them after accumulating real windows; do not change them to make an alert disappear.

PSI definition:

- Bins are learned from the reference distribution and reused unchanged for the current window.
- Reference and current bin vectors are normalised.
- Empty bins are floored at `1e-6` before the logarithm, then both vectors are renormalised. This
  is documented because an unfloored PSI is infinite when a reference bin has no current row.
- Missing values are excluded from PSI and tracked in the missing-value section.

KS uses the stored 101-point reference quantile sample, so the statistic is approximate. The
p-value uses the full reference and current counts as effective sizes and is asymptotic. Neither
is a substitute for looking at the distributions.

---

## 5. Reading a report

Top-level fields:

```json
{
  "schema_version": 1,
  "window": {"start": "...", "end": "...", "hours": 24},
  "summary": {
    "status": "ok|warning|alert|unavailable",
    "data_drift_status": "...",
    "data_quality_status": "...",
    "performance_status": "unavailable",
    "prediction_count": 0,
    "validation_failure_count": 0,
    "feature_snapshot_count": 0,
    "feature_drift_count": 0
  },
  "feature_drift": {},
  "missing_values": {},
  "score_drift": {},
  "validation_failures": {},
  "traffic": {},
  "performance": {},
  "thresholds": {},
  "caveats": []
}
```

- `status` is the worst available signal across data drift and data quality. It never includes
  performance while performance is unavailable.
- `data_drift_status` combines feature and score drift.
- `data_quality_status` combines missing values and validation failures.
- `performance_status` is `unavailable` in this drift report. The reason distinguishes "no
  predictions", "no reviewed outcomes" and "reviewed subset is not complete ground truth".
  Routine performance is reported separately from an operator-supplied verified extract; see
  [`PERFORMANCE.md`](PERFORMANCE.md).
- `feature_drift.breached_feature_count` counts features at warning or alert.
- `traffic.sample_truncated` is true when the bounded score/feature sample did not cover every
  prediction. `traffic.predictions` and the model-version/decision counts remain exact.
- Individual feature rows include `psi`, `ks_statistic`, `ks_p_value`, `missing_rate`,
  `reference_mean`, `current_mean` and `sample_count`.

An empty window is a valid report: `status=unavailable`, all distributions unavailable, and an
explicit "window is empty" reason. It is not an error and it is not a zero-drift result.

---

## 6. Runbook: what to check first

### Validation-failure alert

1. Check the API access log for the status code and endpoint; the monitoring event stores only
   the endpoint path and a reason code, never the body.
2. Ask whether a client shipped a schema change or started sending missing/non-finite features.
3. Compare the failure count with prediction volume. A high rate with low volume may be a single
   broken client; both high suggests a broader integration problem.
4. Do not relax the rate threshold until the client issue is understood.

### Feature PSI alert

1. Open the feature row and compare `current_mean` with `reference_mean`, and inspect the
   per-feature missing rate.
2. Check whether the window contains a legitimate population change (new merchant, campaign,
   time-of-day mix). PSI is sensitive to any composition change.
3. Remember that feature drift is not performance degradation. If labels are unavailable, do not
   claim an effect on fraud detection performance.
4. Feature drift is unavailable when `PERSIST_FEATURE_SNAPSHOT=false`; enabling it changes the
   privacy/storage surface and must be a deliberate decision.

### Score PSI / KS alert

1. Confirm the window has enough predictions; a tiny window makes the statistic unstable.
2. Check model-version usage: a newly promoted or misconfigured model explains a score shift.
3. Check decision mix and threshold changes. A threshold change moves decisions without changing
   the underlying scores.
4. Only after complete delayed labels arrive should anyone evaluate PR-AUC or recall.

### Missing-value alert

1. The API validates inputs, so missing values in stored snapshots usually indicate a malformed
   snapshot writer or a manual data import, not a valid request.
2. Check the affected feature names and the snapshot row count.
3. If snapshots are disabled, the section is unavailable rather than zero.

### Performance

There is no real-time performance metric and this platform does not compute one from analyst
reviews. Wait for complete delayed labels, join them offline, and evaluate with the normal
evaluation path. An analyst-reviewed subset is selected non-randomly and cannot support an
overall precision, recall, PR-AUC or accuracy claim.

---

## 7. Privacy and storage

- Reference profiles and reports are aggregates. They contain no transaction ID, card/account
  identifier, label or full feature row.
- `monitoring_events` stores `event_type`, endpoint path, reason code, timestamp and a count. It
  never stores the request body, headers, token or feature snapshot.
- `drift_reports` stores summary columns plus the same JSON report. Raw prediction rows remain in
  `predictions`; feature snapshots remain opt-in and off by default.
- The report sample is bounded. Loading the bounded sample is the only place feature snapshots
  are read; no feature value is written back to the report.
- `reports/`, `data/raw`, `data/interim`, `data/processed`, `artifacts/` and `mlruns/` are
  git-ignored. Do not commit generated reports or reference profiles.

---

## 8. Tests

`tests/test_monitoring.py` covers:

- PSI against identical and hand-constructed distributions where the value is known.
- Reference-bin reuse and detection of a shifted feature/score distribution.
- KS against a constructed half-distribution shift.
- Missing-value accounting, including `None` and non-finite values.
- Reference profile round-trip and tamper detection.
- Empty windows, absent feature snapshots, truncated samples, validation-failure rates and the
  explicit unavailable-performance state.
- JSON report writing.

`tests/test_db_integration.py` (requires PostgreSQL) covers migration/schema shape, count-only
event persistence, and persistence/round-trip of both an ok report and a shifted-window alert
report. `tests/test_api.py` covers the admin route and validation-failure counting.
`tests/test_models_reference.py` covers the reference CLI orchestration with a synthetic model.
`tests/test_dashboard_client.py` and `tests/test_dashboard_transforms.py` cover the dashboard
summary contract and tables.

No test uses a real trained model or production transaction rows; the report builder uses small
deterministic synthetic fixtures.
