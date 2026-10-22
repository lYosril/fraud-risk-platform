"""Inference and per-prediction explanation.

Responsibility
    Expose the only way anything outside this package scores a transaction: lazy-load plus
    cache an artifact bundle, build features through the fitted transformer, return a
    calibrated score, apply the three-way decision policy, and produce the SHAP top-k
    contributions for that single prediction.

Rules
    - Request handlers never build features by hand and never touch the estimator directly.
    - Loaded model + fitted transformer + SHAP explainer are cached together and reloaded
      atomically when the resolved alias changes.
    - Explanations are computed per request, not by sweeping the dataset.

Status
    Scaffolding only. The XGBoost SHAP core is implemented in ``models.explainability``; this
    module will wire it into the cached serving path. See ``docs/ROADMAP.md`` (M5).
"""
