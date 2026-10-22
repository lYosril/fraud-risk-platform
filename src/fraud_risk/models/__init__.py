"""Model training, evaluation, artifact management and inference (architecture areas 3-4).

Status
    Splitting, the three-family training comparison, the metric layer, the validation-only
    operating policies, the frozen final evaluation, SHAP explanations for the selected
    XGBoost bundle, MLflow run tracking (``fraud_risk.tracking``) and the reusable inference
    service are implemented. A two-run MLflow history, the calibration comparison, the
    artifact registry and HTTP serving are still scaffolding. See ``docs/ROADMAP.md``
    (M4, M5).
"""
