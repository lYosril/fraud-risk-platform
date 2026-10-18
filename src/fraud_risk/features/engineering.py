"""Feature engineering: the shared transform.

Responsibility
    Build the model matrix from validated transactions through one declarative
    ``sklearn`` ``ColumnTransformer`` / ``Pipeline`` that is fitted on training data only,
    then serialised into the model artifact bundle.

Why this module matters
    Training and serving call the *same fitted object* (architecture decision D3). Any
    re-implementation of preprocessing on the inference path is train/serve skew by
    definition and is treated as a defect.

Status
    Scaffolding only. See ``docs/ROADMAP.md`` (M3).
"""
