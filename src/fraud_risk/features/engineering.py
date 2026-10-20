"""Feature engineering: the single shared preprocessing transform.

Responsibility
    Turn a validated transaction frame into the model matrix through one declarative
    ``sklearn`` ``ColumnTransformer``. It is fitted on training rows only and is the *same*
    object that is serialised into the artifact bundle and used at serving time
    (architecture decision D3). Re-implementing any part of it on the inference path is
    train/serve skew by definition.

Feature contract
    The model sees ``V1..V28`` (anonymous PCA components) and ``Amount``. Two dataset
    columns are deliberately excluded, and the reason is part of the schema rather than a
    comment in a script:

    ``Class``
        The label. It must never reach a feature frame: :func:`split_features` refuses to
        build one that contains it, and the transformer drops any column it was not fitted
        on, so an accidental extra column cannot leak either.
    ``Time``
        Seconds elapsed since the *first row of this file*, not wall-clock time. The API
        scores one transaction at a time and has no dataset-relative clock, so this value
        cannot be computed at serving time. It also conflicts with the temporal split
        (``models.splitting``): training rows occupy the earlier part of the range and test
        rows the later one, so a model that learned "position in the file" would be asked to
        extrapolate beyond everything it saw - impossible for trees and arbitrary for linear
        models. ``Time`` is still used to order the temporal split and to window drift
        reports; it is not a feature.

Scaling
    No common scale is assumed. ``Amount`` is on a completely different scale from the PCA
    components (and has a long right tail), so it is never treated as if it were one more
    ``V`` column. The choice belongs to the estimator family:

    * :attr:`ScalingStrategy.NONE` - tree models are invariant to monotone per-feature
      scaling, so raw values are passed through. This is also the reason no transform is
      applied "just in case": it would change nothing for the model and would make SHAP
      values harder to read.
    * :attr:`ScalingStrategy.STANDARD` - linear and distance-based models get a per-column
      ``StandardScaler``. Its statistics are learned from the training rows passed to
      ``fit`` and reused unchanged for validation, test and production.

    No distribution reshaping (log, power, quantile) is applied here. If one is ever
    justified it belongs in this transformer, chosen on training data, so that the same
    object teaches serving the same mapping.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from types import MappingProxyType

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from fraud_risk.data.dataset import (
    AMOUNT_COLUMN,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    TIME_COLUMN,
)

#: Every column the model matrix is built from, in the order the model sees. ``Time`` is
#: absent on purpose; see the module docstring and :data:`EXCLUDED_COLUMNS`.
MODEL_FEATURE_COLUMNS: tuple[str, ...] = (*FEATURE_COLUMNS, AMOUNT_COLUMN)

#: Columns that exist in the dataset but must never be model features, with the reason.
EXCLUDED_COLUMNS: Mapping[str, str] = MappingProxyType(
    {
        TARGET_COLUMN: "label; leaking it into the features would make the score meaningless",
        TIME_COLUMN: (
            "dataset-relative elapsed seconds: unavailable when scoring one transaction and "
            "cannot be extrapolated under a temporal split; used only for split ordering"
        ),
    }
)


class FeatureSchemaError(ValueError):
    """A frame does not satisfy the documented model feature contract."""


class ScalingStrategy(StrEnum):
    """How numeric features are prepared for the estimator."""

    #: Raw values. Correct for tree models, which are scale-invariant.
    NONE = "none"
    #: Per-column standardisation. Fitted on training rows only.
    STANDARD = "standard"


def validate_feature_frame(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str] = MODEL_FEATURE_COLUMNS,
) -> None:
    """Raise :class:`FeatureSchemaError` unless ``frame`` can produce the model matrix.

    Only the declared feature columns are inspected: extra columns (``Time``, ``Class``,
    request identifiers) are allowed and are ignored by the transformer.
    """
    if TARGET_COLUMN in feature_columns:
        msg = f"{TARGET_COLUMN!r} is the label and must never be declared as a model feature"
        raise FeatureSchemaError(msg)

    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        msg = f"missing model feature columns: {', '.join(missing)}"
        raise FeatureSchemaError(msg)

    non_numeric = [
        column for column in feature_columns if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if non_numeric:
        msg = f"model feature columns must be numeric: {', '.join(non_numeric)}"
        raise FeatureSchemaError(msg)

    if frame.loc[:, list(feature_columns)].isna().to_numpy().any():
        msg = "model feature columns contain missing values"
        raise FeatureSchemaError(msg)


def split_features(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str] = MODEL_FEATURE_COLUMNS,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return ``(X, y)`` for ``frame`` without mutating it.

    The result is a copy: later edits to the returned frames cannot reach the caller's
    frame, and the label is not present in ``X`` by construction.
    """
    validate_feature_frame(frame, feature_columns=feature_columns)

    if TARGET_COLUMN not in frame.columns:
        msg = f"missing label column {TARGET_COLUMN!r}"
        raise FeatureSchemaError(msg)

    labels = frame.loc[:, TARGET_COLUMN]
    if not set(pd.unique(labels)) <= {0, 1}:
        msg = f"{TARGET_COLUMN} must contain only the values 0 and 1"
        raise FeatureSchemaError(msg)

    features = frame.loc[:, list(feature_columns)].copy()
    return features, labels.copy()


def build_preprocessor(
    strategy: ScalingStrategy | str = ScalingStrategy.NONE,
    *,
    feature_columns: Sequence[str] = MODEL_FEATURE_COLUMNS,
) -> ColumnTransformer:
    """Build the shared transform. Fitting is the caller's job, on training rows only.

    Both branches are a ``ColumnTransformer`` so that training and serving depend on one
    object with a fixed column order. The passthrough branch is not decorative: it selects
    exactly ``feature_columns`` and ``remainder="drop"`` discards everything else, so a
    stray ``Class`` or ``Time`` column in a request can never become an input feature.
    """
    resolved = ScalingStrategy(strategy)
    columns = list(feature_columns)

    transformers: list[tuple[str, object, list[str]]]
    if resolved is ScalingStrategy.STANDARD:
        transformers = [("scale", StandardScaler(), columns)]
    else:
        transformers = [("raw", "passthrough", columns)]

    return ColumnTransformer(
        transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
