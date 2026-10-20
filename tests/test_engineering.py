"""Tests for the shared preprocessing transform.

Every frame is deterministic and built in memory; nothing here reads the real dataset. The
assertions concentrate on leakage: only declared feature columns are selected, the label can
never enter ``X``, and standardisation statistics come from the training rows alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from fraud_risk.data.dataset import (
    AMOUNT_COLUMN,
    EXPECTED_COLUMNS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    TIME_COLUMN,
)
from fraud_risk.features.engineering import (
    EXCLUDED_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    FeatureSchemaError,
    ScalingStrategy,
    build_preprocessor,
    split_features,
    validate_feature_frame,
)


def make_frame(rows: int = 40, seed: int = 11) -> pd.DataFrame:
    """A deterministic frame that satisfies the documented dataset schema."""
    rng = np.random.default_rng(seed)
    data: dict[str, object] = {TIME_COLUMN: np.arange(rows, dtype="float64") * 60.0}
    for column in FEATURE_COLUMNS:
        data[column] = rng.normal(size=rows)
    data[AMOUNT_COLUMN] = np.round(rng.lognormal(mean=4.0, sigma=1.5, size=rows), 2)
    data[TARGET_COLUMN] = np.array([1] * 4 + [0] * (rows - 4), dtype="int64")
    return pd.DataFrame(data)[list(EXPECTED_COLUMNS)]


# ---------------------------------------------------------------------------
# Feature contract
# ---------------------------------------------------------------------------
def test_model_features_exclude_the_label_and_time() -> None:
    """The schema records the decision, not just a comment in a script."""
    assert AMOUNT_COLUMN in MODEL_FEATURE_COLUMNS
    assert TARGET_COLUMN not in MODEL_FEATURE_COLUMNS
    assert TIME_COLUMN not in MODEL_FEATURE_COLUMNS
    assert set(EXCLUDED_COLUMNS) == {TARGET_COLUMN, TIME_COLUMN}
    assert len(MODEL_FEATURE_COLUMNS) == 29


def test_split_features_returns_the_declared_schema() -> None:
    """``X`` holds exactly the model columns and ``y`` is the untouched label."""
    frame = make_frame()

    features, labels = split_features(frame)

    assert list(features.columns) == list(MODEL_FEATURE_COLUMNS)
    assert TARGET_COLUMN not in features.columns
    assert labels.equals(frame[TARGET_COLUMN])
    assert features.dtypes.eq("float64").all()


def test_split_features_does_not_mutate_or_alias_the_source() -> None:
    """Editing the returned frame must not reach the caller's frame."""
    frame = make_frame()
    before = frame.copy(deep=True)

    features, labels = split_features(frame)
    features.iloc[0, 0] = 12345.0
    labels.iloc[0] = -1

    pd.testing.assert_frame_equal(frame, before)


def test_validate_accepts_extra_columns() -> None:
    """Time, the label and any request metadata are allowed; they are not features."""
    validate_feature_frame(make_frame())


def test_missing_feature_column_is_rejected() -> None:
    frame = make_frame().drop(columns=["V1"])

    with pytest.raises(FeatureSchemaError, match="V1"):
        split_features(frame)


def test_missing_label_is_rejected() -> None:
    with pytest.raises(FeatureSchemaError, match=TARGET_COLUMN):
        split_features(make_frame().drop(columns=[TARGET_COLUMN]))


def test_label_values_other_than_zero_and_one_are_rejected() -> None:
    frame = make_frame()
    frame.loc[0, TARGET_COLUMN] = 2

    with pytest.raises(FeatureSchemaError, match="0 and 1"):
        split_features(frame)


def test_non_numeric_feature_is_rejected() -> None:
    frame = make_frame().assign(V1="not-a-number")

    with pytest.raises(FeatureSchemaError, match="numeric"):
        split_features(frame)


def test_missing_feature_value_is_rejected() -> None:
    """A NaN would be scaled into a NaN score, so it fails at the boundary instead."""
    frame = make_frame()
    frame.loc[3, "V4"] = np.nan

    with pytest.raises(FeatureSchemaError, match="missing values"):
        split_features(frame)


def test_label_cannot_be_declared_as_a_feature() -> None:
    with pytest.raises(FeatureSchemaError, match="label"):
        split_features(
            make_frame(),
            feature_columns=(*MODEL_FEATURE_COLUMNS, TARGET_COLUMN),
        )


# ---------------------------------------------------------------------------
# Preprocessing: tree models
# ---------------------------------------------------------------------------
def test_tree_preprocessor_passes_values_through_unchanged() -> None:
    """Tree models are scale-invariant; scaling would only obscure SHAP values."""
    features, _ = split_features(make_frame())

    preprocessor = build_preprocessor(ScalingStrategy.NONE)
    transformed = preprocessor.fit_transform(features)

    np.testing.assert_allclose(transformed, features.to_numpy())
    assert list(preprocessor.get_feature_names_out()) == list(MODEL_FEATURE_COLUMNS)


def test_tree_preprocessor_has_no_scaler() -> None:
    preprocessor = build_preprocessor("none")

    assert [name for name, _, _ in preprocessor.transformers] == ["raw"]
    assert preprocessor.transformers[0][1] == "passthrough"


# ---------------------------------------------------------------------------
# Preprocessing: linear models
# ---------------------------------------------------------------------------
def test_standard_preprocessor_fits_on_training_rows_only() -> None:
    """Validation is transformed with training statistics, never with its own."""
    features, _ = split_features(make_frame(rows=60))
    train = features.iloc[:40]
    held_out = features.iloc[40:]

    preprocessor = build_preprocessor(ScalingStrategy.STANDARD)
    preprocessor.fit(train)
    transformed = preprocessor.transform(held_out)

    scaler = preprocessor.named_transformers_["scale"]
    np.testing.assert_allclose(scaler.mean_, train.mean().to_numpy())
    expected = (held_out.to_numpy() - train.mean().to_numpy()) / train.std(ddof=0).to_numpy()
    np.testing.assert_allclose(transformed, expected)

    # The whole-frame statistic would give a different answer, which is exactly the leak
    # this test rules out.
    leaked = (held_out.to_numpy() - features.mean().to_numpy()) / features.std(ddof=0).to_numpy()
    assert not np.allclose(transformed, leaked)


def test_standard_preprocessor_output_is_centred_on_training_data() -> None:
    features, _ = split_features(make_frame())
    preprocessor = build_preprocessor(ScalingStrategy.STANDARD).fit(features)

    transformed = preprocessor.transform(features)

    np.testing.assert_allclose(
        transformed.mean(axis=0), np.zeros(len(MODEL_FEATURE_COLUMNS)), atol=1e-9
    )
    np.testing.assert_allclose(transformed.std(axis=0), np.ones(len(MODEL_FEATURE_COLUMNS)))


def test_standard_preprocessor_emits_the_declared_feature_names() -> None:
    features, _ = split_features(make_frame())
    preprocessor = build_preprocessor(ScalingStrategy.STANDARD).fit(features)

    assert list(preprocessor.get_feature_names_out()) == list(MODEL_FEATURE_COLUMNS)


# ---------------------------------------------------------------------------
# Shared guarantees
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("strategy", list(ScalingStrategy))
def test_preprocessor_is_one_column_transformer(strategy: ScalingStrategy) -> None:
    """D3: training and serving share one declarative object."""
    assert isinstance(build_preprocessor(strategy), ColumnTransformer)


@pytest.mark.parametrize("strategy", list(ScalingStrategy))
def test_preprocessor_drops_columns_it_was_not_given(strategy: ScalingStrategy) -> None:
    """Even a frame containing Class and Time produces only model features."""
    frame = make_frame()

    transformed = build_preprocessor(strategy).fit_transform(frame)

    assert transformed.shape == (len(frame), len(MODEL_FEATURE_COLUMNS))


def test_unknown_scaling_strategy_is_rejected() -> None:
    with pytest.raises(ValueError, match="mystery"):
        build_preprocessor("mystery")
