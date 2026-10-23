"""Validation tests for the model artifact bundle contract.

Every bundle here is synthetic and tiny. No trained model, dataset or artifact store is read;
these tests pin the schema guards that keep a malformed bundle from reaching serving.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import joblib
import pytest

from fraud_risk.models.bundle import (
    BUNDLE_SCHEMA_VERSION,
    FrozenModel,
    ModelBundleError,
    finite_number,
    load_model_bundle,
    sha256_file,
)
from fraud_risk.models.evaluation import CostPolicy


class PredictingPipeline:
    """The smallest object that satisfies the bundle's callable check."""

    def predict_proba(self, frame: object) -> list[list[float]]:
        del frame
        return [[0.5, 0.5]]


def make_bundle(**overrides: Any) -> dict[str, Any]:
    bundle: dict[str, Any] = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "seed": 42,
        "feature_columns": ["V1", "Amount"],
        "threshold": 0.5,
        "cost_policy": {"false_negative_cost": 100.0, "false_positive_cost": 1.0},
        "model": {"family": "test-family", "params": {"n_estimators": 10}},
        "pipeline": PredictingPipeline(),
    }
    bundle.update(overrides)
    return bundle


def write_bundle(path: Path, bundle: object) -> Path:
    joblib.dump(bundle, path)
    return path


def test_missing_and_corrupt_bundles_are_typed_errors(tmp_path: Path) -> None:
    with pytest.raises(ModelBundleError, match="not found"):
        load_model_bundle(tmp_path / "missing.joblib")

    corrupt = tmp_path / "corrupt.joblib"
    corrupt.write_text("not a joblib artifact", encoding="utf-8")
    with pytest.raises(ModelBundleError, match="could not load"):
        load_model_bundle(corrupt)


def test_bundle_must_be_a_mapping_with_every_required_key(tmp_path: Path) -> None:
    with pytest.raises(ModelBundleError, match="mapping"):
        load_model_bundle(write_bundle(tmp_path / "list.joblib", [1, 2, 3]))

    bundle = make_bundle()
    del bundle["threshold"]
    with pytest.raises(ModelBundleError, match="missing keys"):
        load_model_bundle(write_bundle(tmp_path / "missing-key.joblib", bundle))


def test_bundle_schema_version_and_pipeline_are_validated(tmp_path: Path) -> None:
    with pytest.raises(ModelBundleError, match="schema_version"):
        load_model_bundle(write_bundle(tmp_path / "version.joblib", make_bundle(schema_version=1)))

    with pytest.raises(ModelBundleError, match="predict_proba"):
        load_model_bundle(
            write_bundle(tmp_path / "pipeline.joblib", make_bundle(pipeline=object()))
        )


@pytest.mark.parametrize(
    ("columns", "message"),
    [
        ([], "non-empty list"),
        ("V1", "non-empty list"),
        (["V1", 2], "non-empty strings"),
        (["V1", ""], "non-empty strings"),
        (["V1", "V1"], "unique"),
    ],
)
def test_feature_columns_are_validated(
    tmp_path: Path,
    columns: object,
    message: str,
) -> None:
    path = write_bundle(tmp_path / "columns.joblib", make_bundle(feature_columns=columns))

    with pytest.raises(ModelBundleError, match=message):
        load_model_bundle(path)


@pytest.mark.parametrize("seed", [True, -1, 2**32, "42"])
def test_seed_must_be_an_unsigned_32_bit_integer(tmp_path: Path, seed: object) -> None:
    path = write_bundle(tmp_path / "seed.joblib", make_bundle(seed=seed))

    with pytest.raises(ModelBundleError, match="seed"):
        load_model_bundle(path)


@pytest.mark.parametrize("threshold", [True, "0.5", math.nan, math.inf, -0.1, 1.1])
def test_threshold_must_be_a_finite_probability(tmp_path: Path, threshold: object) -> None:
    path = write_bundle(tmp_path / "threshold.joblib", make_bundle(threshold=threshold))

    with pytest.raises(ModelBundleError, match="threshold"):
        load_model_bundle(path)


def test_cost_policy_must_be_a_mapping_of_positive_finite_numbers(tmp_path: Path) -> None:
    with pytest.raises(ModelBundleError, match="cost_policy"):
        load_model_bundle(
            write_bundle(tmp_path / "policy-shape.joblib", make_bundle(cost_policy=[]))
        )
    with pytest.raises(ModelBundleError, match="cost_policy"):
        load_model_bundle(
            write_bundle(
                tmp_path / "policy-missing.joblib",
                make_bundle(cost_policy={"false_negative_cost": 100.0}),
            )
        )
    with pytest.raises(ModelBundleError, match="cost_policy"):
        load_model_bundle(
            write_bundle(
                tmp_path / "policy-bool.joblib",
                make_bundle(cost_policy={"false_negative_cost": True, "false_positive_cost": 1.0}),
            )
        )
    with pytest.raises(ModelBundleError, match="cost_policy"):
        load_model_bundle(
            write_bundle(
                tmp_path / "policy-zero.joblib",
                make_bundle(cost_policy={"false_negative_cost": 0.0, "false_positive_cost": 1.0}),
            )
        )


def test_model_metadata_must_be_a_mapping(tmp_path: Path) -> None:
    path = write_bundle(tmp_path / "metadata.joblib", make_bundle(model=[]))

    with pytest.raises(ModelBundleError, match="metadata"):
        load_model_bundle(path)


@pytest.mark.parametrize("fingerprint", [1, "", "   "])
def test_run_fingerprint_must_be_a_non_empty_string_when_present(
    tmp_path: Path, fingerprint: object
) -> None:
    path = write_bundle(
        tmp_path / "fingerprint.joblib",
        make_bundle(run_fingerprint=fingerprint),
    )

    with pytest.raises(ModelBundleError, match="run_fingerprint"):
        load_model_bundle(path)


def test_valid_bundle_loads_the_fitted_object_and_falls_back_to_the_content_hash(
    tmp_path: Path,
) -> None:
    path = write_bundle(tmp_path / "valid.joblib", make_bundle())

    model = load_model_bundle(path)

    assert isinstance(model, FrozenModel)
    assert model.path == path
    assert isinstance(model.pipeline, PredictingPipeline)
    assert model.feature_columns == ("V1", "Amount")
    assert model.threshold == 0.5
    assert model.seed == 42
    assert model.family == "test-family"
    assert model.model_version == sha256_file(path)
    assert model.sha256 == sha256_file(path)
    assert model.cost_policy == CostPolicy(100.0, 1.0)
    assert model.metadata == {"family": "test-family", "params": {"n_estimators": 10}}


def test_valid_bundle_uses_a_run_fingerprint_when_present(tmp_path: Path) -> None:
    path = write_bundle(
        tmp_path / "fingerprinted.joblib",
        make_bundle(run_fingerprint="  run-fingerprint  "),
    )

    model = load_model_bundle(path)

    assert model.model_version == "run-fingerprint"
    assert model.family == "test-family"


def test_family_is_none_when_metadata_does_not_hold_a_string() -> None:
    model = FrozenModel(
        path=Path("model.joblib"),
        sha256="a" * 64,
        model_version="version",
        pipeline=PredictingPipeline(),
        feature_columns=("V1",),
        threshold=0.5,
        cost_policy=CostPolicy(100.0, 1.0),
        seed=1,
        metadata={"family": 7},
    )

    assert model.family is None


def test_finite_number_rejects_bool_non_number_and_non_finite_values() -> None:
    assert finite_number(1, "x") == 1.0
    for value in (True, "1", math.nan, math.inf):
        with pytest.raises(ModelBundleError, match="x must be"):
            finite_number(value, "x")


def test_sha256_file_hashes_bytes(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"abc")

    assert sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
