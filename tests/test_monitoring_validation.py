"""Validation and edge-case tests for the monitoring statistics and reference profile.

The fixtures here are deliberately tiny and deterministic. They exercise the guards that keep
an invalid reference or a malformed window from silently producing a report, without needing
PostgreSQL, a trained model or the production dataset.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from fraud_risk.monitoring.drift import (
    DriftStatus,
    bin_indices,
    bin_proportions,
    finite_values,
    ks_p_value,
    ks_statistic,
    metric_status,
    psi_from_proportions,
    quantile_edges,
    status_value,
)
from fraud_risk.monitoring.reference import (
    ReferenceProfile,
    ReferenceProfileError,
    build_distribution_reference,
    build_reference_profile,
    load_reference_profile,
)

FEATURE_COLUMNS = ("V1", "V2")
NOW = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)


def make_frame() -> pd.DataFrame:
    """A deterministic three-row frame with no missing values."""
    return pd.DataFrame(
        {
            "V1": [-1.0, 0.0, 1.0, 2.0],
            "V2": [10.0, 20.0, 30.0, 40.0],
        }
    )


def make_reference() -> ReferenceProfile:
    """Build a valid reference from the deterministic frame."""
    return build_reference_profile(
        make_frame(),
        [0.1, 0.2, 0.8, 0.9],
        feature_columns=FEATURE_COLUMNS,
        source="synthetic training split",
        model_version="model-1",
        model_sha256="a" * 64,
        threshold=0.5,
        split_strategy="temporal",
        split_seed=7,
        training_rows=10,
        sample_method="all synthetic rows",
        generated_at=NOW,
        bin_count=2,
        quantile_sample_size=5,
    )


def make_payload() -> dict[str, Any]:
    """Return a JSON-ready, valid reference payload."""
    return make_reference().to_dict()


# ---------------------------------------------------------------------------
# Drift statistics: guards
# ---------------------------------------------------------------------------
def test_finite_values_drops_nan_and_infinity() -> None:
    result = finite_values([1.0, float("nan"), float("inf"), -2.0, float("-inf")])

    assert result.tolist() == [1.0, -2.0]


def test_quantile_edges_rejects_invalid_bin_count_and_empty_samples() -> None:
    with pytest.raises(ValueError, match="bin_count"):
        quantile_edges([1.0, 2.0], bin_count=0)
    with pytest.raises(ValueError, match="empty or non-finite"):
        quantile_edges([float("nan"), float("inf")])


def test_bin_indices_rejects_non_finite_or_unsorted_edges_and_non_finite_values() -> None:
    with pytest.raises(ValueError, match="edges must all be finite"):
        bin_indices([0.5], [0.0, float("inf")])
    with pytest.raises(ValueError, match="strictly increasing"):
        bin_indices([0.5], [0.5, 0.5])
    with pytest.raises(ValueError, match="values must all be finite"):
        bin_indices([float("nan")], [0.0])


def test_bin_proportions_rejects_an_empty_sample() -> None:
    with pytest.raises(ValueError, match="empty or non-finite"):
        bin_proportions([float("nan")], [0.5])


def test_psi_rejects_empty_mismatched_non_finite_negative_and_zero_mass_inputs() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        psi_from_proportions([], [0.5])
    with pytest.raises(ValueError, match="same length"):
        psi_from_proportions([0.5, 0.5], [1.0])
    with pytest.raises(ValueError, match="finite"):
        psi_from_proportions([0.5, float("nan")], [0.5, 0.5])
    with pytest.raises(ValueError, match="non-negative"):
        psi_from_proportions([0.5, -0.5], [0.5, 0.5])
    with pytest.raises(ValueError, match="positive mass"):
        psi_from_proportions([0.0, 0.0], [0.5, 0.5])
    with pytest.raises(ValueError, match="epsilon"):
        psi_from_proportions([0.5, 0.5], [0.5, 0.5], epsilon=1.0)


def test_ks_guards_and_endpoints() -> None:
    with pytest.raises(ValueError, match="non-empty finite"):
        ks_statistic([], [1.0])
    with pytest.raises(ValueError, match="positive"):
        ks_p_value(0.1, 0, 10)
    with pytest.raises(ValueError, match="finite and in"):
        ks_p_value(float("nan"), 10, 10)
    with pytest.raises(ValueError, match="finite and in"):
        ks_p_value(1.1, 10, 10)

    assert ks_p_value(0.0, 10, 10) == 1.0
    assert ks_p_value(1.0, 10, 10) == 0.0


def test_metric_status_classifies_bands_and_rejects_inverted_thresholds() -> None:
    assert metric_status(0.04, warning=0.05, alert=0.10) is DriftStatus.OK
    assert metric_status(0.05, warning=0.05, alert=0.10) is DriftStatus.WARNING
    assert metric_status(0.10, warning=0.05, alert=0.10) is DriftStatus.ALERT
    assert metric_status(None, warning=0.05, alert=0.10) is DriftStatus.UNAVAILABLE
    assert metric_status(math.nan, warning=0.05, alert=0.10) is DriftStatus.UNAVAILABLE

    with pytest.raises(ValueError, match="warning threshold"):
        metric_status(0.1, warning=0.2, alert=0.1)


def test_status_value_serialises_optional_enums() -> None:
    assert status_value(DriftStatus.WARNING) == "warning"
    assert status_value(None) is None


# ---------------------------------------------------------------------------
# Reference construction: guards
# ---------------------------------------------------------------------------
def test_distribution_reference_counts_missing_values_and_requires_enough_quantiles() -> None:
    reference = build_distribution_reference(
        np.array([1.0, np.nan, 2.0, np.nan]),
        name="V1",
        bin_count=1,
        quantile_sample_size=2,
    )

    assert reference.count == 2
    assert reference.missing_rate == pytest.approx(0.5)
    assert reference.quantile_sample == (1.0, 2.0)

    with pytest.raises(ReferenceProfileError, match="quantile_sample_size"):
        build_distribution_reference([1.0, 2.0], name="V1", quantile_sample_size=1)


def test_reference_profile_rejects_missing_columns_and_empty_scores() -> None:
    with pytest.raises(ReferenceProfileError, match="missing columns"):
        build_reference_profile(
            make_frame(),
            [0.1, 0.2],
            feature_columns=("V1", "missing"),
            source="synthetic",
            model_version="model-1",
            model_sha256="a" * 64,
            threshold=0.5,
            split_strategy="temporal",
            split_seed=7,
            training_rows=10,
            sample_method="all",
        )
    with pytest.raises(ReferenceProfileError, match="scores must not be empty"):
        build_reference_profile(
            make_frame(),
            [],
            feature_columns=FEATURE_COLUMNS,
            source="synthetic",
            model_version="model-1",
            model_sha256="a" * 64,
            threshold=0.5,
            split_strategy="temporal",
            split_seed=7,
            training_rows=10,
            sample_method="all",
        )


def test_reference_profile_rejects_duplicate_feature_columns() -> None:
    """A duplicate column would be written successfully but rejected when loaded."""
    with pytest.raises(ReferenceProfileError, match="unique"):
        build_reference_profile(
            make_frame(),
            [0.1, 0.2, 0.8, 0.9],
            feature_columns=("V1", "V1"),
            source="synthetic",
            model_version="model-1",
            model_sha256="a" * 64,
            threshold=0.5,
            split_strategy="temporal",
            split_seed=7,
            training_rows=10,
            sample_method="all",
        )


def test_reference_profile_rejects_more_scored_rows_than_training_rows() -> None:
    with pytest.raises(ReferenceProfileError, match="scored_rows"):
        build_reference_profile(
            make_frame(),
            [0.1, 0.2, 0.8, 0.9],
            feature_columns=FEATURE_COLUMNS,
            source="synthetic",
            model_version="model-1",
            model_sha256="a" * 64,
            threshold=0.5,
            split_strategy="temporal",
            split_seed=7,
            training_rows=2,
            sample_method="all",
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"training_rows": 0}, "training_rows"),
        ({"split_seed": -1}, "split_seed"),
        ({"threshold": 1.5}, "threshold"),
    ],
)
def test_reference_profile_builder_enforces_loadable_metadata(
    overrides: dict[str, Any],
    message: str,
) -> None:
    arguments: dict[str, Any] = {
        "feature_columns": FEATURE_COLUMNS,
        "source": "synthetic",
        "model_version": "model-1",
        "model_sha256": "a" * 64,
        "threshold": 0.5,
        "split_strategy": "temporal",
        "split_seed": 7,
        "training_rows": 10,
        "sample_method": "all",
    }
    arguments.update(overrides)

    with pytest.raises(ReferenceProfileError, match=message):
        build_reference_profile(make_frame(), [0.1, 0.2, 0.8, 0.9], **arguments)


def test_reference_profile_builder_requires_a_timezone_aware_generation_time() -> None:
    with pytest.raises(ReferenceProfileError, match="timezone"):
        build_reference_profile(
            make_frame(),
            [0.1, 0.2, 0.8, 0.9],
            feature_columns=FEATURE_COLUMNS,
            source="synthetic",
            model_version="model-1",
            model_sha256="a" * 64,
            threshold=0.5,
            split_strategy="temporal",
            split_seed=7,
            training_rows=10,
            sample_method="all",
            generated_at=NOW.replace(tzinfo=None),
        )


def test_reference_profile_from_dict_rejects_unsupported_schema_and_malformed_sections() -> None:
    payload = make_payload()
    payload["schema_version"] = 99
    with pytest.raises(ReferenceProfileError, match="unsupported"):
        ReferenceProfile.from_dict(payload)

    payload = make_payload()
    payload["features"] = []
    with pytest.raises(ReferenceProfileError, match="must not be empty"):
        ReferenceProfile.from_dict(payload)

    payload = make_payload()
    payload["features"][0]["name"] = payload["features"][1]["name"]
    with pytest.raises(ReferenceProfileError, match="unique"):
        ReferenceProfile.from_dict(payload)

    payload = make_payload()
    payload["features"] = [42]
    with pytest.raises(ReferenceProfileError, match="must be a JSON object"):
        ReferenceProfile.from_dict(payload)


def test_reference_profile_from_dict_rejects_invalid_distribution_values() -> None:
    payload = make_payload()
    payload["features"][0]["bin_edges"] = [0.0, 0.0]
    with pytest.raises(ReferenceProfileError, match="strictly increasing"):
        ReferenceProfile.from_dict(payload)

    payload = make_payload()
    payload["features"][0]["bin_proportions"] = [0.5]
    with pytest.raises(ReferenceProfileError, match="len\\(bin_edges\\) \\+ 1"):
        ReferenceProfile.from_dict(payload)

    payload = make_payload()
    payload["features"][0]["bin_proportions"] = [0.2, 0.2]
    with pytest.raises(ReferenceProfileError, match="sum to 1"):
        ReferenceProfile.from_dict(payload)

    payload = make_payload()
    payload["features"][0]["quantile_sample"] = []
    with pytest.raises(ReferenceProfileError, match="must not be empty"):
        ReferenceProfile.from_dict(payload)

    payload = make_payload()
    payload["features"][0]["quantile_sample"] = [1.0, 0.0]
    with pytest.raises(ReferenceProfileError, match="must be sorted"):
        ReferenceProfile.from_dict(payload)

    payload = make_payload()
    payload["features"][0]["mean"] = float("nan")
    with pytest.raises(ReferenceProfileError, match="finite"):
        ReferenceProfile.from_dict(payload)


def test_reference_profile_from_dict_rejects_naive_time_and_tampering() -> None:
    payload = make_payload()
    payload["generated_at"] = "2026-10-23T12:00:00"
    with pytest.raises(ReferenceProfileError, match="timezone"):
        ReferenceProfile.from_dict(payload)

    payload = make_payload()
    payload["source"] = "edited after generation"
    with pytest.raises(ReferenceProfileError, match="reference_id"):
        ReferenceProfile.from_dict(payload)


def test_load_reference_profile_reports_missing_and_invalid_files(tmp_path: Path) -> None:
    with pytest.raises(ReferenceProfileError, match="not found"):
        load_reference_profile(tmp_path / "missing.json")

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{not json", encoding="utf-8")
    with pytest.raises(ReferenceProfileError, match="could not read"):
        load_reference_profile(invalid)

    array = tmp_path / "array.json"
    array.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(ReferenceProfileError, match="JSON object"):
        load_reference_profile(array)
