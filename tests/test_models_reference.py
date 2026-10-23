"""Unit tests for the offline reference-build CLI.

The dataset, split and model bundle are small fakes: this suite pins the orchestration and the
deterministic sampling rule, not the real artifact. The actual reference builder is covered by
`tests/test_monitoring.py`.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from fraud_risk.models import reference as reference_cli
from fraud_risk.models.splitting import SplitConfig, SplitStrategy


class FakePipeline:
    def predict_proba(self, frame: pd.DataFrame) -> NDArray[np.float64]:
        scores = np.linspace(0.01, 0.99, len(frame))
        return np.column_stack([1.0 - scores, scores])


def test_reference_sampling_is_bounded_and_deterministic() -> None:
    indices, method = reference_cli._sample_indices(100, 10)

    assert indices.tolist() == [0, 11, 22, 33, 44, 55, 66, 77, 88, 99]
    assert method == "evenly spaced 10 of 100 training-split rows"
    assert reference_cli._sample_indices(5, 10)[0].tolist() == [0, 1, 2, 3, 4]


def test_build_reference_uses_the_training_split_and_model_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame({"V1": np.linspace(-1.0, 1.0, 30), "Amount": np.linspace(0.0, 90.0, 30)})
    train = SimpleNamespace(X=frame.iloc[:20], row_count=20)
    splits = SimpleNamespace(
        train=train,
        config=SplitConfig(strategy=SplitStrategy.TEMPORAL, seed=11),
    )
    model = SimpleNamespace(
        feature_columns=("V1", "Amount"),
        pipeline=FakePipeline(),
        model_version="test-model",
        sha256="b" * 64,
        threshold=0.5,
    )
    monkeypatch.setattr(reference_cli, "load_csv", lambda _path: frame)
    monkeypatch.setattr(reference_cli, "validate_dataset", lambda _frame, *, path: None)
    monkeypatch.setattr(reference_cli, "split_dataset", lambda _frame, _config: splits)
    monkeypatch.setattr(reference_cli, "load_model_bundle", lambda _path: model)

    profile = reference_cli.build_reference(
        dataset_path=Path("synthetic.csv"),
        artifact_path=Path("model.joblib"),
        strategy=SplitStrategy.TEMPORAL,
        seed=11,
        max_rows=5,
        bin_count=4,
    )

    assert [feature.name for feature in profile.features] == ["V1", "Amount"]
    assert profile.scored_rows == 5
    assert profile.split_strategy == "temporal"
    assert profile.split_seed == 11
    assert "evenly spaced 5 of 20" in profile.sample_method


def test_build_reference_rejects_invalid_bounds_before_loading() -> None:
    with pytest.raises(ValueError, match="max-rows"):
        reference_cli.build_reference(
            dataset_path=Path("unused"),
            artifact_path=Path("unused"),
            strategy=SplitStrategy.TEMPORAL,
            seed=1,
            max_rows=-1,
            bin_count=10,
        )
    with pytest.raises(ValueError, match="bin-count"):
        reference_cli.build_reference(
            dataset_path=Path("unused"),
            artifact_path=Path("unused"),
            strategy=SplitStrategy.TEMPORAL,
            seed=1,
            max_rows=0,
            bin_count=0,
        )
