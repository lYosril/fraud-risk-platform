"""Build the compact training reference profile used by monitoring.

Responsibility
    Orchestrate the offline reference build: load the validated dataset, isolate the training
    split, load the frozen model bundle, score the training rows and write the aggregate profile.

Boundary
    This module belongs to the offline model layer, so it may import the data, splitting and
    bundle code *and* the leaf ``monitoring.reference`` builder. The reporting path in
    ``fraud_risk.monitoring`` never imports back into ``models``.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from fraud_risk.config import Settings, get_settings
from fraud_risk.data.dataset import DEFAULT_DATASET_PATH
from fraud_risk.data.errors import DataError
from fraud_risk.data.ingestion import load_csv
from fraud_risk.data.validation import validate_dataset
from fraud_risk.models.bundle import ModelBundleError, load_model_bundle
from fraud_risk.models.splitting import (
    SplitConfig,
    SplitError,
    SplitStrategy,
    split_dataset,
)
from fraud_risk.monitoring.reference import (
    ReferenceProfile,
    ReferenceProfileError,
    build_reference_profile,
    write_reference_profile,
)

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_INVALID = 1
EXIT_MISSING = 2


def build_parser() -> argparse.ArgumentParser:
    """Build the reference-profile CLI parser."""
    parser = argparse.ArgumentParser(
        prog="python -m fraud_risk.models.reference",
        description=(
            "Build a compact, aggregate reference profile from the training split for "
            "fraud_risk.monitoring."
        ),
        epilog=(
            "The reference must be built from the training split only. It contains aggregate "
            "statistics, never transaction rows."
        ),
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"validated dataset file (default: {DEFAULT_DATASET_PATH})",
    )
    parser.add_argument(
        "--model-artifact",
        type=Path,
        default=None,
        help="frozen model bundle (default: Settings.model_artifact_path)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="reference profile path (default: Settings.monitoring_reference_path)",
    )
    parser.add_argument(
        "--strategy",
        choices=[strategy.value for strategy in SplitStrategy],
        default=SplitStrategy.TEMPORAL.value,
        help="split strategy used to isolate the training rows (default: temporal)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="split seed (default: Settings.random_seed)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="score at most this many evenly spaced training rows; 0 means all (default: 0)",
    )
    parser.add_argument(
        "--bin-count",
        type=int,
        default=10,
        help="reference quantile bins per distribution (default: 10)",
    )
    return parser


def build_reference(
    *,
    dataset_path: Path,
    artifact_path: Path,
    strategy: SplitStrategy,
    seed: int,
    max_rows: int,
    bin_count: int,
) -> ReferenceProfile:
    """Build a reference profile from the training split without writing it."""
    if max_rows < 0:
        msg = "--max-rows must be zero or positive"
        raise ValueError(msg)
    if bin_count < 1:
        msg = "--bin-count must be at least 1"
        raise ValueError(msg)

    frame = load_csv(dataset_path)
    validate_dataset(frame, path=dataset_path)
    splits = split_dataset(frame, SplitConfig(strategy=strategy, seed=seed))
    model = load_model_bundle(artifact_path)
    missing = [column for column in model.feature_columns if column not in splits.train.X]
    if missing:
        msg = "the model requires feature columns absent from the training split: " + ", ".join(
            missing
        )
        raise ReferenceProfileError(msg)
    training = splits.train.X.loc[:, list(model.feature_columns)]
    indices, sample_method = _sample_indices(len(training), max_rows)
    sampled = training.iloc[indices]
    probabilities = model.pipeline.predict_proba(sampled)
    scores = np.asarray(probabilities, dtype="float64")
    if scores.ndim != 2 or scores.shape[1] != 2:
        msg = "the frozen pipeline did not return a two-column predict_proba result"
        raise ReferenceProfileError(msg)
    return build_reference_profile(
        sampled,
        scores[:, 1],
        feature_columns=model.feature_columns,
        source=f"{dataset_path} training split",
        model_version=model.model_version,
        model_sha256=model.sha256,
        threshold=model.threshold,
        split_strategy=splits.config.strategy.value,
        split_seed=splits.config.seed,
        training_rows=splits.train.row_count,
        sample_method=sample_method,
        bin_count=bin_count,
    )


def main(argv: Sequence[str] | None = None, *, settings: Settings | None = None) -> int:
    """Run the reference-profile CLI and return a process exit code."""
    options = build_parser().parse_args(argv)
    resolved_settings = settings or get_settings()
    logging.basicConfig(level=resolved_settings.log_level)
    path: Path = options.path
    if not path.is_file():
        logger.error(
            "dataset not found: %s; place it at data/raw/creditcard.csv or pass --path",
            path,
        )
        return EXIT_MISSING
    artifact = Path(options.model_artifact or resolved_settings.model_artifact_path)
    if not artifact.is_file():
        logger.error("model artifact not found: %s", artifact)
        return EXIT_MISSING
    seed = resolved_settings.random_seed if options.seed is None else int(options.seed)
    output = Path(options.output or resolved_settings.monitoring_reference_path)
    try:
        profile = build_reference(
            dataset_path=path,
            artifact_path=artifact,
            strategy=SplitStrategy(options.strategy),
            seed=seed,
            max_rows=int(options.max_rows),
            bin_count=int(options.bin_count),
        )
    except (
        DataError,
        SplitError,
        ModelBundleError,
        ReferenceProfileError,
        OSError,
        ValueError,
    ) as exc:
        logger.error("reference construction failed: %s", exc)
        return EXIT_INVALID
    try:
        write_reference_profile(profile, output)
    except (OSError, ValueError) as exc:
        logger.error("could not write reference profile: %s", exc)
        return EXIT_INVALID
    logger.info(
        "wrote reference profile %s (%d features, %d scored training rows, id %s)",
        output,
        len(profile.features),
        profile.scored_rows,
        profile.reference_id,
    )
    return EXIT_OK


def _sample_indices(row_count: int, max_rows: int) -> tuple[NDArray[np.int64], str]:
    """Choose all rows or a deterministic evenly spaced subset of the training split."""
    if max_rows == 0 or max_rows >= row_count:
        return np.arange(row_count, dtype="int64"), "all training-split rows"
    indices = np.unique(np.linspace(0, row_count - 1, max_rows).round().astype("int64"))
    return indices, f"evenly spaced {indices.size} of {row_count} training-split rows"


__all__ = ["build_parser", "build_reference", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
