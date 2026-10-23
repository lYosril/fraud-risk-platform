"""First-party drift statistics.

Responsibility
    Compare a recent window against the training reference distribution and produce the
    metrics a monitoring report needs:

    - Population Stability Index (PSI) on fixed reference quantile bins.
    - Kolmogorov-Smirnov (KS) two-sample statistic and an asymptotic p-value.
    - Missing-value rates and threshold classification.

Design note
    These statistics are implemented here rather than delegated to a framework so that the
    thresholds are version-controlled and the code is unit-testable against distributions
    whose PSI is known analytically. Evidently is an optional extra, used only to render rich
    HTML reports -- not to make the alert decision (architecture decision D8).

Rules
    - A breach is a prompt to investigate, never an automatic retrain trigger.
    - Thresholds come from ``Settings``, not literals inside analysis scripts.
    - PSI bins are learned from the reference distribution and reused unchanged for the
      current window, otherwise the two proportions are not comparable.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import StrEnum
from typing import TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

#: A zero cell would make PSI infinite. The floor is small relative to any realistic bin
#: share; it is applied to both vectors and the vectors are renormalised afterwards.
DEFAULT_PSI_EPSILON = 1e-6
DEFAULT_BIN_COUNT = 10
DEFAULT_QUANTILE_SAMPLE_SIZE = 101

T = TypeVar("T", bound=StrEnum)


class DriftStatus(StrEnum):
    """Alert state for one metric or for the report as a whole."""

    OK = "ok"
    WARNING = "warning"
    ALERT = "alert"
    UNAVAILABLE = "unavailable"


def finite_values(values: ArrayLike) -> NDArray[np.float64]:
    """Return a 1-D float array with non-finite values removed."""
    array = np.asarray(values, dtype="float64").reshape(-1)
    finite = array[np.isfinite(array)]
    return np.asarray(finite, dtype="float64")


def quantile_edges(values: ArrayLike, *, bin_count: int = DEFAULT_BIN_COUNT) -> NDArray[np.float64]:
    """Return internal quantile cut points for ``bin_count`` reference bins.

    The returned edges are strictly increasing. Fewer than ``bin_count - 1`` edges means the
    reference distribution has repeated quantiles (for example a constant feature); callers
    must use the returned edge count, not the requested count.
    """
    if bin_count < 1:
        msg = "bin_count must be at least 1"
        raise ValueError(msg)
    array = finite_values(values)
    if array.size == 0:
        msg = "cannot compute quantile edges from an empty or non-finite sample"
        raise ValueError(msg)
    if bin_count == 1:
        return np.array([], dtype="float64")
    quantiles = np.linspace(0.0, 1.0, bin_count + 1)[1:-1]
    return np.unique(np.quantile(array, quantiles))


def bin_indices(values: ArrayLike, edges: ArrayLike) -> NDArray[np.int64]:
    """Map values to bins defined by strictly increasing internal cut points."""
    array = np.asarray(values, dtype="float64").reshape(-1)
    cut_points = np.asarray(edges, dtype="float64").reshape(-1)
    if cut_points.size:
        if not np.isfinite(cut_points).all():
            msg = "bin edges must all be finite"
            raise ValueError(msg)
        if np.any(np.diff(cut_points) <= 0.0):
            msg = "bin edges must be strictly increasing"
            raise ValueError(msg)
    if not np.isfinite(array).all():
        msg = "values must all be finite; handle missing values before binning"
        raise ValueError(msg)
    return np.searchsorted(cut_points, array, side="right").astype("int64")


def bin_proportions(values: ArrayLike, edges: ArrayLike) -> NDArray[np.float64]:
    """Return the share of finite values in each ``len(edges) + 1`` bin."""
    array = finite_values(values)
    if array.size == 0:
        msg = "cannot compute bin proportions from an empty or non-finite sample"
        raise ValueError(msg)
    indices = bin_indices(array, edges)
    counts = np.bincount(indices, minlength=np.asarray(edges).size + 1)
    return counts.astype("float64") / float(array.size)


def psi_from_proportions(
    reference: ArrayLike,
    current: ArrayLike,
    *,
    epsilon: float = DEFAULT_PSI_EPSILON,
) -> float:
    """Compute PSI from two aligned bin proportions.

    The inputs are normalised, so counts and proportions are both accepted. Empty bins are
    floored at ``epsilon`` before the logarithm; without that floor PSI is infinite whenever
    the current window has no observation in a reference bin. The floor is part of the
    documented definition, not a hidden smoothing choice.
    """
    expected = np.asarray(reference, dtype="float64").reshape(-1)
    actual = np.asarray(current, dtype="float64").reshape(-1)
    if expected.size == 0 or actual.size == 0:
        msg = "PSI needs two non-empty vectors"
        raise ValueError(msg)
    if expected.shape != actual.shape:
        msg = (
            "reference and current proportions must have the same length "
            f"(got {expected.size} and {actual.size})"
        )
        raise ValueError(msg)
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        msg = "PSI inputs must be finite"
        raise ValueError(msg)
    if (expected < 0.0).any() or (actual < 0.0).any():
        msg = "PSI inputs must be non-negative"
        raise ValueError(msg)
    if expected.sum() <= 0.0 or actual.sum() <= 0.0:
        msg = "PSI inputs must have positive mass"
        raise ValueError(msg)
    if not 0.0 < epsilon < 1.0:
        msg = "epsilon must be strictly between 0 and 1"
        raise ValueError(msg)

    expected = expected / expected.sum()
    actual = actual / actual.sum()
    expected = np.maximum(expected, epsilon)
    actual = np.maximum(actual, epsilon)
    expected = expected / expected.sum()
    actual = actual / actual.sum()
    contributions = (actual - expected) * np.log(actual / expected)
    return float(np.sum(contributions))


def population_stability_index(
    reference: ArrayLike,
    current: ArrayLike,
    *,
    bin_count: int = DEFAULT_BIN_COUNT,
    epsilon: float = DEFAULT_PSI_EPSILON,
) -> float:
    """Compute PSI using quantile bins learned from the reference sample."""
    edges = quantile_edges(reference, bin_count=bin_count)
    expected = bin_proportions(reference, edges)
    actual = bin_proportions(current, edges)
    return psi_from_proportions(expected, actual, epsilon=epsilon)


def ks_statistic(reference: ArrayLike, current: ArrayLike) -> float:
    """Return the two-sample Kolmogorov-Smirnov statistic.

    Non-finite values are removed first. The result is exact for the samples supplied; when
    the reference is stored as a quantile sample, the statistic is an approximation.
    """
    ref = finite_values(reference)
    cur = finite_values(current)
    if ref.size == 0 or cur.size == 0:
        msg = "KS needs two non-empty finite samples"
        raise ValueError(msg)
    ref = np.sort(ref)
    cur = np.sort(cur)
    support = np.concatenate([ref, cur])
    support.sort()
    ref_cdf = np.searchsorted(ref, support, side="right") / float(ref.size)
    cur_cdf = np.searchsorted(cur, support, side="right") / float(cur.size)
    return float(np.max(np.abs(ref_cdf - cur_cdf)))


def ks_p_value(statistic: float, reference_size: int, current_size: int) -> float:
    """Return the asymptotic two-sided KS p-value for the supplied effective sample sizes.

    This is the standard Kolmogorov limiting distribution with the small-sample correction
    used by common implementations. It is reported as a diagnostic, not as a decision rule:
    with a large window almost any small distribution change becomes "significant".
    """
    if reference_size < 1 or current_size < 1:
        msg = "KS sample sizes must be positive"
        raise ValueError(msg)
    if not math.isfinite(statistic) or statistic < 0.0 or statistic > 1.0:
        msg = "KS statistic must be finite and in [0, 1]"
        raise ValueError(msg)
    if statistic == 0.0:
        return 1.0
    if statistic >= 1.0:
        return 0.0

    effective = math.sqrt(
        (float(reference_size) * float(current_size)) / float(reference_size + current_size)
    )
    lam = (effective + 0.12 + 0.11 / effective) * statistic
    total = 0.0
    for index in range(1, 101):
        term = math.exp(-2.0 * float(index * index) * lam * lam)
        total += term if index % 2 == 1 else -term
        if term < 1e-12:
            break
    return float(min(1.0, max(0.0, 2.0 * total)))


def missing_rate(values: Sequence[object]) -> float:
    """Return the share of missing (``None`` or non-finite number) values."""
    if len(values) == 0:
        msg = "cannot compute a missing rate for an empty sample"
        raise ValueError(msg)
    missing = 0
    for value in values:
        if value is None:
            missing += 1
            continue
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and not math.isfinite(float(value))
        ):
            missing += 1
    return missing / len(values)


def metric_status(
    value: float | None,
    *,
    warning: float,
    alert: float,
) -> DriftStatus:
    """Classify one metric against configured warning and alert bands."""
    if value is None or not math.isfinite(value):
        return DriftStatus.UNAVAILABLE
    if warning >= alert:
        msg = "warning threshold must be below alert threshold"
        raise ValueError(msg)
    if value >= alert:
        return DriftStatus.ALERT
    if value >= warning:
        return DriftStatus.WARNING
    return DriftStatus.OK


def combine_statuses(statuses: Sequence[DriftStatus]) -> DriftStatus:
    """Combine section statuses, keeping the most severe available state."""
    available = [status for status in statuses if status is not DriftStatus.UNAVAILABLE]
    if not available:
        return DriftStatus.UNAVAILABLE
    if DriftStatus.ALERT in available:
        return DriftStatus.ALERT
    if DriftStatus.WARNING in available:
        return DriftStatus.WARNING
    return DriftStatus.OK


def status_value(value: T | None) -> str | None:
    """Serialise an optional string enum for a JSON report."""
    return None if value is None else value.value


__all__ = [
    "DEFAULT_BIN_COUNT",
    "DEFAULT_PSI_EPSILON",
    "DEFAULT_QUANTILE_SAMPLE_SIZE",
    "DriftStatus",
    "bin_indices",
    "bin_proportions",
    "combine_statuses",
    "finite_values",
    "ks_p_value",
    "ks_statistic",
    "metric_status",
    "missing_rate",
    "population_stability_index",
    "psi_from_proportions",
    "quantile_edges",
    "status_value",
]
