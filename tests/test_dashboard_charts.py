"""Tests for the pure Plotly dashboard charts.

The figures are inspected directly; no browser, Streamlit runtime or API is involved. The tests
pin empty states, trace data, threshold rendering and label-based colours so a mapping-order
change cannot silently swap fraud and legitimate colours.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fraud_risk.dashboard.charts import (
    FRAUD_COLOR,
    LEGITIMATE_COLOR,
    NEUTRAL_COLOR,
    decision_mix_figure,
    explanation_figure,
    review_mix_figure,
    risk_distribution_figure,
    volume_figure,
)


def test_risk_distribution_chart_uses_the_bin_midpoints_widths_and_threshold() -> None:
    rows = [
        {"label": "0-10%", "midpoint": 0.05, "width": 0.1, "count": 3},
        {"label": "90-100%", "midpoint": 0.95, "width": 0.1, "count": 1},
    ]

    figure = risk_distribution_figure(rows, threshold=0.5)

    bar = figure.data[0]
    assert list(bar.x) == [0.05, 0.95]
    assert list(bar.y) == [3, 1]
    assert list(bar.width) == pytest.approx([0.1, 0.1])
    assert list(bar.customdata) == ["0-10%", "90-100%"]
    assert list(figure.layout.xaxis.tickvals) == [0.05, 0.95]
    assert list(figure.layout.xaxis.ticktext) == ["0-10%", "90-100%"]
    assert len(figure.layout.shapes) == 1
    assert figure.layout.shapes[0].x0 == pytest.approx(0.5)


def test_risk_distribution_chart_ignores_an_invalid_threshold_and_has_an_empty_state() -> None:
    rows = [{"label": "0-10%", "midpoint": 0.05, "width": 0.1, "count": 3}]

    assert len(risk_distribution_figure(rows, threshold=1.5).layout.shapes) == 0
    assert len(risk_distribution_figure(rows, threshold=float("nan")).layout.shapes) == 0
    assert len(risk_distribution_figure(rows, threshold=None).layout.shapes) == 0

    empty = risk_distribution_figure([])
    assert len(empty.data) == 0
    assert empty.layout.annotations[0].text == "No score distribution available"


def test_decision_mix_colours_follow_labels_not_mapping_order() -> None:
    figure = decision_mix_figure({"Legitimate": 3, "Flagged fraud": 1})

    pie = figure.data[0]
    assert list(pie.labels) == ["Legitimate", "Flagged fraud"]
    assert list(pie.values) == [3, 1]
    assert list(pie.marker.colors) == [LEGITIMATE_COLOR, FRAUD_COLOR]

    empty = decision_mix_figure({"Legitimate": 0, "Flagged fraud": 0})
    assert len(empty.data) == 0
    assert empty.layout.annotations[0].text == "No decisions recorded"


def test_review_mix_colours_follow_labels_not_mapping_order() -> None:
    figure = review_mix_figure(
        {
            "Confirmed legitimate": 1,
            "Pending": 2,
            "Confirmed fraud": 3,
        }
    )

    bar = figure.data[0]
    assert list(bar.x) == ["Confirmed legitimate", "Pending", "Confirmed fraud"]
    assert list(bar.y) == [1, 2, 3]
    assert list(bar.marker.color) == [LEGITIMATE_COLOR, NEUTRAL_COLOR, FRAUD_COLOR]

    empty = review_mix_figure({"Pending": 0, "Confirmed fraud": 0})
    assert len(empty.data) == 0
    assert empty.layout.annotations[0].text == "No review states recorded"


def test_volume_chart_uses_the_transformed_hourly_frame() -> None:
    frame = pd.DataFrame(
        {
            "hour": pd.to_datetime(["2026-10-23T10:00:00Z", "2026-10-23T11:00:00Z"]),
            "predictions": [4, 6],
        }
    )

    figure = volume_figure(frame)

    bar = figure.data[0]
    assert list(bar.y) == [4, 6]
    assert figure.layout.hovermode == "x unified"

    empty = volume_figure(pd.DataFrame(columns=["hour", "predictions"]))
    assert len(empty.data) == 0
    assert empty.layout.annotations[0].text == "No prediction volume available"


def test_explanation_chart_uses_signed_colours_and_grows_with_the_row_count() -> None:
    frame = pd.DataFrame(
        {
            "feature": ["V1", "V2"],
            "contribution": [0.4, -0.8],
            "direction": ["increases risk", "decreases risk"],
        }
    )

    figure = explanation_figure(frame)

    bar = figure.data[0]
    assert list(bar.x) == [0.4, -0.8]
    assert list(bar.y) == ["V1", "V2"]
    assert list(bar.marker.color) == [FRAUD_COLOR, LEGITIMATE_COLOR]
    assert figure.layout.height == 320

    empty = explanation_figure(pd.DataFrame(columns=["feature", "contribution", "direction"]))
    assert len(empty.data) == 0
    assert empty.layout.annotations[0].text == "No explanation contributions returned"
