"""Plotly figures for the dashboard.

The functions return plain ``plotly.graph_objects.Figure`` objects; only ``app.py`` renders
them. Charts are built from the typed records and aggregate rows, never from model internals.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
import plotly.graph_objects as go

FRAUD_COLOR = "#b42318"
LEGITIMATE_COLOR = "#067647"
NEUTRAL_COLOR = "#475467"
GRID_COLOR = "#eaecf0"


def risk_distribution_figure(
    rows: Sequence[Mapping[str, Any]],
    *,
    threshold: float | None = None,
) -> go.Figure:
    """Render the fixed-width probability distribution, with an optional threshold line."""
    if not rows:
        return _empty_figure("No score distribution available")

    midpoints = [float(row["midpoint"]) for row in rows]
    counts = [int(row["count"]) for row in rows]
    labels = [str(row["label"]) for row in rows]
    widths = [float(row["width"]) for row in rows]

    figure = go.Figure(
        go.Bar(
            x=midpoints,
            y=counts,
            width=widths,
            marker_color=NEUTRAL_COLOR,
            customdata=labels,
            hovertemplate="Score %{customdata}<br>Predictions %{y}<extra></extra>",
        )
    )
    figure.update_layout(
        template="plotly_white",
        xaxis={
            "title": "Predicted fraud probability",
            "tickvals": midpoints,
            "ticktext": labels,
            "tickformat": ".0%",
            "range": [0, 1],
        },
        yaxis={"title": "Predictions", "rangemode": "tozero"},
        bargap=0.05,
        height=320,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        showlegend=False,
    )
    if threshold is not None and math.isfinite(threshold) and 0.0 <= threshold <= 1.0:
        figure.add_vline(
            x=threshold,
            line_dash="dash",
            line_color=FRAUD_COLOR,
            annotation_text=f"threshold {threshold:.4f}",
            annotation_position="top",
        )
    return figure


def decision_mix_figure(counts: Mapping[str, int]) -> go.Figure:
    """Render the flagged/legitimate decision split."""
    values = [int(value) for value in counts.values()]
    if sum(values) == 0:
        return _empty_figure("No decisions recorded")
    return go.Figure(
        go.Pie(
            labels=list(counts),
            values=values,
            hole=0.55,
            marker_colors=[FRAUD_COLOR, LEGITIMATE_COLOR],
            textinfo="label+value",
            hovertemplate="%{label}: %{value}<extra></extra>",
        )
    ).update_layout(
        template="plotly_white",
        height=320,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        showlegend=False,
    )


def review_mix_figure(counts: Mapping[str, int]) -> go.Figure:
    """Render the analyst review queue split."""
    labels = list(counts)
    values = [int(value) for value in counts.values()]
    if sum(values) == 0:
        return _empty_figure("No review states recorded")
    figure = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=[NEUTRAL_COLOR, FRAUD_COLOR, LEGITIMATE_COLOR],
            hovertemplate="%{x}: %{y}<extra></extra>",
        )
    )
    figure.update_layout(
        template="plotly_white",
        xaxis={"title": None},
        yaxis={"title": "Predictions", "rangemode": "tozero"},
        height=320,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        showlegend=False,
    )
    return figure


def volume_figure(frame: pd.DataFrame) -> go.Figure:
    """Render hourly prediction volume (UTC) from a recent-history page."""
    if frame.empty:
        return _empty_figure("No prediction volume available")
    figure = go.Figure(
        go.Bar(
            x=frame["hour"],
            y=frame["predictions"],
            marker_color=NEUTRAL_COLOR,
            hovertemplate="%{x}<br>Predictions %{y}<extra></extra>",
        )
    )
    figure.update_layout(
        template="plotly_white",
        xaxis={"title": "Hour (UTC)"},
        yaxis={"title": "Predictions", "rangemode": "tozero"},
        height=320,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        showlegend=False,
        hovermode="x unified",
    )
    return figure


def explanation_figure(frame: pd.DataFrame) -> go.Figure:
    """Render the top-k SHAP contributions as a horizontal bar chart."""
    if frame.empty:
        return _empty_figure("No explanation contributions returned")
    colors = [
        FRAUD_COLOR if float(value) >= 0 else LEGITIMATE_COLOR for value in frame["contribution"]
    ]
    figure = go.Figure(
        go.Bar(
            x=frame["contribution"],
            y=frame["feature"],
            orientation="h",
            marker_color=colors,
            customdata=frame["direction"],
            hovertemplate="%{y}<br>Contribution %{x:.4f}<br>%{customdata}<extra></extra>",
        )
    )
    figure.update_layout(
        template="plotly_white",
        xaxis={"title": "Contribution (explained output space)", "zeroline": True},
        yaxis={"title": None},
        height=max(320, 36 * len(frame)),
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        showlegend=False,
    )
    return figure


def _empty_figure(message: str) -> go.Figure:
    """Return a chart-shaped, clearly empty state rather than fabricated data."""
    figure = go.Figure()
    figure.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 14, "color": NEUTRAL_COLOR},
    )
    figure.update_layout(
        template="plotly_white",
        height=320,
        xaxis={"visible": False},
        yaxis={"visible": False},
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
    )
    return figure


__all__ = [
    "decision_mix_figure",
    "explanation_figure",
    "review_mix_figure",
    "risk_distribution_figure",
    "volume_figure",
]
