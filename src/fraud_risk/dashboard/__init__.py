"""Streamlit analytics dashboard (architecture area 7).

Responsibility
    Read-only views over stored predictions plus a public transaction-scoring form. The
    dashboard talks to the versioned HTTP API and never imports model internals or database
    code.

Boundary
    ``api_client`` owns HTTP and response validation, ``transformations`` owns deterministic
    table/count logic, ``charts`` owns Plotly figures, and ``app`` owns Streamlit rendering.
    There is no demo dataset: an unreachable API or an empty history is shown as such.

Status
    Overview, filtered history, scoring, explanation and analyst-review views are implemented.
    Drift-history views remain future M7 work.
"""
