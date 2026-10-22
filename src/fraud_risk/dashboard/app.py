"""Streamlit entry point for the fraud-risk analytics dashboard.

Responsibility
    Render the real prediction history and the public scoring form. All data comes from the
    FastAPI backend over HTTP; this module never loads the model, reads model artifacts or
    queries PostgreSQL directly.

Run with
    ``streamlit run src/fraud_risk/dashboard/app.py``

Configuration
    ``DASHBOARD_API_URL`` is the API origin. ``API_ADMIN_TOKEN`` enables the admin-protected
    overview, history and model-metadata routes. The token is read from ``Settings`` (and
    therefore the environment); it is never hardcoded or displayed.

Data honesty
    The dashboard does not ship a demo dataset. When the API is unreachable, the history is
    empty, or a metric is not served by the API, the UI says so instead of substituting
    synthetic numbers. Model performance metrics are unavailable because the current platform
    stores predictions but not delayed feedback labels or an evaluation endpoint.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime, time
from uuid import uuid4

import streamlit as st

from fraud_risk.config import get_settings
from fraud_risk.dashboard.api_client import (
    REVIEW_OUTCOMES,
    ApiAuthenticationError,
    ApiConflictError,
    ApiHTTPError,
    ApiValidationError,
    DashboardApiClient,
    DashboardApiError,
    ModelMetadata,
    Prediction,
    PredictionRecord,
)
from fraud_risk.dashboard.charts import (
    decision_mix_figure,
    explanation_figure,
    review_mix_figure,
    risk_distribution_figure,
    volume_figure,
)
from fraud_risk.dashboard.transformations import (
    decision_counts,
    decision_label,
    explanation_frame,
    format_percentage,
    fraud_rate,
    review_counts,
    review_status_label,
    review_table_frame,
    risk_distribution_rows,
    volume_frame,
)

logger = logging.getLogger(__name__)

PAGE_TITLE = "Fraud Risk Intelligence"
RECENT_PAGE_SIZE = 200
FEATURE_COLUMNS_PER_ROW = 4


def main() -> None:
    """Render one dashboard run."""
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    settings = get_settings()
    client = DashboardApiClient.from_settings(settings)
    _render_header(client)
    metadata, metadata_error = _load_metadata(client)

    overview_tab, review_tab, scoring_tab = st.tabs(
        ["Overview", "Analyst review", "Score a transaction"]
    )
    with overview_tab:
        _render_overview(client, metadata, metadata_error)
    with review_tab:
        _render_review_workflow(client)
    with scoring_tab:
        _render_scoring(client, metadata, metadata_error)


def _render_header(client: DashboardApiClient) -> None:
    """Render the title, data-honesty note and connection controls."""
    st.title(PAGE_TITLE)
    st.caption(
        "Reads stored predictions and scoring results from the FastAPI backend. "
        "No demo dataset or synthetic traffic is loaded."
    )
    with st.sidebar:
        st.header("API connection")
        st.code(client.base_url, language=None)
        st.caption("Configured with `DASHBOARD_API_URL`.")
        if client.admin_available:
            st.success("Admin token configured")
        else:
            st.warning(
                "`API_ADMIN_TOKEN` is not configured. The overview, history and model "
                "metadata are unavailable, but public scoring still works through the API."
            )
        try:
            health = client.health()
            readiness = client.readiness()
        except DashboardApiError as exc:
            st.error(f"API unavailable: {exc}")
        else:
            st.caption(f"API {health.version} · {health.environment}")
            if readiness.model_available:
                st.success("Model ready")
            else:
                st.warning(f"Model not ready: {readiness.detail or 'no detail provided'}")
        st.divider()
        if st.button("Refresh data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()


def _load_metadata(
    client: DashboardApiClient,
) -> tuple[ModelMetadata | None, str | None]:
    """Load the model contract once per rerun; never expose the admin token."""
    if not client.admin_available:
        return None, "API_ADMIN_TOKEN is not configured."
    try:
        return client.model_metadata(), None
    except DashboardApiError as exc:
        return None, str(exc)


def _render_overview(
    client: DashboardApiClient,
    metadata: ModelMetadata | None,
    metadata_error: str | None,
) -> None:
    """Render aggregate history, Plotly charts and the unavailable-performance state."""
    st.subheader("Prediction overview")
    if not client.admin_available:
        st.info(
            "The overview reads admin-protected prediction history. Set `API_ADMIN_TOKEN` "
            "in the dashboard environment to enable it."
        )
        _render_model_performance(metadata, metadata_error)
        return

    try:
        summary = client.prediction_summary()
        recent = client.recent_predictions(limit=RECENT_PAGE_SIZE)
    except DashboardApiError as exc:
        st.error(f"Could not load prediction history: {exc}")
        _render_model_performance(metadata, metadata_error)
        return

    metric_columns = st.columns(4)
    metric_columns[0].metric("Transactions scored", f"{summary.total_predictions:,}")
    metric_columns[1].metric("Model flagged fraud", f"{summary.fraud_predictions:,}")
    metric_columns[2].metric("Flag rate", format_percentage(fraud_rate(summary)))
    metric_columns[3].metric("Pending reviews", f"{summary.pending_reviews:,}")

    if summary.total_predictions == 0:
        st.info(
            "No predictions have been stored yet. Score a transaction from the second tab; "
            "the overview will use the stored API history, not a demo fixture."
        )

    risk_column, decision_column = st.columns(2)
    with risk_column:
        st.markdown("**Risk-score distribution**")
        threshold = metadata.threshold if metadata is not None else None
        st.plotly_chart(
            risk_distribution_figure(risk_distribution_rows(summary), threshold=threshold),
            use_container_width=True,
        )
    with decision_column:
        st.markdown("**Decision mix**")
        st.plotly_chart(
            decision_mix_figure(decision_counts(summary)),
            use_container_width=True,
        )

    st.markdown("**Analyst review states**")
    st.plotly_chart(review_mix_figure(review_counts(summary)), use_container_width=True)

    st.markdown("**Prediction volume (recent page, UTC)**")
    if recent:
        st.plotly_chart(volume_figure(volume_frame(recent)), use_container_width=True)
    else:
        st.caption("No prediction timestamps available yet.")

    _render_model_performance(metadata, metadata_error)
    _render_recent_predictions(recent)


def _render_review_workflow(client: DashboardApiClient) -> None:
    """Render the filtered review queue and one explicit analyst action."""
    st.subheader("Analyst review")
    st.caption(
        "Model predicted labels and analyst review outcomes are separate. A fraud score "
        "starts as **pending review** and is not confirmed fraud until an analyst records "
        "that outcome."
    )
    if not client.admin_available:
        st.info(
            "Review actions require the administrative API token. Set `API_ADMIN_TOKEN` in "
            "the dashboard environment; unauthorized users cannot see or submit review actions."
        )
        return

    _render_review_flash()
    filters = _render_review_filters()
    if filters is None:
        return
    created_from, created_to, min_risk, max_risk, decision, review_status = filters

    try:
        records = client.recent_predictions(
            limit=RECENT_PAGE_SIZE,
            created_from=created_from,
            created_to=created_to,
            min_risk_score=min_risk,
            max_risk_score=max_risk,
            decision=decision,
            review_status=review_status,
        )
    except DashboardApiError as exc:
        st.error(f"Could not load the filtered review queue: {exc}")
        return

    st.caption(
        f"{len(records)} matching prediction(s) on this page. Filters apply before the "
        f"{RECENT_PAGE_SIZE}-record page limit."
    )
    if not records:
        st.info("No persisted predictions match these filters.")
        return

    st.dataframe(
        review_table_frame(records),
        use_container_width=True,
        hide_index=True,
        column_config={
            "prediction_id": st.column_config.TextColumn("Prediction ID"),
            "transaction_id": st.column_config.TextColumn("Transaction ID"),
            "created_at": st.column_config.DatetimeColumn(
                "Scored at (UTC)",
                format="YYYY-MM-DD HH:mm:ss",
            ),
            "model_version": st.column_config.TextColumn("Model version"),
            "risk_score": st.column_config.NumberColumn("Model risk score", format="%.4f"),
            "threshold": st.column_config.NumberColumn("Applied threshold", format="%.4f"),
            "predicted_label": st.column_config.TextColumn("Model predicted label"),
            "review_outcome": st.column_config.TextColumn("Analyst review outcome"),
        },
    )

    pending = [record for record in records if record.review_status == "pending"]
    if not pending:
        st.info(
            "Every prediction on this page has a terminal analyst outcome. Confirmed outcomes "
            "cannot be overwritten from the dashboard."
        )
        return

    selected = st.selectbox(
        "Select a pending prediction",
        options=pending,
        index=None,
        placeholder="Choose a transaction to review",
        format_func=_review_option_label,
    )
    if selected is not None:
        _render_review_action(client, selected)


def _render_review_filters() -> (
    tuple[
        datetime | None,
        datetime | None,
        float | None,
        float | None,
        str | None,
        str | None,
    ]
    | None
):
    """Render filter controls, returning ``None`` when the selection is invalid."""
    with st.expander("Filter persisted predictions", expanded=True):
        date_columns = st.columns(2)
        from_value = date_columns[0].date_input("Scored from (UTC)", value=None)
        to_value = date_columns[1].date_input("Scored to (UTC)", value=None)
        from_date = _coerce_date(from_value)
        to_date = _coerce_date(to_value)
        if from_value is not None and from_date is None:
            st.error("Select a single 'from' date, not a date range.")
            return None
        if to_value is not None and to_date is None:
            st.error("Select a single 'to' date, not a date range.")
            return None
        if from_date is not None and to_date is not None and from_date > to_date:
            st.error("The 'from' date must not be after the 'to' date.")
            return None

        risk_range = st.slider(
            "Model risk score range",
            min_value=0.0,
            max_value=1.0,
            value=(0.0, 1.0),
            step=0.01,
        )
        filter_columns = st.columns(2)
        decision = filter_columns[0].selectbox(
            "Model predicted label",
            options=("Any", "fraud", "legitimate"),
            format_func=lambda value: "Any" if value == "Any" else decision_label(value),
        )
        review_status = filter_columns[1].selectbox(
            "Analyst review outcome",
            options=("Any", "pending", "confirmed_fraud", "confirmed_legitimate"),
            format_func=lambda value: "Any" if value == "Any" else review_status_label(value),
        )

    created_from = _start_of_day(from_date) if from_date is not None else None
    created_to = _end_of_day(to_date) if to_date is not None else None
    min_risk = float(risk_range[0]) if float(risk_range[0]) > 0.0 else None
    max_risk = float(risk_range[1]) if float(risk_range[1]) < 1.0 else None
    resolved_decision = None if decision == "Any" else decision
    resolved_review_status = None if review_status == "Any" else review_status
    return (
        created_from,
        created_to,
        min_risk,
        max_risk,
        resolved_decision,
        resolved_review_status,
    )


def _render_review_action(client: DashboardApiClient, record: PredictionRecord) -> None:
    """Show the selected record and, for a pending row, record one terminal outcome."""
    st.divider()
    st.markdown("**Selected prediction**")
    metric_columns = st.columns(2)
    metric_columns[0].metric("Model predicted label", decision_label(record.decision))
    metric_columns[1].metric("Analyst review outcome", review_status_label(record.review_status))
    st.caption(
        f"Transaction `{record.transaction_id}` · scored "
        f"{record.created_at.strftime('%Y-%m-%d %H:%M:%S %Z')} · risk score "
        f"{record.risk_score:.4f} · threshold {record.threshold:.4f} · model "
        f"`{record.model_version}`"
    )

    if record.review_status != "pending":
        st.info(
            "This prediction already has a terminal analyst outcome. Confirmed outcomes "
            "cannot be changed from the dashboard."
        )
        return

    st.caption(
        "Recording an outcome is a human decision. The model's predicted label is stored "
        "separately and is never copied into the review outcome automatically."
    )
    with st.form("record_review"):
        outcome = st.selectbox(
            "Review outcome",
            options=sorted(REVIEW_OUTCOMES),
            index=None,
            placeholder="Select confirmed fraud or confirmed legitimate",
            format_func=review_status_label,
        )
        attested = st.checkbox(
            "I reviewed this transaction and confirm this outcome reflects analyst judgment, "
            "not the model score."
        )
        submitted = st.form_submit_button("Record review", type="primary")

    if not submitted:
        return
    if outcome is None:
        st.error("Select a review outcome before submitting.")
        return
    if not attested:
        st.error("Confirm that you reviewed the transaction before recording an outcome.")
        return

    try:
        updated = client.review_prediction(record.prediction_id, review_status=outcome)
    except ApiConflictError as exc:
        st.error(f"The review transition was rejected: {exc.detail}")
    except ApiAuthenticationError as exc:
        st.error(
            "The API rejected the administrative token. Check `API_ADMIN_TOKEN` and the "
            f"API's admin configuration. ({exc.detail})"
        )
    except ApiHTTPError as exc:
        st.error(f"The API returned HTTP {exc.status_code}: {exc.detail}")
    except DashboardApiError as exc:
        st.error(f"Could not record the review: {exc}")
    else:
        st.session_state["review_flash"] = (
            f"Recorded {review_status_label(updated.review_status)} for transaction "
            f"{updated.transaction_id}."
        )
        st.rerun()


def _render_review_flash() -> None:
    """Show a one-shot success message after a review-triggered rerun."""
    message = st.session_state.pop("review_flash", None)
    if isinstance(message, str) and message:
        st.success(message)


def _review_option_label(record: PredictionRecord) -> str:
    """Describe a pending row without implying it is confirmed fraud."""
    return (
        f"{record.transaction_id} · score {record.risk_score:.3f} · "
        f"{decision_label(record.decision)} · {record.prediction_id[:8]}"
    )


def _coerce_date(value: object) -> date | None:
    """Accept the single-date value Streamlit returns, rejecting a range tuple."""
    if isinstance(value, date):
        return value
    return None


def _start_of_day(value: date) -> datetime:
    """Convert a calendar date to an inclusive, timezone-aware UTC start."""
    return datetime.combine(value, time.min, tzinfo=UTC)


def _end_of_day(value: date) -> datetime:
    """Convert a calendar date to an inclusive, timezone-aware UTC end."""
    return datetime.combine(value, time.max, tzinfo=UTC)


def _render_model_performance(
    metadata: ModelMetadata | None,
    metadata_error: str | None,
) -> None:
    """Show the loaded model identity and explicitly state the missing performance data."""
    st.subheader("Model performance")
    if metadata is not None:
        st.caption(
            f"{metadata.model_name} · {metadata.model_family} · "
            f"version `{metadata.model_version}` · threshold {metadata.threshold:.6f}"
        )
    elif metadata_error is not None:
        st.caption(f"Model metadata unavailable: {metadata_error}")

    st.info(
        "Unavailable: this platform does not yet collect delayed feedback labels or expose an "
        "evaluation endpoint. PR-AUC, recall and other performance metrics are therefore not "
        "shown, and no illustrative metric is substituted."
    )


def _render_recent_predictions(records: list[PredictionRecord]) -> None:
    """Render the non-sensitive recent-history table."""
    st.subheader("Recent predictions")
    st.caption(
        f"Most recent {RECENT_PAGE_SIZE} stored records. The API never returns feature "
        "snapshots on this route."
    )
    if not records:
        st.info("No stored predictions to display.")
        return
    st.dataframe(
        review_table_frame(records),
        use_container_width=True,
        hide_index=True,
        column_config={
            "created_at": st.column_config.DatetimeColumn(
                "Scored at (UTC)",
                format="YYYY-MM-DD HH:mm:ss",
            ),
            "risk_score": st.column_config.NumberColumn("Model risk score", format="%.4f"),
            "threshold": st.column_config.NumberColumn("Applied threshold", format="%.4f"),
            "predicted_label": st.column_config.TextColumn("Model predicted label"),
            "review_outcome": st.column_config.TextColumn("Analyst review outcome"),
        },
    )


def _render_scoring(
    client: DashboardApiClient,
    metadata: ModelMetadata | None,
    metadata_error: str | None,
) -> None:
    """Render the transaction input form and the last prediction in this session."""
    st.subheader("Score a transaction")
    st.caption(
        "The request is sent to the public `POST /api/v1/predict` route. The model identity "
        "and feature order come from the admin-protected `GET /api/v1/model` contract."
    )
    if metadata is None:
        st.warning(
            "The scoring form needs the loaded model's feature schema. Configure "
            "`API_ADMIN_TOKEN` so the dashboard can read `GET /api/v1/model`."
        )
        if metadata_error is not None:
            st.caption(f"Model metadata unavailable: {metadata_error}")
        return

    if "Amount" not in metadata.feature_columns:
        st.error(
            "This model does not declare the API's top-level `Amount` feature, so the "
            "dashboard form cannot build a valid request. Use `POST /api/v1/predict` directly."
        )
        return

    feature_names = [name for name in metadata.feature_columns if name != "Amount"]
    if "score_transaction_id" not in st.session_state:
        st.session_state["score_transaction_id"] = f"txn-{uuid4().hex[:12]}"

    with st.form("score_transaction"):
        transaction_id = st.text_input(
            "Transaction ID",
            key="score_transaction_id",
            max_chars=128,
        )
        amount = st.number_input(
            "Amount",
            min_value=0.0,
            value=42.50,
            step=1.0,
            format="%.2f",
        )
        explain = st.checkbox("Request SHAP explanation", value=False)
        with st.expander("Model features", expanded=True):
            st.caption("Features are submitted in the exact order declared by the loaded bundle.")
            feature_values: dict[str, float] = {}
            feature_columns = st.columns(FEATURE_COLUMNS_PER_ROW)
            for index, name in enumerate(feature_names):
                with feature_columns[index % FEATURE_COLUMNS_PER_ROW]:
                    feature_values[name] = st.number_input(
                        name,
                        value=0.0,
                        step=0.01,
                        format="%.6f",
                        key=f"feature_{name}",
                    )
        submitted = st.form_submit_button("Score transaction", type="primary")

    if submitted:
        if not transaction_id.strip():
            st.error("Transaction ID must not be blank.")
        else:
            try:
                result = client.predict(
                    transaction_id=transaction_id,
                    amount=float(amount),
                    features=feature_values,
                    explain=bool(explain),
                )
            except ApiAuthenticationError as exc:
                st.error(
                    "The API rejected the administrative token. Check `API_ADMIN_TOKEN` and "
                    f"the API's admin configuration. ({exc.detail})"
                )
            except ApiValidationError as exc:
                st.error(f"The API rejected the transaction: {exc.detail}")
            except ApiHTTPError as exc:
                st.error(f"The API returned HTTP {exc.status_code}: {exc.detail}")
            except DashboardApiError as exc:
                st.error(f"Could not score the transaction: {exc}")
            except ValueError as exc:
                st.error(f"Invalid transaction input: {exc}")
            else:
                st.session_state["last_prediction"] = result

    last_prediction = st.session_state.get("last_prediction")
    if isinstance(last_prediction, Prediction):
        _render_prediction(last_prediction)


def _render_prediction(result: Prediction) -> None:
    """Render probability, threshold, decision and optional explanation."""
    st.divider()
    st.subheader("Last prediction in this session")
    st.caption(
        f"Transaction `{result.transaction_id}` · model {result.model_family} · "
        f"version `{result.model_version}`"
    )
    metric_columns = st.columns(3)
    metric_columns[0].metric("Fraud probability", f"{result.probability:.4%}")
    metric_columns[1].metric("Applied threshold", f"{result.threshold:.4%}")
    metric_columns[2].metric("Decision", result.decision.capitalize())

    if result.explanation is None:
        st.info(
            "No explanation was returned for this prediction. Tick “Request SHAP "
            "explanation” before scoring to receive top-k contributions."
        )
        return

    explanation = result.explanation
    st.markdown("### Explanation")
    st.caption(
        f"{explanation.explainer} · {explanation.output_space} · "
        f"{explanation.n_folds} calibration folds. Contributions are not causal evidence."
    )
    frame = explanation_frame(explanation)
    st.plotly_chart(explanation_figure(frame), use_container_width=True)
    if not frame.empty:
        st.dataframe(frame, use_container_width=True, hide_index=True)
    if explanation.limitations:
        with st.expander("Explanation limitations"):
            for limitation in explanation.limitations:
                st.markdown(f"- {limitation}")


if __name__ == "__main__":
    main()
