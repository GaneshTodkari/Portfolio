from datetime import date

import os
import pandas as pd
import requests
import streamlit as st
from requests.exceptions import RequestException, Timeout
from streamlit.errors import StreamlitSecretNotFoundError

st.set_page_config(page_title="Retail Demand Forecasting", page_icon="📈", layout="wide")


DATASET_SIZE = "1.01M rows"
MODEL_NAME = "XGBoost Regressor"
MODEL_R2 = 0.85
AVERAGE_DAILY_SALES = 5773.0


def get_backend_url() -> str:
    env_url = os.getenv("BACKEND_URL")
    if env_url:
        return env_url

    local_url = "http://127.0.0.1:8000"
    cloud_url = "https://portfolio-i8re.onrender.com"
    try:
        return st.secrets.get("BACKEND_URL", local_url)
    except StreamlitSecretNotFoundError:
        try:
            response = requests.get(f"{local_url}/health", timeout=1)
            if response.ok:
                return local_url
        except Exception:
            pass
        return cloud_url


BACKEND_URL = get_backend_url()


def call_api(path: str, payload=None, timeout: int = 20):
    url = BACKEND_URL.rstrip("/") + path
    response = None
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json(), None
    except Timeout:
        return None, "The forecasting service took too long to respond."
    except RequestException as exc:
        detail = None
        try:
            detail = response.json().get("detail") if response is not None else None
        except Exception:
            detail = str(exc)
        return None, f"Forecast request failed: {detail}"
    except Exception as exc:
        return None, f"Unexpected error: {exc}"


def render_kpi_card(title: str, value: str, caption: str):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_payload(
    store_id: int,
    forecast_dt: date,
    promo: bool,
    state_holiday: str,
    school_holiday: bool,
    store_type: str,
    assortment: str,
    competition_distance: float,
    promo2: bool,
    promo2_since_week: int,
    promo2_since_year: int,
    competition_age: int,
    lag_1_sales: float,
    lag_7_sales: float,
    rolling_7_mean_sales: float,
    customers: int,
):
    return {
        "store_id": int(store_id),
        "forecast_date": forecast_dt.isoformat(),
        "promo": int(promo),
        "competition_distance": float(competition_distance),
        "state_holiday": state_holiday,
        "school_holiday": int(school_holiday),
        "store_type": store_type,
        "assortment": assortment,
        "promo2": int(promo2),
        "promo2_since_week": int(promo2_since_week),
        "promo2_since_year": int(promo2_since_year),
        "competition_age": int(competition_age),
        "lag_1_sales": float(lag_1_sales),
        "lag_7_sales": float(lag_7_sales),
        "rolling_7_mean_sales": float(rolling_7_mean_sales),
        "customers": int(customers),
    }


def history_frame(history: list[dict]) -> pd.DataFrame:
    if not history:
        return pd.DataFrame()
    frame = pd.DataFrame(history)
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.set_index("date")
    return frame


def feature_importance_frame(items: list[dict]) -> pd.DataFrame:
    if not items:
        return pd.DataFrame(columns=["feature", "importance"])
    frame = pd.DataFrame(items)
    return frame.sort_values("importance", ascending=True)


def interpret_sales(predicted_sales: float, benchmark: dict, actual_sales: float | None) -> str:
    pct_vs_average = benchmark.get("pct_vs_average", 0.0)
    if predicted_sales <= 0:
        return "This store-day combination is expected to be closed, so no sales are forecast."
    if pct_vs_average >= 12:
        base = "Sales are materially above the network average, which suggests a strong trading day."
    elif pct_vs_average >= 0:
        base = "Sales are slightly above the network average, indicating healthy expected demand."
    elif pct_vs_average <= -12:
        base = "Sales are projected below the network average, so inventory and staffing should stay conservative."
    else:
        base = "Sales are close to the network average, so this looks like a typical trading day."

    if actual_sales is not None and actual_sales > 0:
        variance = predicted_sales - actual_sales
        if abs(variance) < 200:
            return f"{base} The demo actual value is closely aligned with the prediction, which supports model reliability for planning."
        if variance > 0:
            return f"{base} The forecast is higher than the demo actual value, so this scenario suggests upside that store teams should monitor."
        return f"{base} The forecast is below the demo actual value, which implies stronger realised demand than expected."
    return base


def scenario_delta(primary: dict | None, comparison: dict | None) -> tuple[float, float]:
    if not primary or not comparison:
        return 0.0, 0.0
    primary_sales = float(primary.get("predicted_sales", 0.0))
    comparison_sales = float(comparison.get("predicted_sales", 0.0))
    delta = comparison_sales - primary_sales
    pct = (delta / primary_sales) * 100 if primary_sales else 0.0
    return delta, pct


st.markdown(
    """
    <style>
    [data-testid="stMainBlockContainer"] {
        max-width: 1180px;
        padding-top: 1rem;
    }
    .page-kicker {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #147a66;
        font-weight: 700;
        margin-bottom: 0.45rem;
    }
    .hero-card, .panel-card, .kpi-card {
        background: var(--secondary-background-color);
        border: 1px solid rgba(127, 127, 127, 0.18);
        box-shadow: 0 12px 28px rgba(15, 23, 42, 0.06);
        border-radius: 16px;
    }
    .hero-card {
        padding: 1.4rem 1.5rem;
        margin-bottom: 1rem;
    }
    .hero-title {
        font-size: 2.35rem;
        line-height: 1.05;
        font-weight: 800;
        margin: 0 0 0.6rem 0;
        letter-spacing: -0.03em;
    }
    .hero-copy {
        opacity: 0.88;
        max-width: 880px;
        font-size: 1.05rem;
        margin-bottom: 0;
    }
    .kpi-card {
        padding: 1rem 1.05rem;
        height: 100%;
    }
    .kpi-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.72;
        font-weight: 700;
    }
    .kpi-value {
        font-size: 1.45rem;
        font-weight: 800;
        margin-top: 0.2rem;
    }
    .kpi-caption {
        font-size: 0.88rem;
        opacity: 0.78;
        margin-top: 0.3rem;
    }
    .panel-card {
        padding: 1.15rem 1.2rem;
        margin-bottom: 1rem;
    }
    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .section-copy {
        font-size: 0.94rem;
        opacity: 0.82;
        margin-bottom: 0;
    }
    .metric-strip {
        padding: 0.8rem 0.95rem;
        border-radius: 14px;
        background: rgba(20, 122, 102, 0.08);
        border: 1px solid rgba(20, 122, 102, 0.18);
        margin-bottom: 0.9rem;
    }
    .metric-strip strong {
        display: block;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.18rem;
    }
    .chip-row span {
        display: inline-block;
        margin: 0.18rem 0.28rem 0 0;
        padding: 0.33rem 0.72rem;
        border-radius: 999px;
        border: 1px solid rgba(127, 127, 127, 0.24);
        background: rgba(127, 127, 127, 0.08);
        font-size: 0.78rem;
        font-weight: 600;
        color: var(--text-color);
    }
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        font-weight: 700;
        min-height: 2.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Model Snapshot")
    st.info(
        "Model: XGBoost Regressor\n\n"
        "Validation R²: 0.85\n\n"
        "Dataset: 1.01M daily sales records\n\n"
        "Target user: Store managers and regional planners"
    )
    st.caption("This demo shows how feature engineering and business rules are applied at inference time.")


st.markdown("<div class='page-kicker'>Production ML Product Demo</div>", unsafe_allow_html=True)
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">Rossmann Retail Demand Forecasting</div>
        <p class="hero-copy">
            This product demo turns an XGBoost forecasting model into a manager-facing planning tool.
            It combines calendar effects, promotions, store metadata, and competitive pressure to support
            staffing, replenishment, and campaign decisions at store level.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

kpi_cols = st.columns(3, gap="medium")
with kpi_cols[0]:
    render_kpi_card("Model", MODEL_NAME, "Tree-based demand forecasting for store-day predictions")
with kpi_cols[1]:
    render_kpi_card("Validation Score", f"R² {MODEL_R2:.2f}", "Strong fit on historical Rossmann validation data")
with kpi_cols[2]:
    render_kpi_card("Dataset Size", DATASET_SIZE, "Daily observations across stores, promos, and seasonality")

st.write("")

st.markdown(
    """
    <div class="panel-card">
        <div class="section-title">Why this matters for retail operations</div>
        <p class="section-copy">
            Accurate store-level demand helps managers decide how much inventory to position,
            when to schedule labor, and whether a promotion is likely to create meaningful uplift.
            The goal is not just prediction accuracy, but better day-to-day planning decisions.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

input_col, output_col = st.columns([0.95, 1.25], gap="large")

with input_col:
    st.markdown(
        """
        <div class="panel-card">
            <div class="section-title">Forecast Configuration</div>
            <p class="section-copy">Provide a small set of business inputs. Calendar features are derived automatically from the selected date.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=False):
        forecast_date = st.date_input("Forecast Date", value=date(2015, 7, 31))
        store_id = st.number_input("Store ID", min_value=1, max_value=1115, value=85, step=1)

        base_a, base_b = st.columns(2)
        promo = base_a.toggle("Promo Active", value=True)
        school_holiday = base_b.toggle("School Holiday", value=False)

        meta_a, meta_b = st.columns(2)
        store_type = meta_a.selectbox("Store Type", ["a", "b", "c", "d"], index=0)
        assortment = meta_b.selectbox("Assortment", ["a", "b", "c"], index=0)

        ops_a, ops_b = st.columns(2)
        state_holiday_label = ops_a.selectbox(
            "State Holiday",
            ["Regular Day", "Public Holiday", "Easter Holiday", "Christmas"],
            index=0,
        )
        competition_distance = ops_b.number_input(
            "Competition Distance (m)",
            min_value=0.0,
            max_value=100000.0,
            value=1270.0,
            step=100.0,
        )

        promo2 = st.toggle("Promo2 Program Active", value=False)
        promo_col1, promo_col2 = st.columns(2)
        promo2_since_week = promo_col1.number_input("Promo2 Since Week", min_value=0, max_value=53, value=0, step=1, disabled=not promo2)
        promo2_since_year = promo_col2.number_input("Promo2 Since Year", min_value=0, max_value=2030, value=0, step=1, disabled=not promo2)

        st.markdown("##### Operational Context")
        context_col1, context_col2 = st.columns(2)
        lag_1_sales = context_col1.number_input("Yesterday Sales", min_value=0.0, value=6200.0, step=100.0)
        lag_7_sales = context_col2.number_input("Last Week Same-Day Sales", min_value=0.0, value=5900.0, step=100.0)
        rolling_7_mean_sales = context_col1.number_input("7-Day Mean Sales", min_value=0.0, value=6050.0, step=100.0)
        customers = context_col2.number_input("Expected Customers", min_value=0, value=540, step=10)
        competition_age = st.slider("Competition Age (months)", min_value=0, max_value=120, value=18)

        state_holiday_map = {
            "Regular Day": "0",
            "Public Holiday": "a",
            "Easter Holiday": "b",
            "Christmas": "c",
        }

        payload = build_payload(
            store_id=store_id,
            forecast_dt=forecast_date,
            promo=promo,
            state_holiday=state_holiday_map[state_holiday_label],
            school_holiday=school_holiday,
            store_type=store_type,
            assortment=assortment,
            competition_distance=competition_distance,
            promo2=promo2,
            promo2_since_week=promo2_since_week,
            promo2_since_year=promo2_since_year,
            competition_age=competition_age,
            lag_1_sales=lag_1_sales,
            lag_7_sales=lag_7_sales,
            rolling_7_mean_sales=rolling_7_mean_sales,
            customers=customers,
        )

        scenario_payload = dict(payload)
        scenario_payload["promo"] = 0 if payload["promo"] == 1 else 1

        run_forecast = st.button("Generate Forecast", type="primary")

with output_col:
    st.markdown(
        """
        <div class="panel-card">
            <div class="section-title">Decision Output</div>
            <p class="section-copy">Use the forecast to compare expected demand with the network average and understand the likely impact of promotions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    result = None
    scenario_result = None
    error = None
    scenario_error = None

    if run_forecast:
        with st.spinner("Scoring the store-day scenario..."):
            result, error = call_api("/predict/rossmann", payload=payload)
            if not error:
                scenario_result, scenario_error = call_api("/predict/rossmann", payload=scenario_payload)

    if error:
        st.error(error)
    elif result:
        business_rule = result.get("business_rule")
        if business_rule:
            st.warning(business_rule)

        predicted_sales = float(result.get("predicted_sales", 0.0))
        actual_sales = result.get("actual_sales")
        benchmark = result.get("benchmark", {})
        delta, scenario_pct = scenario_delta(result, scenario_result)

        top_metrics = st.columns(3, gap="medium")
        top_metrics[0].metric("Predicted Sales", f"EUR {predicted_sales:,.0f}")
        top_metrics[1].metric(
            "Vs Network Average",
            f"{benchmark.get('pct_vs_average', 0.0):.1f}%",
            delta=f"EUR {benchmark.get('delta_vs_average', 0.0):,.0f}",
        )
        top_metrics[2].metric(
            "Promo Scenario Impact",
            f"{scenario_pct:.1f}%",
            delta=f"EUR {delta:,.0f}",
        )

        if actual_sales is not None:
            compare_cols = st.columns(2, gap="medium")
            compare_cols[0].metric("Demo Actual Sales", f"EUR {float(actual_sales):,.0f}")
            compare_cols[1].metric("Forecast Accuracy View", f"EUR {predicted_sales - float(actual_sales):,.0f}")

        history = history_frame(result.get("history", []))
        if not history.empty:
            st.markdown("##### Actual vs Predicted Trend")
            st.line_chart(history[["predicted_sales", "actual_sales"]], use_container_width=True)
            st.caption("The chart shows a short decision window around the forecast date so managers can compare expected demand with recent realised performance.")

        st.markdown(
            f"""
            <div class="metric-strip">
                <strong>Manager Interpretation</strong>
                {interpret_sales(predicted_sales, benchmark, float(actual_sales) if actual_sales is not None else None)}
            </div>
            """,
            unsafe_allow_html=True,
        )

        derived = result.get("derived_features", {})
        if derived:
            st.markdown("##### Derived Features Used at Inference")
            st.markdown(
                f"""
                <div class="chip-row">
                    <span>DayOfWeek: {derived.get('day_of_week')}</span>
                    <span>Month: {derived.get('month')}</span>
                    <span>Year: {derived.get('year')}</span>
                    <span>WeekOfYear: {derived.get('week_of_year')}</span>
                    <span>Promo Month: {derived.get('is_promo_month')}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        feature_df = feature_importance_frame(result.get("feature_importance", []))
        if not feature_df.empty:
            st.markdown("##### Feature Importance Snapshot")
            st.bar_chart(feature_df.set_index("feature"), use_container_width=True)

        if scenario_error:
            st.caption(f"Scenario simulation unavailable: {scenario_error}")
    else:
        st.info("Run a forecast to see predicted sales, store performance context, feature importance, and scenario simulation.")


st.write("")
st.markdown(
    """
    <div class="panel-card">
        <div class="section-title">Production-thinking improvements included in this demo</div>
        <div class="chip-row">
            <span>Derived calendar features from date input</span>
            <span>Business-rule closure logic</span>
            <span>Operational context inputs</span>
            <span>Scenario simulation</span>
            <span>Feature importance visibility</span>
            <span>Manager-facing interpretation</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
