import requests
import streamlit as st
from requests.exceptions import RequestException, Timeout
from streamlit.errors import StreamlitSecretNotFoundError

st.set_page_config(page_title="Retail Demand Forecasting", page_icon="📈", layout="wide")


def get_backend_url() -> str:
    default_url = "https://portfolio-i8re.onrender.com"
    try:
        return st.secrets.get("BACKEND_URL", default_url)
    except StreamlitSecretNotFoundError:
        return default_url


BACKEND_URL = get_backend_url()


def call_api(path: str, method: str = "get", payload=None, timeout: int = 15):
    url = BACKEND_URL.rstrip("/") + path
    try:
        if method.lower() == "get":
            response = requests.get(url, timeout=timeout)
        elif method.lower() == "post":
            response = requests.post(url, json=payload, timeout=timeout)
        else:
            return None, "Invalid HTTP method."
        response.raise_for_status()
        return response.json(), None
    except Timeout:
        return None, "Request timed out."
    except RequestException as exc:
        return None, f"Network error: {exc}"
    except Exception as exc:
        return None, f"Unexpected error: {exc}"


st.markdown(
    """
    <style>
    [data-testid="stMainBlockContainer"] { max-width: 1140px; padding-top: 1.1rem; }
    .page-kicker { font-size: 0.84rem; text-transform: uppercase; letter-spacing: 0.08em; color: #1f77b4; font-weight: 700; }
    .panel {
        background: var(--secondary-background-color);
        border: 1px solid rgba(49, 51, 63, 0.16);
        border-radius: 12px;
        box-shadow: 0 6px 16px rgba(15, 23, 42, 0.08);
        padding: 1rem 1.1rem;
        margin-bottom: 0.8rem;
    }
    .panel p { margin-bottom: 0.2rem; opacity: 0.86; }
    .stButton > button { width: 100%; border-radius: 8px; font-weight: 600; }
    .badge-row span {
        display: inline-block;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 0.2rem 0.2rem 0 0;
        padding: 0.25rem 0.65rem;
        border-radius: 999px;
        color: #1f77b4;
        border: 1px solid rgba(31, 119, 180, 0.3);
        background: rgba(31, 119, 180, 0.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='page-kicker'>Case Study</div>", unsafe_allow_html=True)
st.title("Retail Demand Forecasting")
st.markdown(
    """
    <div class="panel">
      <h4 style="margin:0 0 0.35rem 0;">Business Context</h4>
      <p>Rossmann needed reliable six-week, store-level sales forecasts to improve inventory planning and reduce stock inefficiencies.</p>
    </div>
    <div class="panel">
      <h4 style="margin:0 0 0.35rem 0;">Approach</h4>
      <p>Built an end-to-end forecasting workflow with seasonal feature engineering, business-rule variables, robust missing value treatment, and model benchmarking.</p>
    </div>
    <div class="panel">
      <h4 style="margin:0 0 0.35rem 0;">Outcome</h4>
      <p>Final model achieved <strong>R² = 0.85</strong> and reduced forecast error by approximately <strong>15%</strong>.</p>
      <div class="badge-row">
        <span>Python</span><span>XGBoost</span><span>Time Series</span><span>Feature Engineering</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

inputs_col, insight_col = st.columns([1, 1.2], gap="large")

with inputs_col:
    st.subheader("Forecast Inputs")
    store_id = st.number_input("Store ID", min_value=1, max_value=1115, value=1)
    c1, c2 = st.columns(2)
    promo = c1.radio("Promotion Active", ["No", "Yes"])
    day = c2.selectbox(
        "Day of Week",
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        index=4,
    )
    dist = st.slider("Competition Distance (m)", min_value=0, max_value=75000, value=500)
    holiday = st.selectbox(
        "State Holiday",
        ["None (Regular Day)", "Public Holiday (a)", "Easter (b)", "Christmas (c)"],
    )

    day_map = {"Monday": 1, "Tuesday": 2, "Wednesday": 3, "Thursday": 4, "Friday": 5, "Saturday": 6, "Sunday": 7}
    promo_map = {"No": 0, "Yes": 1}
    holiday_map = {"None (Regular Day)": "0", "Public Holiday (a)": "a", "Easter (b)": "b", "Christmas (c)": "c"}

with insight_col:
    st.subheader("Interpretation Guide")
    st.info("Use this simulation to estimate daily sales and compare with the portfolio baseline.")
    st.markdown(
        """
        - Captures weekly seasonality and promotional uplift  
        - Includes holiday and competition effects  
        - Useful for inventory and staffing decisions
        """
    )

if st.button("Run Forecast"):
    payload = {
        "store_id": int(store_id),
        "day_of_week": day_map[day],
        "promo": promo_map[promo],
        "competition_distance": float(dist),
        "state_holiday": holiday_map[holiday],
    }

    with st.spinner("Running model inference..."):
        result, error = call_api("/predict/rossmann", method="post", payload=payload)

    if error:
        st.error(f"Backend connection failed: {error}")
    else:
        sales = float(result.get("predicted_sales", 0))
        business_rule = result.get("business_rule")
        if business_rule:
            st.warning(f"Store closed condition triggered: {business_rule}")
        else:
            st.success("Forecast generated.")
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Predicted Daily Sales", f"EUR {sales:,.2f}")
            with m2:
                benchmark = 5773.0
                delta = sales - benchmark
                pct = ((sales / benchmark) - 1) * 100 if benchmark else 0
                st.metric("Vs Baseline Average", f"{pct:.1f}%", delta=f"{delta:,.0f} EUR")

with st.sidebar:
    st.header("Model Snapshot")
    st.info("Algorithm: XGBoost Regressor\n\nR²: 0.85\n\nTraining Size: 1.1M rows")
    st.caption("Developed by Ganesh Todkari")
