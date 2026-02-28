import requests
import streamlit as st
from requests.exceptions import RequestException, Timeout
from streamlit.errors import StreamlitSecretNotFoundError

st.set_page_config(page_title="Credit Card Security Analysis", page_icon="🛡️", layout="wide")


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
st.title("Credit Card Security Analysis")
st.markdown(
    """
    <div class="panel">
      <h4 style="margin:0 0 0.35rem 0;">Business Context</h4>
      <p>Fraud detection in payment systems is an extreme imbalance problem where false negatives directly translate to financial loss.</p>
    </div>
    <div class="panel">
      <h4 style="margin:0 0 0.35rem 0;">Approach</h4>
      <p>Implemented feature engineering, geospatial clustering, SMOTE-based balancing, and model comparison optimized for AUC-ROC and F1.</p>
    </div>
    <div class="panel">
      <h4 style="margin:0 0 0.35rem 0;">Outcome</h4>
      <p>The final classifier delivered <strong>AUC-ROC ≈ 0.998</strong> with strong precision-recall behavior for fraud monitoring.</p>
      <div class="badge-row">
        <span>SMOTE</span><span>XGBoost</span><span>Risk Analytics</span><span>Geo Features</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()
st.subheader("Transaction Risk Simulation")

col1, col2 = st.columns(2, gap="large")
with col1:
    amount = st.number_input("Transaction Amount (USD)", min_value=0.0, max_value=10000.0, value=50.0)
    method = st.selectbox("Payment Method", ["Swipe", "Online"])
with col2:
    location = st.selectbox("Scenario", ["Home - Ohio", "Vacation - California", "International - Paris"])

loc_map = {
    "Home - Ohio": (40.55, -81.0),
    "Vacation - California": (36.77, -119.4),
    "International - Paris": (48.85, 2.35),
}
lat, lon = loc_map[location]

if st.button("Analyze Risk"):
    payload = {
        "amount": float(amount),
        "lat": float(lat),
        "long": float(lon),
        "use_chip": method,
    }

    with st.spinner("Scoring transaction risk..."):
        result, error = call_api("/predict/fraud", method="post", payload=payload)

    if error:
        st.error(f"Connection error: {error}")
    else:
        prob = result.get("fraud_probability")
        is_fraud = bool(result.get("is_fraud", False))
        if prob is None:
            st.error("Unexpected response from backend.")
        else:
            risk_score = float(prob) * 100
            if is_fraud:
                st.error(f"High-risk transaction detected. Risk score: {risk_score:.2f}%")
            else:
                st.success(f"Transaction appears safe. Risk score: {risk_score:.2f}%")
            st.progress(min(max(risk_score / 100.0, 0.0), 1.0))
            if result.get("notes"):
                st.info("Model notes")
                st.json(result["notes"])

with st.sidebar:
    st.header("Model Snapshot")
    st.info("Algorithm: XGBoost\n\nSampling: SMOTE\n\nAUC-ROC: 0.99")
    st.caption("Developed by Ganesh Todkari")
