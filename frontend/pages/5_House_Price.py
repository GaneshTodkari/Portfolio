import pandas as pd
import requests
import streamlit as st
from requests.exceptions import RequestException, Timeout
from streamlit.errors import StreamlitSecretNotFoundError

st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="wide")


def get_backend_url() -> str:
    default_url = "https://portfolio-i8re.onrender.com"
    try:
        return st.secrets.get("BACKEND_URL", default_url)
    except StreamlitSecretNotFoundError:
        return default_url


BACKEND_URL = get_backend_url()


def call_api(path: str, method: str = "post", payload=None, timeout: int = 15):
    url = BACKEND_URL.rstrip("/") + path
    try:
        if method.lower() == "post":
            response = requests.post(url, json=payload, timeout=timeout)
        else:
            response = requests.get(url, timeout=timeout)
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
    .value-card {
        background: var(--secondary-background-color);
        border: 1px solid rgba(49, 51, 63, 0.22);
        border-left: 4px solid #1f77b4;
        border-radius: 12px;
        text-align: center;
        padding: 1rem 1.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='page-kicker'>Case Study</div>", unsafe_allow_html=True)
st.title("House Price Prediction Engine")
st.markdown(
    """
    <div class="panel">
      <h4 style="margin:0 0 0.35rem 0;">Business Context</h4>
      <p>Property valuation requires balancing structural attributes with location effects and nonlinear market behavior.</p>
    </div>
    <div class="panel">
      <h4 style="margin:0 0 0.35rem 0;">Approach</h4>
      <p>Used spatial clustering, outlier capping, and feature selection before training an XGBoost regressor for robust pricing estimates.</p>
    </div>
    <div class="panel">
      <h4 style="margin:0 0 0.35rem 0;">Outcome</h4>
      <p>Delivered a deployment-ready pricing model with <strong>R² = 0.78</strong>.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

input_col, map_col = st.columns([1, 1.2], gap="large")

with input_col:
    st.subheader("Property Inputs")
    prop_type = st.selectbox("Property Type", ["Apartment", "House", "Other"], index=0)
    year_built = st.number_input("Year Built", min_value=1950, max_value=2026, value=2015)
    area = st.number_input("Area (sq. meters)", min_value=20.0, value=100.0, step=5.0)
    bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=3)
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
    parking = st.number_input("Parking Spots", min_value=0, max_value=10, value=1)
    rooms = st.slider("Additional Rooms", min_value=0, max_value=5, value=0)

with map_col:
    st.subheader("Location Inputs")
    st.info("Adjust latitude and longitude to reflect the property location.")
    default_lat, default_lon = -5.83, -35.20
    lat = st.number_input("Latitude", value=default_lat, format="%.6f")
    lon = st.number_input("Longitude", value=default_lon, format="%.6f")
    st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}), zoom=12)

if st.button("Estimate Property Value"):
    payload = {
        "area": float(area),
        "year_built": int(year_built),
        "bedrooms": int(bedrooms),
        "bathrooms": int(bathrooms),
        "parking_spots": int(parking),
        "attached_rooms": int(rooms),
        "type": prop_type.lower(),
        "lat": float(lat),
        "lon": float(lon),
    }

    with st.spinner("Running valuation model..."):
        result, error = call_api("/predict/house", payload=payload)

    if error:
        st.error(f"Connection error: {error}")
    else:
        price = result.get("predicted_price")
        if price is None:
            st.error("Model response did not include predicted_price.")
        else:
            value = float(price)
            st.success("Valuation generated.")
            st.markdown(
                f"""
                <div class="value-card">
                    <div style="opacity:0.75;">Estimated Market Value</div>
                    <div style="font-size:2.2rem; font-weight:700; margin:0.25rem 0;">{value:,.2f}</div>
                    <div style="opacity:0.75;">XGBoost model prediction</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

with st.sidebar:
    st.header("Model Snapshot")
    st.info(
        "Model: XGBoost Regressor\n\nTraining Data: 4,000+ records\n\n"
        "Key Drivers: Area, Property Age, Spatial Cluster, City Distance"
    )
    st.caption("Developed by Ganesh Todkari")
