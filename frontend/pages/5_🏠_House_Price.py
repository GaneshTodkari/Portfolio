import streamlit as st
import requests
import pandas as pd
from requests.exceptions import RequestException, Timeout
import os

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(page_title="House Price Prediction", page_icon="🏡", layout="wide")

# -------------------------------------------------------
# BACKEND URL (Auto-switch between local & deployed)
# -------------------------------------------------------
BACKEND_URL = st.secrets.get("BACKEND_URL", "https://portfolio-i8re.onrender.com")

def call_api(path, method="post", json=None, timeout=15):
    """Safe wrapper for backend requests."""
    url = BACKEND_URL.rstrip("/") + path
    try:
        if method.lower() == "post":
            r = requests.post(url, json=json, timeout=timeout)
        else:
            r = requests.get(url, timeout=timeout)

        r.raise_for_status()
        return r.json(), None

    except Timeout:
        return None, "⏳ Request timed out."
    except RequestException as e:
        return None, f"🔌 Network error: {e}"
    except Exception as e:
        return None, f"⚠ Unexpected error: {e}"

# -------------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------------
st.markdown("""
<style>
.stButton>button {
    width: 100%;
    border-radius: 10px;
    height: 3em;
    background-color: #2E86C1;
    color: white;
    font-weight: bold;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    border: 1px solid #e0e0e0;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# PAGE HEADER
# -------------------------------------------------------
st.title("🏡 House Price Prediction")
st.markdown("## Case Study")

st.markdown("### 🎯 The Challenge")
st.markdown("""
Real estate pricing requires models that understand **location**, **features**, and **non-linear relationships**.
""")

st.markdown("### 🛠️ My Solution")
st.markdown("""
✔ Spatial clustering using **K-Means**  
✔ Outlier handling using **percentile-based capping**  
✔ RFECV for **top feature selection**  
✔ Final model: **XGBoost Regressor**
""")

st.markdown("### 🚀 The Results")
st.markdown("""
* **R² = 0.78**
* **Improved feature robustness**
* **Deployment-ready model pipeline**
""")

st.divider()

# -------------------------------------------------------
# INPUT UI
# -------------------------------------------------------
left_col, right_col = st.columns([1, 1.5], gap="large")

with left_col:
    st.subheader("📋 Property Details")

    prop_type = st.selectbox("Property Type", ["Apartment", "House", "Other"], index=0)
    year_built = st.number_input("Year Built", min_value=1950, max_value=2025, value=2015)

    area = st.number_input("Area (m²)", min_value=20.0, value=100.0, step=5.0)
    bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=3)
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
    parking = st.number_input("Parking Spots", min_value=0, max_value=10, value=1)
    rooms = st.slider("Extra Attached Rooms", 0, 5, 0)

with right_col:
    st.subheader("📍 Location Intelligence")
    st.info("Enter coordinates or adjust them manually.")

    default_lat, default_lon = -5.83, -35.20
    lat = st.number_input("Latitude", value=default_lat, format="%.6f")
    lon = st.number_input("Longitude", value=default_lon, format="%.6f")

    st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}), zoom=12)

st.divider()

# -------------------------------------------------------
# ACTION BUTTON
# -------------------------------------------------------
if st.button("💰 Calculate Market Value"):

    payload = {
        "area": area,
        "year_built": int(year_built),
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "parking_spots": parking,
        "attached_rooms": rooms,
        "type": prop_type.lower(),
        "lat": float(lat),
        "lon": float(lon)
    }

    with st.spinner("Predicting price using XGBoost model..."):
        result, err = call_api("/predict/house", json=payload)

        if err:
            st.error(f"❌ Connection Error: {err}")
        else:
            price = result.get("predicted_price")

            st.balloons()
            st.markdown(f"""
            <div style="background-color:#d4edda;padding:20px;
                        border-radius:10px;border:1px solid #c3e6cb;text-align:center;">
                <h2 style="color:#155724;margin:0;">Estimated Market Value</h2>
                <h1 style="color:#155724;font-size:50px;margin:10px 0;">
                    {price:,.2f}
                </h1>
                <p style="color:#155724;margin:0;">XGBoost Model Prediction</p>
            </div>
            """, unsafe_allow_html=True)

# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------
with st.sidebar:
    st.header("About the Model")
    st.info("""
    **Model:** XGBoost Regressor  
    **Training Data:** 4,000+ Listings  
    **Key Predictors:**  
    • Area  
    • Age  
    • Clustering (K-Means)  
    • Distance From City Center  
    """)
    st.caption("Developed by Ganesh Todkari")
