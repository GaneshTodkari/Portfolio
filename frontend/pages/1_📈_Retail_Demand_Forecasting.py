import streamlit as st
import requests
import pandas as pd
from requests.exceptions import RequestException, Timeout

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(page_title="Retail Demand Forecasting", page_icon="📈", layout="wide")

# ---------------------------------------------------
# BACKEND CONNECTION HELPER
# ---------------------------------------------------
BACKEND_URL = st.secrets.get("BACKEND_URL", "https://portfolio-i8re.onrender.com")

def call_api(path: str, method="get", json=None, timeout=15):
    """
    Safe wrapper for backend communication.
    Returns (result_json, error_message)
    """
    url = BACKEND_URL.rstrip("/") + path
    try:
        if method == "get":
            r = requests.get(url, timeout=timeout)
        elif method == "post":
            r = requests.post(url, json=json, timeout=timeout)
        else:
            return None, "Invalid method"

        r.raise_for_status()
        return r.json(), None

    except Timeout:
        return None, "⏳ Request timed out."

    except RequestException as e:
        return None, f"🔌 Network error: {e}"

    except Exception as e:
        return None, f"⚠️ Unexpected error: {e}"


# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------
st.markdown("""
    <style>
    .big-font { font-size:24px !important; font-weight: bold; color: #2E86C1; }
    .stButton>button { width: 100%; border-radius: 10px; height: 3em; background-color: #2E86C1; color: white; font-weight: bold;}
    .metric-card { background-color: #f0f2f6; padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #e0e0e0;}
    .metric-value { font-size: 1.5em; font-weight: bold; color: #333; }
    .metric-label { font-size: 0.9em; color: #555; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.title("📈 Retail Demand Forecasting")
st.markdown("## Case Study")

# The Challenge
st.markdown("### 🎯 The Challenge")
st.markdown("""
Rossmann operated over **3,000 drug stores** across Europe and needed a forecasting system
to predict daily sales for 6 weeks ahead.

The dataset included **1.1M rows**, missing competitor info, promo interactions, and strong seasonal effects.
""")

st.markdown("---")

# The Solution
st.markdown("### 🛠️ My Solution")
st.markdown("""
I built a full end-to-end ML pipeline with:
- **Log1p Target Normalization**
- **Cyclical sine–cosine features**
- **Business logic features (CompetitionAge, PromoMonths)**
- **99th percentile replacement for missing CompetitionDistance**
- **Model comparison → XGBoost won (R² = 0.85)**
""")

st.markdown("---")

# Results
st.markdown("### 🚀 The Results")
st.markdown("""
- **High Accuracy:** RMSE 0.16 (log scale)
- **R² = 0.85**
- **Production-ready pipeline capable of forecasting 40k+ rows**
""")

st.divider()

# ---------------------------------------------------
# UI LAYOUT
# ---------------------------------------------------
left_col, right_col = st.columns([1, 1.5], gap="large")

with left_col:
    st.subheader("🏪 Store Parameters")

    store_id = st.number_input("Store ID", min_value=1, max_value=1115, value=1)

    c1, c2 = st.columns(2)
    promo = c1.radio("Active Promotion?", ["No", "Yes"])
    day = c2.selectbox("Day of Week", 
                       ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                       index=4)

    dist = st.slider("Competition Distance (m)", 0, 75000, 500)
    holiday = st.selectbox("State Holiday", 
                           ["None (Regular Day)", "Public Holiday (a)", "Easter (b)", "Christmas (c)"])

    # Mappings
    day_map = {"Monday":1, "Tuesday":2, "Wednesday":3, "Thursday":4, "Friday":5, "Saturday":6, "Sunday":7}
    promo_map = {"No": 0, "Yes": 1}
    holiday_map = {"None (Regular Day)": "0", "Public Holiday (a)": "a", "Easter (b)": "b", "Christmas (c)": "c"}

with right_col:
    st.subheader("📊 Forecast Insights")
    st.info("Adjust parameters on the left to see real-time sales projections.")

    st.markdown("""
    **Model Capabilities:**
    - 📅 Seasonality patterns  
    - 🏷️ Promo impact  
    - 🎄 Holiday logic  
    """)

st.divider()

# ---------------------------------------------------
# ACTION BUTTON
# ---------------------------------------------------
if st.button("💰 Generate Sales Forecast"):
    payload = {
        "store_id": int(store_id),
        "day_of_week": day_map[day],
        "promo": promo_map[promo],
        "competition_distance": float(dist),
        "state_holiday": holiday_map[holiday]
    }

    with st.spinner("Running XGBoost inference..."):
        result, err = call_api("/predict/rossmann", method="post", json=payload)

        if err:
            st.error(f"Backend Connection Failed: {err}")
        else:
            # Success
            sales = result.get("predicted_sales", 0)
            business_rule = result.get("business_rule")

            if business_rule:
                st.error(f"⛔ **Store Closed:** {business_rule}")
            else:
                st.success("Prediction successful!")
                st.balloons()

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Predicted Daily Sales", f"€{sales:,.2f}")

                with col2:
                    benchmark = 5773
                    delta = sales - benchmark
                    pct = ((sales / benchmark) - 1) * 100
                    st.metric("Vs Dataset Average", f"{pct:.1f}%", delta=f"{delta:,.0f}€")

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
with st.sidebar:
    st.header("Model Specs")
    st.info("**R² Score:** 0.85\n\n**Algorithm:** XGBoost Regressor\n\n**Training Data:** 1.1M rows")
    st.caption("Developed by Ganesh Todkari")
