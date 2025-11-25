import streamlit as st
import requests
from requests.exceptions import RequestException, Timeout

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(page_title="Credit Card Security Analysis", page_icon="🛡️")

# ---------------------------------------------------
# BACKEND CONNECTION HELPER
# ---------------------------------------------------
# Use Streamlit secrets (set BACKEND_URL in Streamlit Cloud -> Settings -> Secrets)
BACKEND_URL = st.secrets.get("BACKEND_URL", "https://portfolio-i8re.onrender.com")

def call_api(path: str, method="get", json=None, timeout=15):
    """
    Safe wrapper for backend communication.
    Returns (result_json, error_message)
    """
    url = BACKEND_URL.rstrip("/") + path
    try:
        if method.lower() == "get":
            r = requests.get(url, timeout=timeout)
        elif method.lower() == "post":
            r = requests.post(url, json=json, timeout=timeout)
        else:
            return None, "Invalid HTTP method"

        r.raise_for_status()
        return r.json(), None

    except Timeout:
        return None, "⏳ Request timed out."

    except RequestException as e:
        # network / HTTP / connection failures
        return None, f"🔌 Network error: {e}"

    except Exception as e:
        return None, f"⚠️ Unexpected error: {e}"


# ---------------------------------------------------
# PAGE CONTENT
# ---------------------------------------------------
st.title("🛡️ Credit Card Security Analysis")
st.markdown("## Case Study")

# --- The Challenge ---
st.markdown("### 🎯 The Challenge")
st.markdown("""
In financial security, the "Needle in a Haystack" problem is the norm. I worked with a dataset of **over 10,000 transactions** where only **1.5% (151 cases)** were fraudulent.
A standard model would achieve 98.5% accuracy just by guessing "Legitimate" every time, failing to catch a single fraud. The challenge was to build a model sensitive enough to catch the fraud without flagging valid customers (False Positives).
""")

# --- The Solution ---
st.markdown("### 🛠️ My Solution")
st.markdown("""
I managed the full lifecycle from ETL to Visualization, focusing on overcoming class imbalance:

* **Geospatial Feature Engineering:** I applied **K-Means Clustering** on merchant coordinates (Latitude/Longitude) to group 1,700+ merchants into regional clusters, allowing the model to spot location anomalies.
* **Customer History:** Derived `years_with_bank` from account opening dates to identify if tenure correlated with fraud risk.
* **Handling Imbalance (SMOTE):** I used **Synthetic Minority Over-sampling Technique (SMOTE)**. Instead of simply duplicating fraud records, SMOTE mathematically generates new, synthetic examples of fraud to train the model effectively.
* **Model Selection:** I benchmarked Logistic Regression (Baseline) against Random Forest and XGBoost. I optimized for **AUC-ROC and F1-Score** rather than simple accuracy, as these are the true measures of success in imbalance problems.
""")

# --- The Results ---
st.markdown("### 🚀 The Results")
st.markdown("""
* **Near-Perfect Detection:** The final **XGBoost Classifier** achieved an **AUC-ROC of 0.9985** and an **F1-Score of 0.99**, proving it could distinguish between fraud and genuine transactions with extreme precision.
* **Business Intelligence:** I didn't stop at Python. I exported the results to **Power BI**, building a dashboard to visualize fraud hotspots by State and Merchant Cluster, transforming the model output into an executive-level risk monitor.
""")

st.divider()
st.markdown("## Security Parameters")

col1, col2 = st.columns(2)

with col1:
    amount = st.number_input("Transaction Amount ($)", min_value=0.0, max_value=10000.0, value=50.0)
    method = st.selectbox("Method", ["Swipe", "Online"])

with col2:
    st.write("📍 **Simulate Location**")
    location = st.selectbox("Select Scenario", [
        "Home - Ohio",
        "Vacation - California",
        "International - Paris"
    ])

# Map Locations to Lat/Long
loc_map = {
    "Home - Ohio": (40.55, -81.0),       # Matches cluster 0 or 1
    "Vacation - California": (36.77, -119.4),
    "International - Paris": (48.85, 2.35) # Anomalous lat/long
}
lat, long = loc_map[location]

# Debug: (developer note) path to last uploaded screenshot in this session
# Use this local path as a reference URL if needed:
SCREENSHOT_PATH = "/mnt/data/7a611118-44f9-4d3d-b8c1-aaaa0f11c857.png"
# (Displayed only as text so you can copy it into docs or repo if required)
st.caption(f"Debug screenshot (local path): {SCREENSHOT_PATH}")

if st.button("🔍 Analyze Transaction"):
    # Build minimal payload to match FastAPI's expected schema
    payload = {
        "amount": float(amount),
        "lat": float(lat),
        "long": float(long),
        "use_chip": method  # backend expects "Swipe" or "Online"
    }

    with st.spinner("Scanning for anomalies..."):
        result, err = call_api("/predict/fraud", method="post", json=payload)

        if err:
            st.error(f"Connection Error. Is the backend running? {err}")
        else:
            # parse response
            prob = result.get("fraud_probability")
            is_fraud = result.get("is_fraud", False)

            if prob is None:
                st.error("Unexpected response from backend.")
            else:
                # display nicely
                pct = prob * 100
                if is_fraud:
                    st.error(f"🚨 FRAUD DETECTED! Risk Score: {pct:.2f}%")
                    # optionally show notes if backend sends any
                    if result.get("notes"):
                        st.json(result.get("notes"))
                else:
                    st.success(f"✅ Transaction Safe. Risk Score: {pct:.2f}%")
                    if result.get("notes"):
                        st.info("Backend notes:")
                        st.json(result.get("notes"))

# --- Sidebar ---
with st.sidebar:
    st.header("Model Specs")
    st.info("**AUC-ROC:** 0.99\n\n**Technique:** SMOTE + XGBoost\n\n**Feature:** K-Means Location Clustering")
    st.caption("Developed by Ganesh Todkari")
