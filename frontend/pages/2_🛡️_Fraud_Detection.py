import streamlit as st
import requests

st.set_page_config(page_title="Credit Card Security Analysis", page_icon="🛡️")

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

if st.button("🔍 Analyze Transaction"):
    api_url = "http://127.0.0.1:8000/predict/fraud" # Change to Render URL later
    
    payload = {
        "amount": amount,
        "lat": lat,
        "long": long,
        "use_chip": method,
        "merchant_city": "Unknown"
    }
    
    with st.spinner("Scanning for anomalies..."):
        try:
            res = requests.post(api_url, json=payload).json()
            
            prob = res['fraud_probability']
            
            if res['is_fraud']:
                st.error(f"🚨 FRAUD DETECTED! Risk Score: {prob:.2%}")
                st.write(f"Reason: High-value transaction in unusual cluster.")
            else:
                st.success(f"✅ Transaction Safe. Risk Score: {prob:.2%}")
                
        except:
            st.error("Connection Error. Is the backend running?")

# --- Sidebar ---
with st.sidebar:
    st.header("Model Specs")
    st.info("**AUC-ROC:** 0.99\n\n**Technique:** SMOTE + XGBoost\n\n**Feature:** K-Means Location Clustering")
    st.caption("Developed by Ganesh Todkari")