import streamlit as st
import requests
import pandas as pd
import os

# --- Page Config ---
st.set_page_config(page_title="Retail Demand ForecastingI", page_icon="📈", layout="wide")

# --- Custom CSS ---
st.markdown("""
    <style>
    .big-font { font-size:24px !important; font-weight: bold; color: #2E86C1; }
    .stButton>button { width: 100%; border-radius: 10px; height: 3em; background-color: #2E86C1; color: white; font-weight: bold;}
    .metric-card { background-color: #f0f2f6; padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #e0e0e0;}
    .metric-value { font-size: 1.5em; font-weight: bold; color: #333; }
    .metric-label { font-size: 0.9em; color: #555; }
    </style>
    """, unsafe_allow_html=True)

# --- Header ---
st.title("📈 Retail Demand Forecasting")
st.markdown("## Case Study")

# --- The Challenge ---
st.markdown("### 🎯 The Challenge")
st.markdown("""
Rossmann operated over **3,000 drug stores** in 7 European countries. The management task was to predict daily sales for up to six weeks in advance. 

The data was challenging: it contained **over 1 million records** with heavy right-skewed sales distributions, missing competitor data, and complex seasonal dependencies (holidays, school closures, and promo cycles).
""")

st.markdown("---")

# --- The Solution ---
st.markdown("### 🛠️ My Solution")
st.markdown("""
I built a full data science pipeline focusing on **Business Logic feature engineering**:

* **Statistical Normalization:** EDA revealed a heavy right-skew in sales data. I applied a **Log1p Transformation (`np.log1p`)** to the target variable, converting the skewed distribution into a normal distribution for better model stability.
* **Cyclical Time Features:** Instead of simple one-hot encoding, I used **Trigonometric transformations (Sine/Cosine)** on Month and DayOfWeek. This allowed the model to understand that December (12) is mathematically close to January (1).
* **Business Logic:** Created specific features like `CompetitionAge` (calculating months since a competitor opened) and `IsPromoMonth` to capture the impact of marketing campaigns.
* **Smart Imputation:** Handled missing `CompetitionDistance` values by imputing them with the **99th percentile** (outliers) rather than the mean, preserving the signal that these stores had no nearby competitors.
* **Model Selection:** I compared three distinct approaches:
    * *Linear Regression:* Failed to capture non-linearities ($R^2$ 0.24).
    * *Decision Tree:* Improved performance ($R^2$ 0.83).
    * **XGBoost Regressor:** Proved best (**$R^2$ 0.85**) with the lowest RMSE.
""")

st.markdown("---")

# --- The Results ---
st.markdown("### 🚀 The Results")
st.markdown("""
* **High Precision:** The final model achieved an RMSE of 0.16 (on log scale) and an $R^2$ of **0.85**.
* **Visual Validation:** Residual plots showed a tight, symmetrical clustering of *Predicted vs. Actual* sales, confirming the model was unbiased across both low and high sales volume stores.
* **Scalable Output:** The final pipeline generated a submission file capable of predicting sales for **40,000+ future instances** across all stores.
""")

st.divider()

# --- Layout ---
left_col, right_col = st.columns([1, 1.5], gap="large")

with left_col:
    st.subheader("🏪 Store Parameters")
    
    store_id = st.number_input("Store ID", min_value=1, max_value=1115, value=1)
    
    c1, c2 = st.columns(2)
    promo = c1.radio("Active Promotion?", ["No", "Yes"])
    day = c2.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], index=4)
    
    dist = st.slider("Competition Distance (m)", 0, 75000, 500)
    holiday = st.selectbox("State Holiday", ["None (Regular Day)", "Public Holiday (a)", "Easter (b)", "Christmas (c)"])

    # Mappings
    day_map = {"Monday":1, "Tuesday":2, "Wednesday":3, "Thursday":4, "Friday":5, "Saturday":6, "Sunday":7}
    promo_map = {"No": 0, "Yes": 1}
    holiday_map = {"None (Regular Day)": "0", "Public Holiday (a)": "a", "Easter (b)": "b", "Christmas (c)": "c"}

with right_col:
    st.subheader("📊 Forecast Insights")
    st.info("Adjust parameters on the left to see real-time sales projections.")
    
    # Placeholder for chart or static info
    st.markdown("""
    **Model Capabilities:**
    - 📅 **Seasonality:** Detects weekly and yearly patterns.
    - 🏷️ **Promo Impact:** Measures the lift from active promotions.
    - 🛑 **Holiday Logic:** Automatically handles store closures.
    """)

# --- Action ---
st.divider()
api_url = os.getenv("ROSSMANN_API_URL", "http://127.0.0.1:8000/predict/rossmann")

if st.button("💰 Generate Sales Forecast"):
    payload = {
        "store_id": store_id,
        "day_of_week": day_map[day],
        "promo": promo_map[promo],
        "competition_distance": dist,
        "state_holiday": holiday_map[holiday]
    }
    
    with st.spinner("Running XGBoost inference..."):
        try:
            response = requests.post(api_url, json=payload)
            if response.status_code == 200:
                data = response.json()
                sales = data["predicted_sales"]
                business_rule = data.get("business_rule", None)
                
                # --- RESULT LOGIC ---
                if business_rule:
                    st.error(f"⛔ **Store Closed:** {business_rule}. No revenue expected.")
                else:
                    # --- METRICS SECTION ---
                    st.balloons()
                    
                    metric_col1, metric_col2 = st.columns(2)
                    
                    with metric_col1:
                        st.metric(label="Predicted Daily Sales", value=f"€{sales:,.2f}")
                    
                    with metric_col2:
                        if sales > 0:
                            benchmark = 5773 # Dataset Average
                            delta = sales - benchmark
                            
                            # Ensure proper float formatting so Streamlit detects sign automatically
                            # Using delta_color="normal" makes Positive=Green, Negative=Red
                            st.metric(
                                label="Vs. Dataset Average", 
                                value=f"{((sales/benchmark)-1)*100:.1f}%", 
                                delta=f"{delta:,.0f} €",
                                delta_color="normal" 
                            )
                        else:
                            st.metric(label="Vs. Dataset Average", value="0%", delta="0 €", delta_color="off")

                    # --- SUCCESS CARD ---
                    st.markdown(f"""
                    <div style="background-color: #d4edda; padding: 20px; border-radius: 10px; border: 1px solid #c3e6cb; text-align: center; margin-top: 20px;">
                        <h2 style="color: #155724; margin:0;">Predicted Daily Revenue</h2>
                        <h1 style="color: #155724; font-size: 50px; margin:10px 0;">€{sales:,.2f}</h1>
                        <p style="color: #155724; margin:0;">XGBoost Model Confidence: High</p>
                    </div>
                    """, unsafe_allow_html=True)

            else:
                st.error(f"Error: {response.text}")
        except:
            st.error("🚨 Backend Connection Failed")

# --- Sidebar ---
with st.sidebar:
    st.header("Model Specs")
    st.info("**R² Score:** 0.85\n\n**Algorithm:** XGBoost Regressor\n\n**Training Data:** 1.1M Transactions")
    st.caption("Developed by Ganesh Todkari")