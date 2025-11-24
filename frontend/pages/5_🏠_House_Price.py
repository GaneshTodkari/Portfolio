import streamlit as st
import requests
import pandas as pd
import os

# --- Page Config ---
st.set_page_config(page_title="House Price Prediction", page_icon="🏡", layout="wide")

# --- Custom CSS for "Friendly" Look ---
st.markdown("""
    <style>
    .big-font { font-size:24px !important; font-weight: bold; color: #2E86C1; }
    .stButton>button { width: 100%; border-radius: 10px; height: 3em; background-color: #2E86C1; color: white; font-weight: bold;}
    .metric-card { background-color: #f0f2f6; padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #e0e0e0;}
    .metric-label { font-size: 0.9em; color: #555; }
    .metric-value { font-size: 1.2em; font-weight: bold; color: #333; }
    </style>
    """, unsafe_allow_html=True)

# --- Header ---
st.title("🏡 House Price Prediction")
st.markdown("## Case Study")

# --- The Challenge ---
st.markdown("### 🎯 The Challenge")
st.markdown("""
Real estate data is often messy and heavily skewed. The challenge was to build a machine learning model capable of predicting property prices based on diverse features like area, room count, and physical location.

Standard linear models often fail to capture the non-linear relationships between these complex variables, requiring a more robust approach to pricing.
""")


# --- The Solution ---
st.markdown("### 🛠️ My Solution")
st.markdown("""
I engineered a full data pipeline focusing on **advanced feature engineering** to drive model performance:

* **Smart Preprocessing:** Implemented **percentile-based capping** to handle extreme outliers in price and area without losing valuable data density.
* **Spatial Feature Engineering:** Instead of using raw coordinates, I applied **K-Means Clustering** to group properties into 10 distinct "location clusters," capturing neighborhood value trends effectively. I also calculated a dynamic `distance_from_center` metric.
* **Rigorous Feature Selection:** Utilized **RFECV (Recursive Feature Elimination with Cross-Validation)** to narrow down 23 raw features to the top 9 predictive inputs, reducing noise and overfitting.
* **Model Optimization:** Benchmarked Linear Regression vs. Random Forest vs. XGBoost. I tuned the XGBoost model using `RandomizedSearchCV`, optimizing hyperparameters like `learning_rate` and `max_depth`.
""")

# --- The Results ---
st.markdown("### 🚀 The Results")
st.markdown("""
* **High Accuracy:** The final **XGBoost model** achieved an $R^2$ of **0.78**, significantly outperforming the baseline Linear Regression ($R^2$ 0.65).
* **Key Drivers:** Feature importance analysis revealed that `area`, `total_rooms`, and the engineered `location_cluster` were the strongest predictors of price.
* **Deployment Ready:** The final model and artifacts (scaler, K-means object) were serialized into a pipeline for instant real-time inference.
""")
st.divider()

# --- Layout: Two Columns ---
left_col, right_col = st.columns([1, 1.5], gap="large")

with left_col:
    st.subheader("📋 Property Details")
    
    # Property Type
    prop_type = st.selectbox("Property Type", ["Apartment", "House", "Other"], index=0)
    
    # Critical Feature: Year Built (Used for Age)
    year_built = st.number_input("Year Built", min_value=1950, max_value=2025, value=2015, step=1, help="Used to calculate property age.")

    c1, c2 = st.columns(2)
    area = c1.number_input("Area (m²)", min_value=20.0, value=100.0, step=5.0)
    bedrooms = c2.number_input("Bedrooms", min_value=0, max_value=10, value=3)
    
    c3, c4 = st.columns(2)
    bathrooms = c3.number_input("Bathrooms", min_value=1, max_value=10, value=2)
    parking = c4.number_input("Parking Spots", min_value=0, max_value=10, value=1)
    
    rooms = st.slider("Extra Attached Rooms", 0, 5, 0, help="Maid's room, storage, or office spaces.")

with right_col:
    st.subheader("📍 Location Intelligence")
    st.info("Drag the pin or enter coordinates to see how location affects value.")
    
    # Defaults (Natal, Brazil based on your data)
    default_lat, default_lon = -5.83, -35.20
    
    lc1, lc2 = st.columns(2)
    lat = lc1.number_input("Latitude", value=default_lat, format="%.6f")
    lon = lc2.number_input("Longitude", value=default_lon, format="%.6f")

    # Map Visualization
    map_data = pd.DataFrame({'lat': [lat], 'lon': [lon]})
    st.map(map_data, zoom=12)

# --- Action Section ---
st.divider()
api_url = os.getenv("HOUSE_API_URL", "http://127.0.0.1:8000/predict/house")

if st.button("💰 Calculate Market Value"):
    # Construct Payload matching HouseInput in main.py
    payload = {
        "area": area,
        "year_built": int(year_built),
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "parking_spots": parking,
        "attached_rooms": rooms,
        "type": prop_type,
        "lat": lat,
        "lon": lon
    }

    with st.spinner("Analyzing market trends, calculating distance, and clustering neighborhood..."):
        try:
            response = requests.post(api_url, json=payload, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                price = result.get("predicted_price", 0)
                debug = result.get("debug_info", {}) # Get the engineered features
                
                # --- Success Display ---
                st.balloons()
                
                # 1. Main Price Card
                st.markdown(f"""
                <div style="background-color: #d4edda; padding: 20px; border-radius: 10px; border: 1px solid #c3e6cb; text-align: center; margin-bottom: 20px;">
                    <h2 style="color: #155724; margin:0;">Estimated Market Value</h2>
                    <h1 style="color: #155724; font-size: 50px; margin:10px 0;"> {price:,.2f}</h1>
                    <p style="color: #155724; margin:0;">Based on XGBoost prediction</p>
                </div>
                """, unsafe_allow_html=True)

                
        except requests.exceptions.ConnectionError:
            st.error("❌ Connection Failed. Is the FastAPI backend running?")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

# --- Sidebar info ---
with st.sidebar:
    st.header("About the Model")
    st.info("""
    **Model Engine:** XGBoost Regressor
    
    **Trained On:** 4,000+ Real Estate Listings
    
    **Key Predictors:**
    - Property Area & Age
    - Room Counts
    - Distance from City Center
    - Neighborhood Cluster (KMeans)
    """)
    st.caption("Developed by Ganesh Todkari")