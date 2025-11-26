# app.py
"""
Multi-model FastAPI app (Rossmann, Fraud, Chat, House price)
- Loads artifacts from models/ directory
- Robust /predict/house endpoint that DOES NOT add city_lat/city_lon as features
  (only computes distance_from_center internally and passes only model features)
Run:
    uvicorn app:app --reload --host 0.0.0.0 --port 8000
"""
import os
import pickle
from datetime import datetime
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
MODEL_DIR = os.getenv("MODEL_DIR", "models")
HOUSE_ARTIFACT_NAME = os.getenv("HOUSE_ARTIFACT_NAME", "house_price_model.pkl")
HOUSE_ARTIFACT_PATH = os.path.join(MODEL_DIR, HOUSE_ARTIFACT_NAME)

ROSSMANN_MODEL_PATH = os.path.join(MODEL_DIR, "rossmann_model.pkl")
FRAUD_MODEL_PATH = os.path.join(MODEL_DIR, "fraud_model.pkl")
FRAUD_SCALER_PATH = os.path.join(MODEL_DIR, "fraud_scaler.pkl")
FRAUD_KMEANS_PATH = os.path.join(MODEL_DIR, "fraud_kmeans.pkl")

CURRENT_YEAR = 2025

# ---------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------
app = FastAPI(title="Multi-Model Prediction API")

@app.get("/ping")
async def ping():
    # KEEP THIS VERY SMALL — cron services prefer short responses
    return {"status": "ok"}
# ---------------------------------------------------------------------
# Global storage for loaded artifacts
# ---------------------------------------------------------------------
models: Dict[str, Any] = {
    "rossmann": None,
    "fraud_model": None,
    "fraud_scaler": None,
    "fraud_kmeans": None,
    "house_artifacts": None
}


# ---------------------------------------------------------------------
# Helper: load pickle safely
# ---------------------------------------------------------------------
def try_load(path: str):
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"✅ Loaded: {path}")
        return obj
    except Exception as e:
        print(f"⚠️ Failed to load {path}: {e}")
        return None


# ---------------------------------------------------------------------
# Startup: load models/artifacts
# ---------------------------------------------------------------------
models["rossmann"] = try_load(ROSSMANN_MODEL_PATH)
models["fraud_model"] = try_load(FRAUD_MODEL_PATH)
models["fraud_scaler"] = try_load(FRAUD_SCALER_PATH)
models["fraud_kmeans"] = try_load(FRAUD_KMEANS_PATH)
models["house_artifacts"] = try_load(HOUSE_ARTIFACT_PATH)


# ---------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------
@app.get("/health")
def health():
    loaded = {k: (v is not None) for k, v in models.items()}
    return {"status": "ok", "loaded": loaded}


# ---------------------------------------------------------------------
# 1) ROSSMANN PREDICTOR
# ---------------------------------------------------------------------
class RossmannInput(BaseModel):
    store_id: int
    day_of_week: int      # 1=Mon .. 7=Sun
    promo: int            # 0 or 1
    competition_distance: float
    state_holiday: str    # "0", "a", "b", "c"


@app.post("/predict/rossmann")
def predict_sales(data: RossmannInput):
    model = models.get("rossmann")
    if model is None:
        raise HTTPException(status_code=500, detail="Rossmann model not loaded")

    # Business rules
    if data.day_of_week == 7 or data.state_holiday != "0":
        return {
            "predicted_sales": 0.0,
            "business_rule": "Store closed (Sunday or holiday)"
        }

    if data.competition_distance < 0:
        raise HTTPException(status_code=400, detail="competition_distance cannot be negative")

    # Feature engineering (mirror training pipeline as close as possible)
    now = datetime.now()
    holiday_map = {'0': 0, 'a': 1, 'b': 2, 'c': 3}
    input_data = {
        'Store': data.store_id,
        'DayOfWeek': data.day_of_week,
        'Promo': data.promo,
        'StateHoliday': holiday_map.get(data.state_holiday, 0),
        'SchoolHoliday': 0,
        'StoreType': 0,
        'Assortment': 0,
        'CompetitionDistance': data.competition_distance,
        'Promo2': 0,
        'Promo2SinceWeek': 0,
        'Promo2SinceYear': 0,
        'Month': now.month,
        'Year': now.year,
        'WeekOfYear': now.isocalendar()[1],
        'Month_sin': np.sin(2 * np.pi * now.month / 12),
        'Month_cos': np.cos(2 * np.pi * now.month / 12),
        'DayOfWeek_sin': np.sin(2 * np.pi * data.day_of_week / 7),
        'DayOfWeek_cos': np.cos(2 * np.pi * data.day_of_week / 7),
        'CompetitionAge': 0,
        'IsPromoMonth': 0
    }

    cols = ['Store', 'DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday',
            'StoreType', 'Assortment', 'CompetitionDistance', 'Promo2',
            'Promo2SinceWeek', 'Promo2SinceYear', 'Month', 'Year', 'WeekOfYear',
            'Month_sin', 'Month_cos', 'DayOfWeek_sin', 'DayOfWeek_cos',
            'CompetitionAge', 'IsPromoMonth']

    df = pd.DataFrame([input_data])[cols]

    try:
        raw_pred = model.predict(df)
        pred_val = float(raw_pred[0]) if hasattr(raw_pred, "__len__") else float(raw_pred)
        predicted_sales = float(np.expm1(pred_val))
        return {"predicted_sales": round(predicted_sales, 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rossmann prediction failed: {e}")


# ---------------------------------------------------------------------
# 2) FRAUD DETECTION
# ---------------------------------------------------------------------
class FraudInput(BaseModel):
    amount: float
    lat: float
    long: float
    use_chip: str  # "Swipe" or "Online"


@app.post("/predict/fraud")
def predict_fraud(data: FraudInput):
    f_model = models.get("fraud_model")
    f_scaler = models.get("fraud_scaler")
    f_kmeans = models.get("fraud_kmeans")

    if not all([f_model, f_scaler, f_kmeans]):
        raise HTTPException(status_code=500, detail="Fraud model artifacts not fully loaded")

    if data.amount < 0:
        raise HTTPException(status_code=400, detail="amount cannot be negative")

    # location cluster (trained on [lon, lat])
    try:
        cluster = int(f_kmeans.predict([[data.long, data.lat]])[0])
    except Exception:
        cluster = 0

    chip_val = 1 if str(data.use_chip).strip().lower() == "swipe" else 0

    # Build input (keep same ordering as scaler expects)
    input_data = {
        'current_age': 55,
        'retirement_age': 66,
        'credit_score': 685,
        'num_credit_cards': 3,
        'num_cards_issued': 66,
        'year_pin_last_changed': 2018,
        'amount': data.amount,
        'use_chip': chip_val,
        'gender': 1,
        'card_brand': 2,
        'card_type': 1,
        'has_chip': 1,
        'years_with_bank': 10,
        'location_cluster': cluster
    }

    cols = ['current_age', 'retirement_age', 'credit_score', 'num_credit_cards',
            'num_cards_issued', 'year_pin_last_changed', 'amount', 'use_chip',
            'gender', 'card_brand', 'card_type', 'has_chip', 'years_with_bank',
            'location_cluster']

    df = pd.DataFrame([input_data])[cols]

    try:
        df_scaled = f_scaler.transform(df)
        proba = f_model.predict_proba(df_scaled)[0][0]  # probability of class 0 = fraud
        return {
            "fraud_probability": float(proba),
            "is_fraud": bool(proba > 0.5),
            "risk_level": "CRITICAL" if proba > 0.8 else "HIGH" if proba > 0.5 else "LOW"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fraud prediction failed: {e}")


# # ---------------------------------------------------------------------
# # 3) BASIC CHATBOT (RULE-BASED)
# # ---------------------------------------------------------------------
# # --- Knowledge Base ---
# resume_data = {
#     "skills": "I specialize in Python, SQL, Power BI, and Machine Learning (XGBoost, Scikit-learn). I also build full-stack apps with Flutter and Firebase.",
#     "education": "I am currently pursuing an MBA-IT at SICSR (Symbiosis), Pune (Exp. 2026). I hold a BSc in Computer Science (2023).",
#     "rossmann": "My Rossmann Sales project used XGBoost to forecast daily store sales. I achieved an R² of 0.85 by engineering cyclical time features and filling missing data.",
#     "fraud": "For the Fraud Detection project, I handled a highly imbalanced dataset (151 frauds vs 10k legit). I used SMOTE and K-Means clustering to achieve 99% AUC-ROC.",
#     "vybe": "VybeRiders is a bike rental admin panel I built using Flutter. It reduced manual billing time from 10 mins to 15 seconds using automated logic.",
#     "house": "I built a House Price Estimator using XGBoost that predicts property values based on location clustering and amenities.",
#     "contact": "You can reach me via email at ganeshtodkari@example.com or connect on LinkedIn.",
#     "default": "I am Ganesh's virtual assistant. I can tell you about his **Skills**, **Education**, **Rossmann Project**, **Fraud Project**, or **Contact Info**."
# }

# class ChatInput(BaseModel):
#     message: str

# @app.post("/chat")
# def chat_bot(data: ChatInput):
#     msg = data.message.lower()
#     response = resume_data["default"]
    
#     # Simple Keyword Matching
#     if any(x in msg for x in ["skill", "tech", "stack", "python", "sql", "power bi", "xgboost"]):
#         response = resume_data["skills"]
#     elif any(x in msg for x in ["edu", "degree", "college", "mba", "study", "education"]):
#         response = resume_data["education"]
#     elif "rossmann" in msg or "sales" in msg:
#         response = resume_data["rossmann"]
#     elif "fraud" in msg:
#         response = resume_data["fraud"]
#     elif "vybe" in msg or "bike" in msg or "app" in msg:
#         response = resume_data["vybe"]
#     elif "house" in msg or "price" in msg or "real estate" in msg:
#         response = resume_data["house"]
#     elif "contact" in msg or "email" in msg or "hire" in msg:
#         response = resume_data["contact"]
#     elif "Explain the Rossmann Sales Model" in msg or "email" in msg or "hire" in msg:
#         response = '''I developed a forecasting model to predict daily sales for thousands of Rossmann stores to solve inventory management issues (under/over-stocking).

# The Data: I processed over 1 million rows, handling missing data by imputing competition distance with the 99th percentile and using log-transforms to fix skewed sales data.

# The Strategy: I focused heavily on feature engineering, creating cyclical sine-cosine features for time seasonality and flags for promo months.

# The Result: After comparing Linear Regression and Decision Trees, my final XGBoost Regressor achieved an R² of 0.85 with the lowest RMSE, providing a production-ready solution for accurate stocking.
#     return {"response": response}'''


# ---------------------------------------------------------------------
# 4) HOUSE PRICE PREDICTOR (clean, full implementation)
# ---------------------------------------------------------------------
class HouseInput(BaseModel):
    area: float = Field(..., gt=0)
    year_built: Optional[int] = None
    bathrooms: int = Field(..., ge=0)
    bedrooms: int = Field(..., ge=0)
    parking_spots: int = Field(..., ge=0)
    attached_rooms: int = Field(..., ge=0)
    type: str = Field(..., description="apartment | house | other")
    lat: Optional[float] = None
    lon: Optional[float] = None
    include_extras: Optional[bool] = False


@app.post("/predict/house")
def predict_house(data: HouseInput):
    artifacts = models.get("house_artifacts")
    if artifacts is None:
        raise HTTPException(status_code=500, detail="House artifacts not loaded on server")

    # Extract saved objects if present
    model = artifacts.get("model")
    scaler = artifacts.get("scaler")               # optional
    preprocessor = artifacts.get("preprocessor")   # optional
    saved_feature_names = artifacts.get("feature_names") or artifacts.get("columns") or artifacts.get("feature_list")
    medians = artifacts.get("medians", {})

    if model is None:
        raise HTTPException(status_code=500, detail="House artifact missing required key 'model'")

    notes = {}

    # --- Known final feature set (the 9 features your model expects) ---
    final_9 = [
        "area",
        "parking_spots",
        "attached_rooms",
        "age",
        "type",
        "total_rooms",
        "area_per_room",
        "distance_from_center",
        "location_cluster",
    ]

    # If saved_feature_names exist but length doesn't match model expected shape, override with final_9
    model_expected_n = None
    try:
        # sklearn-like models often have n_features_in_
        model_expected_n = int(getattr(model, "n_features_in_", None)) if getattr(model, "n_features_in_", None) is not None else None
    except Exception:
        model_expected_n = None

    # Decide which feature_names to use
    if saved_feature_names:
        # ensure saved_feature_names is a list
        if not isinstance(saved_feature_names, (list, tuple)):
            saved_feature_names = list(saved_feature_names)
        saved_len = len(saved_feature_names)
    else:
        saved_len = 0

    # If model tells a number and saved names length mismatches -> override
    if model_expected_n is not None and saved_len != model_expected_n:
        notes["feature_names_warning"] = (
            f"artifact.feature_names length ({saved_len}) != model.n_features_in_ ({model_expected_n}). "
            "Overriding with final 9-feature safe list."
        )
        feature_names = final_9
    else:
        # If no model shape info, but saved_feature_names exists and length matches final_9, keep it
        if saved_feature_names and (len(saved_feature_names) == len(final_9)):
            feature_names = list(saved_feature_names)
        else:
            # fallback to final_9 to be safe (this ensures only the 9 expected features are passed)
            feature_names = final_9
            if saved_feature_names and saved_len != len(final_9):
                notes["feature_names_info"] = (
                    f"Using fallback final_9 features because artifact.feature_names length ({saved_len}) "
                    f"doesn't match expected 9."
                )

    # --- Feature engineering (internal only) ---
    # Age
    if data.year_built:
        age = CURRENT_YEAR - int(data.year_built)
    else:
        age = int(medians.get("age", medians.get("year_built", 0) or 0))

    total_rooms = int(data.bedrooms + data.bathrooms + data.attached_rooms)
    area_per_room = float(data.area) / (total_rooms + 1)

    # Compute distance_from_center internally only (do NOT add center lat/lon to df_input)
    input_lat = data.lat if data.lat is not None else None
    input_lon = data.lon if data.lon is not None else None
    center_lat = medians.get("center_lat", medians.get("lat"))
    center_lon = medians.get("center_lon", medians.get("lon"))

    if input_lat is None or input_lon is None:
        distance_from_center = float(medians.get("distance_from_center", 0.0))
        notes["distance_warning"] = "Lat/Lon not provided; used fallback distance_from_center from medians."
    else:
        if center_lat is not None and center_lon is not None:
            try:
                distance_from_center = float(((input_lat - center_lat) ** 2 + (input_lon - center_lon) ** 2) ** 0.5)
            except Exception:
                distance_from_center = float(medians.get("distance_from_center", 0.0))
                notes["distance_warning"] = "Distance computation failed; used fallback distance."
        else:
            distance_from_center = 0.0
            notes["distance_warning"] = "No training centroid saved; distance_from_center set to 0."

    # location_cluster using saved kmeans if available
    kmeans = artifacts.get("kmeans") or artifacts.get("km") or artifacts.get("kmeans_model")
    if kmeans is not None and (input_lat is not None and input_lon is not None):
        try:
            location_cluster = int(kmeans.predict([[input_lon, input_lat]])[0])
        except Exception:
            location_cluster = int(medians.get("location_cluster", 0))
            notes["cluster_warning"] = "kmeans.predict failed; used median cluster."
    else:
        location_cluster = int(medians.get("location_cluster", 0))
        if kmeans is None:
            notes["cluster_warning"] = "kmeans not found in artifacts; used fallback cluster."

    # type encoding: use saved label encoder if exists, else fallback mapping
    type_enc = None
    le_type = artifacts.get("label_encoder_type") or artifacts.get("type_encoder")
    if le_type is not None:
        try:
            type_enc = int(le_type.transform([data.type.lower()])[0])
        except Exception:
            type_enc = None

    if type_enc is None:
        fallback_type_map = {"apartment": 0, "house": 1, "other": 2}
        type_enc = fallback_type_map.get(str(data.type).strip().lower(), int(medians.get("type", 2)))
        notes["type_note"] = "Used fallback type mapping; save LabelEncoder in artifacts to avoid this."

    # Build the candidate dict for ONLY the final features (no lat/lon entries)
    candidate = {
        "area": float(data.area),
        "parking_spots": int(data.parking_spots),
        "attached_rooms": int(data.attached_rooms),
        "age": int(age),
        "type": int(type_enc),
        "total_rooms": int(total_rooms),
        "area_per_room": float(area_per_room),
        "distance_from_center": float(distance_from_center),
        "location_cluster": int(location_cluster),
    }

    # Create df_input and only keep the columns that are in feature_names
    df_input = pd.DataFrame([candidate])

    # If artifact's feature_names include extra columns, we fill only those names but we will ensure final shape matches model
    missing = {}
    for col in feature_names:
        if col not in df_input.columns:
            df_input[col] = medians.get(col, 0.0)
            missing[col] = medians.get(col, 0.0)
    if missing:
        notes["missing_fallbacks"] = missing

    # Reorder to exact model expected order (feature_names), but make sure no extra columns like 'lat','lon' exist
    try:
        # ensure we only pass columns in feature_names and in df_input
        cols_to_pass = [c for c in feature_names if c in df_input.columns]
        df_input = df_input[cols_to_pass]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature ordering mismatch: {e}")

    # Final safety: confirm df_input has the same number of columns as the model expects (if available)
    if model_expected_n is not None:
        provided_n = df_input.shape[1]
        if provided_n != model_expected_n:
            # If mismatch still persists, try the authoritative fallback: final_9
            notes["shape_mismatch_before_predict"] = f"model expects {model_expected_n} features but df has {provided_n}."
            # Force df_input to use final_9 (intersection with df_input or fill from medians)
            forced = {}
            for col in final_9:
                if col not in df_input.columns:
                    df_input[col] = medians.get(col, 0.0)
                    forced[col] = medians.get(col, 0.0)
            df_input = df_input[[c for c in final_9]]
            notes["shape_mismatch_forced_fill"] = {"forced_fill": forced, "final_columns_used": final_9}
            # final check
            if df_input.shape[1] != model_expected_n:
                raise HTTPException(status_code=500, detail=f"After forcing final_9 columns, still mismatch: model expects {model_expected_n}, df has {df_input.shape[1]}")

    # Apply preprocessor/scaler if available; prefer preprocessor
    X_pred = None
    if preprocessor is not None:
        try:
            X_pred = preprocessor.transform(df_input)
        except Exception as e:
            notes["preprocessor_warning"] = f"preprocessor.transform failed: {e}; will attempt scaler or raw features"

    if X_pred is None:
        if scaler is not None:
            try:
                X_pred = scaler.transform(df_input.values)
            except Exception as e:
                notes["scaler_warning"] = f"scaler.transform failed: {e}; using raw features"
                X_pred = df_input.values
        else:
            X_pred = df_input.values

    # Predict
    try:
        raw_pred = model.predict(X_pred)
        pred_val = float(raw_pred[0]) if hasattr(raw_pred, "__len__") else float(raw_pred)
        predicted_price = max(0.0, pred_val)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    response = {
        "predicted_price": round(predicted_price, 2),
        "model_type": artifacts.get("model_type", "unknown"),
        "input_features": df_input.iloc[0].to_dict(),
        "notes": notes
    }

    if data.include_extras:
        extras = {}
        if hasattr(model, "feature_importances_"):
            extras["feature_importances"] = dict(zip(df_input.columns.tolist(), [float(x) for x in model.feature_importances_]))
        response["extras"] = extras

    return response

# ---------------------------------------------------------------------
# Run with: uvicorn app:app --reload
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    # run the FastAPI instance from this file as module "main"
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)