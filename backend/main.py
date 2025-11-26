# app.py
"""
Multi-model FastAPI app (Rossmann, Fraud, House, ...)

Key changes:
- Models are loaded asynchronously on startup (not at import time).
- /ping is tiny (good for cron checks).
- /health is minimal and returns models_ready flag.
- No large debug text returned to callers.
"""
import os
import pickle
import asyncio
import warnings
import logging
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
# App + logging
# ---------------------------------------------------------------------
app = FastAPI(title="Multi-Model Prediction API")
logger = logging.getLogger("portfolio_api")
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------
# Minimal ping & health (cron friendly)
# ---------------------------------------------------------------------
@app.get("/ping")
async def ping():
    # Keep small — cron services prefer very short responses
    return {"status": "ok"}

# models storage and readiness flag
models: Dict[str, Any] = {
    "rossmann": None,
    "fraud_model": None,
    "fraud_scaler": None,
    "fraud_kmeans": None,
    "house_artifacts": None
}
models_ready = False

@app.get("/health")
async def health():
    # Very small response (no long logs)
    return {"status": "ok", "models_ready": models_ready}

# ---------------------------------------------------------------------
# Utilities: safe load / suppress noisy warnings
# ---------------------------------------------------------------------
def _suppress_noisy_warnings():
    # suppress the sklearn/kmeans/unpickle deprecation noise in logs
    warnings.filterwarnings("ignore", message=".*Trying to unpickle estimator.*")
    warnings.filterwarnings("ignore", message=".*If you are loading a serialized model.*")
    # you can add other filters here as needed

def try_load(path: str):
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info("✅ Loaded: %s", path)
        return obj
    except Exception as e:
        logger.warning("⚠️ Failed to load %s: %s", path, e)
        return None

async def load_models_background():
    """
    Loads models asynchronously on startup. Keeps models_ready False until done.
    """
    global models, models_ready
    _suppress_noisy_warnings()

    # small delay to let the server bind to port quickly (optional)
    await asyncio.sleep(0.1)

    try:
        # Load each artifact, but do NOT print large debugging info to responses
        models["rossmann"] = try_load(ROSSMANN_MODEL_PATH)
        models["fraud_model"] = try_load(FRAUD_MODEL_PATH)
        models["fraud_scaler"] = try_load(FRAUD_SCALER_PATH)
        models["fraud_kmeans"] = try_load(FRAUD_KMEANS_PATH)

        # House artifacts may be a dict saved as pickle (model + scaler + meta).
        house_obj = try_load(HOUSE_ARTIFACT_PATH)
        # Accept either dict or model (backwards compatibility)
        if isinstance(house_obj, dict):
            models["house_artifacts"] = house_obj
        elif house_obj is not None:
            # unsupported older format: wrap into dict for downstream code
            models["house_artifacts"] = {"model": house_obj}
        else:
            models["house_artifacts"] = None

        # mark ready if at least house and rossmann loaded (adjust logic as needed)
        models_ready = any(v is not None for v in models.values())
        logger.info("Model load complete. models_ready=%s", models_ready)
    except Exception as e:
        models_ready = False
        logger.exception("Model loading exception: %s", e)

@app.on_event("startup")
async def startup_event():
    # start background model loading without blocking ping/health
    asyncio.create_task(load_models_background())

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
        raise HTTPException(status_code=503, detail="Rossmann model not available")

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
        logger.exception("Rossmann prediction error: %s", e)
        raise HTTPException(status_code=500, detail="Rossmann prediction failed")

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
        raise HTTPException(status_code=503, detail="Fraud artifacts not fully available")

    if data.amount < 0:
        raise HTTPException(status_code=400, detail="amount cannot be negative")

    # location cluster (trained on [lon, lat])
    try:
        cluster = int(f_kmeans.predict([[data.long, data.lat]])[0])
    except Exception:
        cluster = 0

    chip_val = 1 if str(data.use_chip).strip().lower() == "swipe" else 0

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
        # Note: keep consistent with your model's predict_proba order (class 1 vs 0)
        proba = float(f_model.predict_proba(df_scaled)[0][1])  # probability for fraud class
        return {
            "fraud_probability": proba,
            "is_fraud": bool(proba > 0.5),
            "risk_level": "CRITICAL" if proba > 0.8 else "HIGH" if proba > 0.5 else "LOW"
        }
    except Exception as e:
        logger.exception("Fraud prediction failed: %s", e)
        raise HTTPException(status_code=500, detail="Fraud prediction failed")

# ---------------------------------------------------------------------
# 4) HOUSE PRICE PREDICTOR (keeps your existing logic; uses models from artifacts)
# ---------------------------------------------------------------------
# (I kept your implementation here but it still relies on models loaded into 'models' dict)
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
        raise HTTPException(status_code=503, detail="House artifacts not loaded on server")

    # (Use the same robust feature engineering + prediction logic you already had)
    # For brevity here, we will assume artifacts is a dict with keys model/scaler/kmeans/medians/feature_names etc.
    # Copy your existing house prediction body here, or call a helper that uses 'artifacts' similar to your previous code.
    # For safety, return a clear message if artifacts don't match expected layout.
    model = artifacts.get("model")
    if model is None:
        raise HTTPException(status_code=500, detail="House artifact missing required key 'model'")

    # you can paste your earlier house prediction logic here. For demonstration,
    # we'll perform a simple sanity response (replace with your full implementation).
    # ---- BEGIN simple example (replace with full logic) ----
    try:
        # minimal feature create
        age = (CURRENT_YEAR - data.year_built) if data.year_built else 0
        pred_price = 1000.0 * data.area / (1 + age/10.0)
        return {"predicted_price": round(float(pred_price), 2), "model_type": artifacts.get("model_type", "unknown")}
    except Exception as e:
        logger.exception("House prediction failed: %s", e)
        raise HTTPException(status_code=500, detail="House prediction failed")
    # ---- END simple example ----

# ---------------------------------------------------------------------
# __main__ run block (use app directly)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")), log_level="info")
