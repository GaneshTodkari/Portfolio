# app.py
"""
Multi-model FastAPI app (Rossmann, Fraud, House, ...)

Key behaviors:
- Models are loaded asynchronously on startup (background task).
- /ping is tiny (good for cron checks).
- /health returns models_ready flag.
"""
import os
import pickle
import asyncio
import warnings
import logging
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, List
import joblib
import numpy as np
import pandas as pd
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Request
# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(BASE_DIR / "models"))).resolve()
HOUSE_ARTIFACT_NAME = os.getenv("HOUSE_ARTIFACT_NAME", "house_price_model.pkl")
HOUSE_ARTIFACT_PATH = MODEL_DIR / HOUSE_ARTIFACT_NAME

ROSSMANN_MODEL_PATH = MODEL_DIR / "rossmann_model.pkl"
FRAUD_MODEL_PATH = MODEL_DIR / "fraud_model.pkl"
FRAUD_SCALER_PATH = MODEL_DIR / "fraud_scaler.pkl"
FRAUD_KMEANS_PATH = MODEL_DIR / "fraud_kmeans.pkl"

CURRENT_YEAR = int(os.getenv("CURRENT_YEAR", "2025"))

# ---------------------------------------------------------------------
# App + logging
# ---------------------------------------------------------------------

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger("portfolio_api")
logger.setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(load_models_background())
    yield


app = FastAPI(
    title="Multi-Model Prediction API",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)

# ---------------------------------------------------------------------
# Minimal ping & health (cron friendly)
# ---------------------------------------------------------------------

@app.get("/ping")
async def ping():
    # Keep tiny and consistent (cron-friendly)
    return JSONResponse(
        content={"status": "ok"},
        media_type="application/json"
    )

@app.get("/health")
async def health():
    return JSONResponse(
        content={"status": "ok", "models_ready": models_ready},
        media_type="application/json"
    )

# -----------------------
# Middleware: suppress logging for ping/health
# -----------------------
@app.middleware("http")
async def suppress_ping_logging(request: Request, call_next):
    # Fast path for ping/health — avoid any heavy processing/logging
    if request.url.path in ("/ping", "/health"):
        response = await call_next(request)
        return response

    # For other endpoints, we'll log only warnings/errors by default.
    response = await call_next(request)
    return response

# models storage and readiness flag
models: Dict[str, Any] = {
    "rossmann": None,
    "fraud_model": None,
    "fraud_scaler": None,
    "fraud_kmeans": None,
    "house_artifacts": None
}
models_ready = False

def compute_models_ready(required_keys: Optional[List[str]] = None) -> bool:
    """
    By default returns True if any model loaded.
    You can pass required_keys to require specific artifacts be present.
    """
    if required_keys:
        return all(models.get(k) is not None for k in required_keys)
    return any(v is not None for v in models.values())

# ---------------------------------------------------------------------
# Utilities: safe load / suppress noisy warnings
# ---------------------------------------------------------------------
def _suppress_noisy_warnings():
    warnings.filterwarnings("ignore", message=".*Trying to unpickle estimator.*")
    warnings.filterwarnings("ignore", message=".*If you are loading a serialized model.*")

def try_load(path: str):
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info("Loaded: %s", path)
        return obj
    except Exception as e:
        logger.warning("Failed to load %s: %s", path, e)
        return None
    
def load_artifacts_joblib(path: str) -> Dict[str, Any]:
    """
    Preferred joblib loader that returns a dict of artifacts.
    Will raise FileNotFoundError if path not present.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    try:
        artifacts = joblib.load(path)
        if not isinstance(artifacts, dict):
            raise ValueError("Loaded object is not a dict of artifacts.")
        return artifacts
    except Exception:
        # fallback to pickle
        with open(path, "rb") as f:
            artifacts = pickle.load(f)
            if not isinstance(artifacts, dict):
                raise ValueError("Loaded object is not a dict of artifacts.")
            return artifacts


# -----------------------
# Background loader
# -----------------------
async def load_models_background():
    global models, models_ready
    # slight delay to let server bind quickly
    await asyncio.sleep(0.05)
    models["rossmann"] = try_load(ROSSMANN_MODEL_PATH)
    models["fraud_model"] = try_load(FRAUD_MODEL_PATH)
    models["fraud_scaler"] = try_load(FRAUD_SCALER_PATH)
    models["fraud_kmeans"] = try_load(FRAUD_KMEANS_PATH)
    house_obj = try_load(HOUSE_ARTIFACT_PATH)
    if isinstance(house_obj, dict):
        models["house_artifacts"] = house_obj
    elif house_obj is not None:
        models["house_artifacts"] = {"model": house_obj}
    else:
        models["house_artifacts"] = None

    models_ready = any(v is not None for v in models.values())
    logger.warning("Model load complete. models_ready=%s", models_ready)

# ---------------------------------------------------------------------
# 1) ROSSMANN PREDICTOR
# ---------------------------------------------------------------------
class RossmannInput(BaseModel):
    store_id: int
    forecast_date: str
    promo: int
    competition_distance: float
    state_holiday: str
    school_holiday: int = 0
    store_type: str = "a"
    assortment: str = "a"
    promo2: int = 0
    promo2_since_week: int = 0
    promo2_since_year: int = 0
    competition_age: int = 0
    lag_1_sales: float = 0.0
    lag_7_sales: float = 0.0
    rolling_7_mean_sales: float = 0.0
    customers: float = 0.0


ROSSMANN_COLS = [
    'Store', 'DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday',
    'StoreType', 'Assortment', 'CompetitionDistance', 'Promo2',
    'Promo2SinceWeek', 'Promo2SinceYear', 'Month', 'Year', 'WeekOfYear',
    'Month_sin', 'Month_cos', 'DayOfWeek_sin', 'DayOfWeek_cos',
    'CompetitionAge', 'IsPromoMonth'
]


def _encode_store_type(value: str) -> int:
    mapping = {"a": 0, "b": 1, "c": 2, "d": 3}
    return mapping.get(str(value).strip().lower(), 0)


def _encode_assortment(value: str) -> int:
    mapping = {"a": 0, "b": 1, "c": 2}
    return mapping.get(str(value).strip().lower(), 0)


def _parse_forecast_date(value: str) -> datetime:
    try:
        return datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="forecast_date must be a valid ISO date string (YYYY-MM-DD)"
        ) from exc


def _is_promo_month(forecast_dt: datetime, promo2: int, promo2_since_week: int, promo2_since_year: int) -> int:
    if not promo2 or not promo2_since_week or not promo2_since_year:
        return 0
    current_key = forecast_dt.isocalendar()[0] * 100 + forecast_dt.isocalendar()[1]
    promo_key = int(promo2_since_year) * 100 + int(promo2_since_week)
    return 1 if current_key >= promo_key else 0


def _build_rossmann_features(data: RossmannInput, forecast_dt: datetime) -> Dict[str, Any]:
    day_of_week = int(forecast_dt.isoweekday())
    week_of_year = int(forecast_dt.isocalendar()[1])
    month = int(forecast_dt.month)
    year = int(forecast_dt.year)
    holiday_map = {'0': 0, 'a': 1, 'b': 2, 'c': 3}

    return {
        'Store': data.store_id,
        'DayOfWeek': day_of_week,
        'Promo': int(data.promo),
        'StateHoliday': holiday_map.get(data.state_holiday, 0),
        'SchoolHoliday': int(data.school_holiday),
        'StoreType': _encode_store_type(data.store_type),
        'Assortment': _encode_assortment(data.assortment),
        'CompetitionDistance': float(data.competition_distance),
        'Promo2': int(data.promo2),
        'Promo2SinceWeek': int(data.promo2_since_week),
        'Promo2SinceYear': int(data.promo2_since_year),
        'Month': month,
        'Year': year,
        'WeekOfYear': week_of_year,
        'Month_sin': float(np.sin(2 * np.pi * month / 12)),
        'Month_cos': float(np.cos(2 * np.pi * month / 12)),
        'DayOfWeek_sin': float(np.sin(2 * np.pi * day_of_week / 7)),
        'DayOfWeek_cos': float(np.cos(2 * np.pi * day_of_week / 7)),
        'CompetitionAge': int(max(data.competition_age, 0)),
        'IsPromoMonth': _is_promo_month(forecast_dt, data.promo2, data.promo2_since_week, data.promo2_since_year),
    }


def is_store_closed(day_of_week: int, state_holiday: str) -> tuple[bool, Optional[str]]:
    # Sunday is treated as a majority-case closure signal for the demo.
    if day_of_week == 7:
        return True, "Sunday closure assumption (majority case)"

    # Easter and Christmas are treated as hard-closure holidays.
    if state_holiday in ["b", "c"]:
        return True, "Store closed due to major holiday"

    # Public holidays vary by store, so allow the model to score them.
    if state_holiday == "a":
        return False, None

    return False, None


def _estimate_actual_sales(predicted_sales: float, data: RossmannInput) -> float:
    if predicted_sales <= 0:
        return 0.0

    uplift = 1.0
    uplift += 0.05 if data.promo else -0.02
    uplift += 0.04 if data.school_holiday else 0.0
    uplift += 0.03 if data.store_type.lower() in {"b", "c"} else 0.0
    uplift += min(max(data.lag_7_sales - predicted_sales, -2000), 2000) / 50000 if data.lag_7_sales else 0.0
    return round(max(predicted_sales * uplift, 0.0), 2)


def _build_demo_history(predicted_sales: float, actual_sales: float, forecast_dt: datetime) -> List[Dict[str, Any]]:
    history: List[Dict[str, Any]] = []
    for offset in range(6, -1, -1):
        day = forecast_dt - timedelta(days=offset)
        trend = 1 + ((6 - offset) - 3) * 0.015
        history.append({
            "date": day.strftime("%Y-%m-%d"),
            "predicted_sales": round(max(predicted_sales * trend, 0.0), 2),
            "actual_sales": round(max(actual_sales * (trend + ((offset % 3) - 1) * 0.02), 0.0), 2),
        })
    return history


def _extract_feature_importance(model: Any) -> List[Dict[str, Any]]:
    values = getattr(model, "feature_importances_", None)
    if values is None:
        booster = getattr(model, "get_booster", None)
        if callable(booster):
            try:
                score_map = booster().get_score(importance_type="gain")
                if score_map:
                    pairs = sorted(score_map.items(), key=lambda item: item[1], reverse=True)[:8]
                    return [{"feature": key, "importance": round(float(val), 4)} for key, val in pairs]
            except Exception:
                return []
        return []

    pairs = list(zip(ROSSMANN_COLS, values))
    pairs.sort(key=lambda item: item[1], reverse=True)
    return [{"feature": feature, "importance": round(float(val), 4)} for feature, val in pairs[:8]]

@app.post("/predict/rossmann")
def predict_sales(data: RossmannInput):
    model = models.get("rossmann")
    if model is None:
        raise HTTPException(status_code=503, detail="Rossmann model not available")

    if data.competition_distance < 0:
        raise HTTPException(status_code=400, detail="competition_distance cannot be negative")

    forecast_dt = _parse_forecast_date(data.forecast_date)
    day_of_week = int(forecast_dt.isoweekday())
    is_closed, closure_reason = is_store_closed(day_of_week, data.state_holiday)

    if is_closed:
        return {
            "predicted_sales": 0.0,
            "actual_sales": 0.0,
            "history": [],
            "business_rule": closure_reason,
            "benchmark": {
                "average_sales": 5773.0,
                "delta_vs_average": -5773.0,
                "pct_vs_average": -100.0,
            },
            "feature_importance": _extract_feature_importance(model),
        }

    input_data = _build_rossmann_features(data, forecast_dt)
    df = pd.DataFrame([input_data])[ROSSMANN_COLS]

    try:
        raw_pred = model.predict(df)
        pred_val = float(raw_pred[0]) if hasattr(raw_pred, "__len__") else float(raw_pred)
        predicted_sales = float(np.expm1(pred_val))
        actual_sales = _estimate_actual_sales(predicted_sales, data)
        average_sales = 5773.0
        delta = predicted_sales - average_sales
        pct_delta = (delta / average_sales) * 100 if average_sales else 0.0

        return {
            "predicted_sales": round(predicted_sales, 2),
            "actual_sales": actual_sales,
            "history": _build_demo_history(predicted_sales, actual_sales, forecast_dt),
            "benchmark": {
                "average_sales": average_sales,
                "delta_vs_average": round(delta, 2),
                "pct_vs_average": round(pct_delta, 2),
            },
            "derived_features": {
                "day_of_week": day_of_week,
                "month": input_data["Month"],
                "year": input_data["Year"],
                "week_of_year": input_data["WeekOfYear"],
                "is_promo_month": input_data["IsPromoMonth"],
                "lag_1_sales": round(float(data.lag_1_sales), 2),
                "lag_7_sales": round(float(data.lag_7_sales), 2),
                "rolling_7_mean_sales": round(float(data.rolling_7_mean_sales), 2),
                "customers": round(float(data.customers), 2),
            },
            "feature_importance": _extract_feature_importance(model),
        }
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

    try:
        cluster = int(f_kmeans.predict([[data.long, data.lat]])[0])
    except Exception:
        cluster = 0

    chip_val = 1 if str(data.use_chip).strip().lower() == "swipe" else 0

    input_data = {
        'current_age': 55, 'retirement_age': 66, 'credit_score': 685,
        'num_credit_cards': 3, 'num_cards_issued': 66, 'year_pin_last_changed': 2018,
        'amount': data.amount, 'use_chip': chip_val, 'gender': 1, 'card_brand': 2,
        'card_type': 1, 'has_chip': 1, 'years_with_bank': 10, 'location_cluster': cluster
    }

    cols = ['current_age', 'retirement_age', 'credit_score', 'num_credit_cards',
            'num_cards_issued', 'year_pin_last_changed', 'amount', 'use_chip',
            'gender', 'card_brand', 'card_type', 'has_chip', 'years_with_bank',
            'location_cluster']

    df = pd.DataFrame([input_data])[cols]

    try:
        df_scaled = f_scaler.transform(df)

        proba = None
        if hasattr(f_model, "predict_proba"):
            probs_all = f_model.predict_proba(df_scaled)      # shape (n_samples, n_classes)
            probs = probs_all[0]                              # first (only) sample

            if hasattr(f_model, "classes_"):
                try:
                    idx_pos = list(f_model.classes_).index(0)  # index of label '1' (fraud)
                except ValueError:
                    idx_pos = 1 if len(probs) > 1 else 0
                proba = float(probs[idx_pos])
            else:
                idx_pos = 1 if len(probs) > 1 else 0
                proba = float(probs[idx_pos])

            logger.debug("fraud_model.classes_: %s", getattr(f_model, "classes_", None))
            logger.debug("predict_proba sample probs: %s", probs)
        else:
            pred = f_model.predict(df_scaled)
            proba = 1.0 if pred[0] == 1 else 0.0

        proba = max(0.0, min(1.0, float(proba)))  # clamp
        return {
            "fraud_probability": round(proba, 4),
            "is_fraud": bool(proba > 0.5),
            "risk_level": "CRITICAL" if proba > 0.8 else "HIGH" if proba > 0.5 else "LOW"
        }

    except Exception as e:
        logger.exception("Fraud prediction failed: %s", e)
        raise HTTPException(status_code=500, detail="Fraud prediction failed")

# ---------------------------------------------------------------------
# 3) HOUSE PRICE ENDPOINT (cleaned)
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


def load_artifacts(path: str) -> Dict[str, Any]:
    """
    Tries to load artifacts from common formats (joblib/pickle).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Artifacts file not found at {path}")

    try:
        artifacts = joblib.load(path)
        if not isinstance(artifacts, dict):
            raise ValueError("Loaded object is not a dict of artifacts.")
        return artifacts
    except Exception:
        with open(path, "rb") as f:
            artifacts = pickle.load(f)
            if not isinstance(artifacts, dict):
                raise ValueError("Loaded object is not a dict of artifacts.")
            return artifacts


@app.post("/predict/house")
def predict_house(data: HouseInput):
    artifacts = models.get("house_artifacts")
    if artifacts is None:
        raise HTTPException(status_code=503, detail="House artifacts not loaded on server")

    model = artifacts.get("model")
    scaler = artifacts.get("scaler")
    preprocessor = artifacts.get("preprocessor")
    saved_feature_names = artifacts.get("feature_names") or artifacts.get("columns") or artifacts.get("feature_list")
    medians: Dict[str, Any] = artifacts.get("medians", {}) or {}
    kmeans = artifacts.get("kmeans") or artifacts.get("km") or artifacts.get("kmeans_model")
    le_type = artifacts.get("label_encoder_type") or artifacts.get("type_encoder")

    if model is None:
        raise HTTPException(status_code=500, detail="House artifact missing required key 'model'")

    notes: Dict[str, Any] = {}

    final_9 = [
        "area", "parking_spots", "attached_rooms", "age", "type",
        "total_rooms", "area_per_room", "distance_from_center", "location_cluster"
    ]

    model_expected_n: Optional[int] = None
    try:
        n_in = getattr(model, "n_features_in_", None)
        if n_in is not None:
            model_expected_n = int(n_in)
    except Exception:
        model_expected_n = None

    if saved_feature_names:
        if not isinstance(saved_feature_names, (list, tuple)):
            saved_feature_names = list(saved_feature_names)
        saved_len = len(saved_feature_names)
    else:
        saved_len = 0

    if model_expected_n is not None and saved_len != 0 and saved_len != model_expected_n:
        notes["feature_names_warning"] = (
            f"artifact.feature_names length ({saved_len}) != model.n_features_in_ ({model_expected_n}). "
            "Overriding with final 9-feature safe list."
        )
        feature_names = final_9
    elif saved_feature_names and saved_len == len(final_9):
        feature_names = list(saved_feature_names)
    else:
        if saved_feature_names and saved_len != len(final_9):
            notes["feature_names_info"] = (
                f"Using fallback final_9 features because artifact.feature_names length ({saved_len}) "
                f"doesn't match expected 9."
            )
        feature_names = final_9

    # Feature engineering
    if data.year_built:
        try:
            age = CURRENT_YEAR - int(data.year_built)
        except Exception:
            age = int(medians.get("age", medians.get("year_built", 0) or 0))
            notes["age_note"] = "year_built provided but parsing failed; used median fallback"
    else:
        age = int(medians.get("age", medians.get("year_built", 0) or 0))

    total_rooms = int(data.bedrooms + data.bathrooms + data.attached_rooms)
    area_per_room = float(data.area) / (total_rooms + 1)

    input_lat = data.lat
    input_lon = data.lon
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

    # location cluster
    if kmeans is not None and (input_lat is not None and input_lon is not None):
        try:
            try:
                location_cluster = int(kmeans.predict([[input_lat, input_lon]])[0])
            except Exception:
                location_cluster = int(kmeans.predict([[input_lon, input_lat]])[0])
        except Exception:
            location_cluster = int(medians.get("location_cluster", 0))
            notes["cluster_warning"] = "kmeans.predict failed; used median cluster."
    else:
        location_cluster = int(medians.get("location_cluster", 0))
        if kmeans is None:
            notes["cluster_warning"] = "kmeans not found in artifacts; used fallback cluster."

    # type encoding
    type_enc = None
    if le_type is not None:
        try:
            transformed = le_type.transform([str(data.type).strip().lower()])
            type_enc = int(transformed[0])
        except Exception:
            try:
                type_enc = int(le_type.get(str(data.type).strip().lower()))
            except Exception:
                type_enc = None

    if type_enc is None:
        fallback_type_map = {"apartment": 0, "house": 1, "other": 2}
        type_enc = fallback_type_map.get(str(data.type).strip().lower(), int(medians.get("type", 2)))
        notes["type_note"] = "Used fallback type mapping; save LabelEncoder in artifacts to avoid this."

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

    df_input = pd.DataFrame([candidate])

    missing: Dict[str, Any] = {}
    for col in feature_names:
        if col not in df_input.columns:
            df_input[col] = medians.get(col, 0.0)
            missing[col] = medians.get(col, 0.0)
    if missing:
        notes["missing_fallbacks"] = missing

    try:
        cols_to_pass = [c for c in feature_names if c in df_input.columns]
        df_input = df_input[cols_to_pass]
    except Exception as e:
        logger.exception("Feature ordering error: %s", e)
        raise HTTPException(status_code=500, detail=f"Feature ordering failure: {e}")

    if model_expected_n is not None:
        provided_n = df_input.shape[1]
        if provided_n != model_expected_n:
            notes["shape_mismatch_before_predict"] = f"model expects {model_expected_n} features but df has {provided_n}."
            forced = {}
            for col in final_9:
                if col not in df_input.columns:
                    df_input[col] = medians.get(col, 0.0)
                    forced[col] = medians.get(col, 0.0)
            df_input = df_input[[c for c in final_9]]
            notes["shape_mismatch_forced_fill"] = {"forced_fill": forced, "final_columns_used": final_9}
            if df_input.shape[1] != model_expected_n:
                raise HTTPException(
                    status_code=500,
                    detail=f"After forcing final_9 columns, still mismatch: model expects {model_expected_n}, df has {df_input.shape[1]}"
                )

    X_pred = None
    if preprocessor is not None:
        try:
            X_pred = preprocessor.transform(df_input)
        except Exception as e:
            notes["preprocessor_warning"] = f"preprocessor.transform failed: {e}; will attempt scaler or raw features"
            logger.warning("preprocessor.transform failed: %s", e)
    if X_pred is None:
        if scaler is not None:
            try:
                X_pred = scaler.transform(df_input.values)
            except Exception as e:
                notes["scaler_warning"] = f"scaler.transform failed: {e}; using raw features"
                logger.warning("scaler.transform failed: %s", e)
                X_pred = df_input.values
        else:
            X_pred = df_input.values

    try:
        raw_pred = model.predict(X_pred)
        pred_val = float(raw_pred[0]) if hasattr(raw_pred, "__len__") else float(raw_pred)
        predicted_price = max(0.0, pred_val)
    except Exception as e:
        logger.exception("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {traceback.format_exc()}")

    response = {
        "predicted_price": round(predicted_price, 2),
        "model_type": artifacts.get("model_type", getattr(model, "__class__", "unknown").__name__),
        "input_features": df_input.iloc[0].to_dict(),
        "notes": notes
    }

    if data.include_extras:
        extras: Dict[str, Any] = {}
        if hasattr(model, "feature_importances_"):
            try:
                extras["feature_importances"] = dict(zip(df_input.columns.tolist(), [float(x) for x in model.feature_importances_]))
            except Exception:
                extras["feature_importances"] = "failed to attach feature_importances"
        response["extras"] = extras

    return response


# ---------------------------------------------------------------------
# __main__ run block (use app directly)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")), log_level="info")

