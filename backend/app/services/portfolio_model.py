from __future__ import annotations

from pathlib import Path

import joblib

from app.services.reusable_match_model import ResumeMatchModel


_MODEL_CACHE = None
_MODEL_PATH = None


def _load_model(model_path: str):
    global _MODEL_CACHE, _MODEL_PATH
    resolved = str(Path(model_path).resolve())
    if _MODEL_CACHE is not None and _MODEL_PATH == resolved:
        return _MODEL_CACHE

    try:
        model = joblib.load(resolved)
    except Exception:
        model = ResumeMatchModel()

    _MODEL_CACHE = model
    _MODEL_PATH = resolved
    return model


def score_resume(
    resume_path: str,
    jd_text: str,
    model_path: str = "models/portfolio_resume_match_model.joblib",
):
    model = _load_model(model_path)
    if hasattr(model, "score_resume"):
        return model.score_resume(resume_path, jd_text)
    raise AttributeError("Loaded model does not implement score_resume(resume_path, jd_text).")
