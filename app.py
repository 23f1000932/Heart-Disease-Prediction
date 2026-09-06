"""
FastAPI service for the heart disease prediction model.

Endpoints:
    GET  /health   - liveness/readiness check
    POST /predict  - predict heart disease for a single patient

Usage (local):
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import os
from datetime import datetime, timezone

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = os.environ.get("MODEL_PATH", "model/model.joblib")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("heart-disease-api")

app = FastAPI(title="Heart Disease Prediction API", version="1.0.0")

_artifact = None
_model = None
_feature_columns = None


class PatientFeatures(BaseModel):
    age: float = Field(..., example=63)
    gender: int = Field(..., description="0 or 1, as encoded by training (see /health for mapping)", example=0)
    cp: int = Field(..., description="chest pain type (0-3)", example=3)
    trestbps: float = Field(..., description="resting blood pressure", example=145.0)
    chol: float = Field(..., description="serum cholesterol mg/dl", example=233.0)
    fbs: int = Field(..., description="fasting blood sugar > 120 mg/dl (0/1)", example=1)
    restecg: int = Field(..., description="resting ECG results (0,1,2)", example=0)
    thalach: float = Field(..., description="max heart rate achieved", example=150.0)
    exang: int = Field(..., description="exercise induced angina (0/1)", example=0)
    oldpeak: float = Field(..., example=2.3)
    slope: int = Field(..., example=0)
    ca: int = Field(..., description="number of major vessels (0-3)", example=0)
    thal: int = Field(..., example=1)


class PredictionResponse(BaseModel):
    prediction: str
    probability_yes: float
    timestamp: str


@app.on_event("startup")
def load_model():
    global _artifact, _model, _feature_columns
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model artifact not found at {MODEL_PATH}")
    _artifact = joblib.load(MODEL_PATH)
    _model = _artifact["model"]
    _feature_columns = _artifact["feature_columns"]
    logger.info(f"Model loaded from {MODEL_PATH}. Features: {_feature_columns}")


@app.get("/health")
def health():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "ok",
        "model_loaded": True,
        "feature_columns": _feature_columns,
        "gender_categories": _artifact.get("gender_categories"),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientFeatures):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    row = pd.DataFrame([patient.dict()])[_feature_columns]

    pred = _model.predict(row)[0]
    proba = _model.predict_proba(row)[0]
    classes = list(_model.classes_)
    prob_yes = float(proba[classes.index("yes")]) if "yes" in classes else float(max(proba))

    timestamp = datetime.now(timezone.utc).isoformat()

    logger.info(
        "prediction request | input=%s | prediction=%s | probability_yes=%.4f | timestamp=%s",
        patient.dict(), pred, prob_yes, timestamp,
    )

    return PredictionResponse(
        prediction=str(pred),
        probability_yes=prob_yes,
        timestamp=timestamp,
    )
