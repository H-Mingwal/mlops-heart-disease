"""FastAPI model-serving API with request logging and Prometheus metrics."""
from pathlib import Path
import logging
import time
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("heart-disease-api")
MODEL_PATH = Path("models/model.joblib")

REQUEST_COUNT = Counter("api_requests_total", "Total API requests", ["method", "endpoint", "status"])
REQUEST_LATENCY = Histogram("api_request_duration_seconds", "API request latency", ["endpoint"])


class PatientFeatures(BaseModel):
    age: float = Field(..., ge=1, le=120)
    sex: float = Field(..., ge=0, le=1)
    cp: float = Field(..., ge=0, le=4)
    trestbps: float = Field(..., ge=0, le=400)
    chol: float = Field(..., ge=0, le=1000)
    fbs: float = Field(..., ge=0, le=1)
    restecg: float = Field(..., ge=0, le=3)
    thalach: float = Field(..., ge=0, le=300)
    exang: float = Field(..., ge=0, le=1)
    oldpeak: float = Field(..., ge=-10, le=20)
    slope: float = Field(..., ge=0, le=3)
    ca: float = Field(..., ge=0, le=4)
    thal: float = Field(..., ge=0, le=7)


app = FastAPI(title="Heart Disease Risk Classification API", version="1.0.0",
              description="Educational ML demo; not for medical diagnosis.")


@app.middleware("http")
async def observe_requests(request: Request, call_next):
    start = time.perf_counter()
    status = "500"
    try:
        response = await call_next(request)
        status = str(response.status_code)
        return response
    finally:
        duration = time.perf_counter() - start
        REQUEST_COUNT.labels(request.method, request.url.path, status).inc()
        REQUEST_LATENCY.labels(request.url.path).observe(duration)
        logger.info("method=%s path=%s status=%s latency_seconds=%.6f",
                    request.method, request.url.path, status, duration)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict(patient: PatientFeatures) -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model not found. Run `python -m src.train` first.")
    try:
        bundle = joblib.load(MODEL_PATH)
        model = bundle["model"]
        data = pd.DataFrame([patient.model_dump()])
        expected = bundle["feature_names"]
        # Keep the same feature order as training; missing/extra fields are rejected by schema/model checks.
        data = data.reindex(columns=expected)
        prediction = int(model.predict(data)[0])
        probability = float(model.predict_proba(data)[0, 1])
        return {
            "prediction": prediction,
            "label": "disease_present" if prediction == 1 else "disease_not_detected",
            "confidence": probability,
            "model": bundle.get("selected_model", "unknown"),
            "disclaimer": "Educational prediction only; not a medical diagnosis."
        }
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed; check server logs.") from exc
