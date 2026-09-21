import time

from fastapi import APIRouter, HTTPException

from api.schemas import PredictRequest, PredictResponse, HealthResponse
from api.predictor import predictor
from monitoring.metrics import metrics_store

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health():
    return {"status": "healthy"}


@router.get("/metrics")
def metrics():
    return metrics_store.snapshot()


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    start = time.perf_counter()
    error = False

    try:
        result = predictor.predict(request.model_dump())

        metrics_store.record_prediction(result["predicted_class"], result["confidence"])

        return {
            "predicted_class": result["predicted_class"],
            "predicted_magnitude": result["predicted_magnitude"],
            "confidence": result["confidence"],
            "explanation": result["explanation"],
        }

    except Exception as e:
        error = True
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        latency = time.perf_counter() - start

        metrics_store.record_request(latency_s=latency, error=error)
