from fastapi import APIRouter, HTTPException

from api.schemas import PredictRequest, PredictResponse, HealthResponse
from api.predictor import predictor

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health():
    return {"status": "healthy"}


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        result = predictor.predict(request.model_dump())

        return {
            "predicted_class": result["predicted_class"],
            "predicted_magnitude": result["predicted_magnitude"],
            "confidence": result["confidence"],
            "explanation": result["explanation"],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
