from pydantic import BaseModel, Field
from typing import Dict, Any


class PredictRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    depth: float = Field(..., ge=0, le=800)
    year: int = Field(..., ge=1900, le=2100)
    month: int = Field(..., ge=1, le=12)
    day: int = Field(..., ge=1, le=31)
    hour: int = Field(..., ge=0, le=23)
    type: str = "Earthquake"

    class Config:
        json_schema_extra = {
            "example": {
                "latitude": 35.7,
                "longitude": -5.8,
                "depth": 10,
                "year": 2026,
                "month": 9,
                "day": 7,
                "hour": 14,
                "type": "Earthquake",
            }
        }


class PredictResponse(BaseModel):
    predicted_class: str
    predicted_magnitude: float
    confidence: float
    explanation: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
