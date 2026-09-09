from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict_valid_request_returns_200():
    payload = {
        "latitude": 35.7,
        "longitude": -5.8,
        "depth": 10,
        "year": 2026,
        "month": 9,
        "day": 7,
        "hour": 14,
        "type": "Earthquake",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "predicted_class" in body
    assert "predicted_magnitude" in body


def test_predict_invalid_request_returns_422():
    payload = {"latitude": 999, "longitude": -5.8, "depth": 10}  # missing fields + out of range
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
