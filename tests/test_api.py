import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Mock des modèles AVANT d'importer l'API
# Cela évite l'erreur FileNotFoundError dans GitHub Actions
with patch('api.predictor.Predictor._load_models'):
    from api.main import app

client = TestClient(app)


def test_health_returns_200():
    """Test que l'endpoint health fonctionne"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict_valid_request_returns_200():
    """Test qu'une requête valide retourne 200"""
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
    
    # Mock de la méthode predict pour éviter d'avoir besoin des vrais modèles
    with patch('api.predictor.Predictor.predict') as mock_predict:
        mock_predict.return_value = {
            "predicted_class": "moderate",
            "predicted_magnitude": 5.5,
            "confidence": 0.85,
            "explanation": {"top_features": ["depth", "latitude"]}
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert "predicted_class" in body
        assert "predicted_magnitude" in body


def test_predict_invalid_request_returns_422():
    """Test qu'une requête invalide retourne 422"""
    payload = {
        "latitude": 999,  # valeur hors limite
        "longitude": -5.8,
        "depth": 10
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422