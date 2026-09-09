"""
api/predictor.py

Loads the classifier + regressor (preferably from the MLflow Model
Registry, falling back to the local models/*.pkl copies) and builds the
feature vector for a single incoming request, using the SAME feature
logic as training (src/features/engineering.py).
"""

import os
import numpy as np
import joblib

from src.features.engineering import FEATURES, region_enc
from src.explainability.shap_explainer import explain_prediction


MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000"
)

USE_MLFLOW_REGISTRY = (
    os.environ.get("USE_MLFLOW_REGISTRY", "false").lower() == "true"
)


class Predictor:
    def __init__(self):
        self.classifier = None
        self.regressor = None
        self.scaler = None
        self.label_encoder = None
        self.type_encoder = None

        self._load()

    def _load(self):
        """
        Load models from MLflow Registry if enabled.
        Otherwise, load local .pkl models.
        """

        if USE_MLFLOW_REGISTRY:
            try:
                import mlflow

                mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

                self.classifier = mlflow.pyfunc.load_model(
                    "models:/earthquake_classifier/Production"
                )

                self.regressor = mlflow.pyfunc.load_model(
                    "models:/earthquake_regressor/Production"
                )

                print("[predictor] Loaded models from MLflow Model Registry")

            except Exception as e:
                print(
                    f"[predictor] Could not load from MLflow ({e}); "
                    "falling back to local .pkl files"
                )

        # Fallback to local models
        if self.classifier is None:
            self.classifier = joblib.load(
                "models/best_classifier.pkl"
            )

            self.regressor = joblib.load(
                "models/best_regressor.pkl"
            )

            self.scaler = joblib.load(
                "models/scaler.pkl"
            )

            self.label_encoder = joblib.load(
                "models/label_encoder.pkl"
            )

            self.type_encoder = joblib.load(
                "models/type_encoder.pkl"
            )

            print(
                "[predictor] Loaded models from local models/*.pkl"
            )

    def build_features(self, payload: dict) -> np.ndarray:
        """
        Build the same 13 features used during model training.
        """

        lat = payload["latitude"]
        lon = payload["longitude"]
        depth = payload["depth"]

        # Encode earthquake type
        type_str = payload.get("type", "Earthquake")

        if (
            self.type_encoder is not None
            and type_str in list(self.type_encoder.classes_)
        ):
            type_enc = self.type_encoder.transform([type_str])[0]
        else:
            # Unknown type → fallback value
            type_enc = 0

        # Build feature dictionary
        row = {
            "Latitude": lat,
            "Longitude": lon,
            "Depth": depth,

            "Year": payload["year"],
            "Month": payload["month"],
            "Day": payload["day"],
            "Hour": payload["hour"],

            "Type_enc": type_enc,
            "Region_enc": region_enc(lat, lon),

            # Same binning logic as dbt + training
            "lat_bin": int(np.floor((lat + 90) / 10)),
            "lon_bin": int(np.floor((lon + 180) / 10)),

            "distance_center": float(
                np.sqrt(lat ** 2 + lon ** 2)
            ),

            "is_deep": int(depth > 300),
        }

        # Respect the exact training feature order
        X = np.array(
            [[row[feature] for feature in FEATURES]],
            dtype=float
        )

        return X

    def predict(self, payload: dict) -> dict:
        """
        Generate classification, regression and SHAP explanation.
        """

        # Build input features
        X = self.build_features(payload)

        # -----------------------------
        # Classification
        # -----------------------------
        pred_class_idx = self.classifier.predict(X)[0]

        if self.label_encoder is not None:
            pred_class = self.label_encoder.inverse_transform(
                [pred_class_idx]
            )[0]
        else:
            pred_class = str(pred_class_idx)

        # -----------------------------
        # Confidence
        # -----------------------------
        if hasattr(self.classifier, "predict_proba"):
            probabilities = self.classifier.predict_proba(X)
            confidence = float(np.max(probabilities))
        else:
            confidence = 1.0

        # -----------------------------
        # Regression
        # -----------------------------
        pred_magnitude = float(
            self.regressor.predict(X)[0]
        )

        # -----------------------------
        # SHAP explanation
        # -----------------------------
        explanation = explain_prediction(
            self.classifier,
            X,
            top_k=5
        )

        # -----------------------------
        # Final response
        # -----------------------------
        return {
            "predicted_class": pred_class,
            "predicted_magnitude": round(
                pred_magnitude,
                2
            ),
            "confidence": round(
                confidence,
                4
            ),
            "explanation": explanation,
            "_features": X,
        }


# Global predictor instance
predictor = Predictor()