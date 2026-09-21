"""
api/predictor.py

Loads the classifier + regressor from local models/*.pkl files.
For the large classifier, it supports both:
    - models/best_classifier.pkl
    - models/best_classifier.pkl.gz

The predictor builds exactly the same 13 features used during training
and generates a SHAP explanation for the prediction.
"""

import gzip
import os

import joblib
import numpy as np

from src.explainability.shap_explainer import explain_prediction
from src.features.engineering import FEATURES, region_enc

MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000",
)

USE_MLFLOW_REGISTRY = os.environ.get("USE_MLFLOW_REGISTRY", "false").lower() == "true"


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

        Otherwise, load local models from models/.

        The classifier can be loaded from:
            models/best_classifier.pkl
        or:
            models/best_classifier.pkl.gz
        """

        # ============================================================
        # 1. Try MLflow Model Registry if explicitly enabled
        # ============================================================
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
                    "[predictor] Could not load from MLflow "
                    f"({e}); falling back to local models"
                )

        # ============================================================
        # 2. Local models fallback
        # ============================================================
        if self.classifier is None:

            model_dir = "models"

            if not os.path.exists(model_dir):
                print(
                    "[predictor] Warning: 'models' directory not found. "
                    "Running without local models."
                )
                return

            try:

                # ----------------------------------------------------
                # Classifier
                # ----------------------------------------------------
                classifier_path = os.path.join(
                    model_dir,
                    "best_classifier.pkl",
                )

                classifier_gz_path = os.path.join(
                    model_dir,
                    "best_classifier.pkl.gz",
                )

                # First choice: normal .pkl
                if os.path.exists(classifier_path):

                    self.classifier = joblib.load(classifier_path)

                    print("[predictor] Loaded classifier " "from best_classifier.pkl")

                # Second choice: compressed .pkl.gz
                elif os.path.exists(classifier_gz_path):

                    with gzip.open(
                        classifier_gz_path,
                        "rb",
                    ) as f:

                        self.classifier = joblib.load(f)

                    print(
                        "[predictor] Loaded classifier " "from best_classifier.pkl.gz"
                    )

                else:

                    raise FileNotFoundError(
                        "Neither best_classifier.pkl nor "
                        "best_classifier.pkl.gz was found"
                    )

                # ----------------------------------------------------
                # Regressor
                # ----------------------------------------------------
                self.regressor = joblib.load(
                    os.path.join(
                        model_dir,
                        "best_regressor.pkl",
                    )
                )

                # ----------------------------------------------------
                # Scaler
                # ----------------------------------------------------
                self.scaler = joblib.load(
                    os.path.join(
                        model_dir,
                        "scaler.pkl",
                    )
                )

                # ----------------------------------------------------
                # Label encoder
                # ----------------------------------------------------
                self.label_encoder = joblib.load(
                    os.path.join(
                        model_dir,
                        "label_encoder.pkl",
                    )
                )

                # ----------------------------------------------------
                # Type encoder
                # ----------------------------------------------------
                self.type_encoder = joblib.load(
                    os.path.join(
                        model_dir,
                        "type_encoder.pkl",
                    )
                )

                print("[predictor] Loaded remaining models from models/")

            except FileNotFoundError as e:

                print(
                    "[predictor] Warning: Could not find local "
                    f"model files ({e}). Running without them."
                )

            except Exception as e:

                print("[predictor] Warning: Unexpected error " f"loading models: {e}")

    def build_features(self, payload: dict) -> np.ndarray:
        """
        Build the same 13 features used during model training.

        Features:
            Latitude
            Longitude
            Depth
            Year
            Month
            Day
            Hour
            Type_enc
            Region_enc
            lat_bin
            lon_bin
            distance_center
            is_deep
        """

        lat = payload["latitude"]
        lon = payload["longitude"]
        depth = payload["depth"]

        type_str = payload.get(
            "type",
            "Earthquake",
        )

        # ============================================================
        # Encode earthquake type
        # ============================================================
        if self.type_encoder is not None and type_str in list(
            self.type_encoder.classes_
        ):

            type_enc = self.type_encoder.transform([type_str])[0]

        else:

            type_enc = 0

        # ============================================================
        # Build feature row
        # ============================================================
        row = {
            "Latitude": lat,
            "Longitude": lon,
            "Depth": depth,
            "Year": payload["year"],
            "Month": payload["month"],
            "Day": payload["day"],
            "Hour": payload["hour"],
            "Type_enc": type_enc,
            "Region_enc": region_enc(
                lat,
                lon,
            ),
            "lat_bin": int(np.floor((lat + 90) / 10)),
            "lon_bin": int(np.floor((lon + 180) / 10)),
            "distance_center": float(np.sqrt(lat**2 + lon**2)),
            "is_deep": int(depth > 300),
        }

        # ============================================================
        # Keep exactly the training feature order
        # ============================================================
        X = np.array(
            [[row[feature] for feature in FEATURES]],
            dtype=float,
        )

        return X

    def predict(self, payload: dict) -> dict:
        """
        Generate:

            - earthquake magnitude class
            - predicted magnitude
            - confidence
            - SHAP explanation
        """

        # ============================================================
        # Check that models are available
        # ============================================================
        if self.classifier is None or self.regressor is None:

            raise RuntimeError(
                "Models are not loaded. " "Please make sure the model files exist."
            )

        # ============================================================
        # Build features
        # ============================================================
        X = self.build_features(payload)

        # ============================================================
        # Classification
        # ============================================================
        pred_class_idx = self.classifier.predict(X)[0]

        # ============================================================
        # Convert encoded class to original class name
        # ============================================================
        if self.label_encoder is not None:

            pred_class = self.label_encoder.inverse_transform([pred_class_idx])[0]

        else:

            pred_class = str(pred_class_idx)

        # ============================================================
        # Confidence
        # ============================================================
        if hasattr(
            self.classifier,
            "predict_proba",
        ):

            probabilities = self.classifier.predict_proba(X)

            confidence = float(np.max(probabilities))

        else:

            confidence = 1.0

        # ============================================================
        # Regression
        # ============================================================
        pred_magnitude = float(self.regressor.predict(X)[0])

        # ============================================================
        # SHAP explanation
        # ============================================================
        explanation = explain_prediction(
            self.classifier,
            X,
            top_k=5,
        )

        # ============================================================
        # Return result
        # ============================================================
        return {
            "predicted_class": pred_class,
            "predicted_magnitude": round(
                pred_magnitude,
                2,
            ),
            "confidence": round(
                confidence,
                4,
            ),
            "explanation": explanation,
            "_features": X,
        }


# ================================================================
# Global predictor instance
# ================================================================
predictor = Predictor()
