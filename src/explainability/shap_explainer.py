"""
src/explainability/shap_explainer.py

Given a trained model and a single input row, returns the features that
most influenced the prediction. Used by api/predictor.py so /predict can
return an `explanation` field.
"""

import numpy as np
import shap

from src.features.engineering import FEATURES


def explain_prediction(
    model, X_row: np.ndarray, background: np.ndarray | None = None, top_k: int = 5
) -> dict:
    """
    model: a fitted sklearn/xgboost/lightgbm classifier or regressor
    X_row: shape (1, n_features), same feature order as FEATURES
    background: optional background dataset for the explainer (else zeros)
    """
    if background is None:
        background = np.zeros((1, X_row.shape[1]))

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_row)
    except Exception:
        # Fallback for models TreeExplainer doesn't support (e.g. LogisticRegression)
        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(X_row, nsamples=100)

    # Multi-class classifiers return a list of arrays (one per class);
    # use the class with the highest predicted probability if available.
    if isinstance(shap_values, list):
        if hasattr(model, "predict_proba"):
            pred_class = int(np.argmax(model.predict_proba(X_row)[0]))
        else:
            pred_class = 0
        values = shap_values[pred_class][0]
    else:
        values = np.array(shap_values).reshape(-1)

    contributions = list(zip(FEATURES, values))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    explanation = {}
    for name, val in contributions[:top_k]:
        strength = (
            "+++"
            if val > 0.5
            else (
                "++"
                if val > 0.15
                else (
                    "+"
                    if val > 0
                    else "---" if val < -0.5 else "--" if val < -0.15 else "-"
                )
            )
        )
        explanation[name] = {"impact": round(float(val), 4), "direction": strength}

    return explanation
