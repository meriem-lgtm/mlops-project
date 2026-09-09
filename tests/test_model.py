import os
import joblib
import numpy as np
import pytest

MODELS_DIR = "models"


@pytest.mark.skipif(
    not os.path.exists(f"{MODELS_DIR}/best_classifier.pkl"),
    reason="No trained model found; run src/training/train_classifier.py first",
)
def test_classifier_loads_and_predicts():
    clf = joblib.load(f"{MODELS_DIR}/best_classifier.pkl")
    features = joblib.load(f"{MODELS_DIR}/features.pkl")
    X = np.zeros((1, len(features)))
    pred = clf.predict(X)
    assert pred.shape == (1,)


@pytest.mark.skipif(
    not os.path.exists(f"{MODELS_DIR}/best_regressor.pkl"),
    reason="No trained model found; run src/training/train_regressor.py first",
)
def test_regressor_loads_and_predicts():
    reg = joblib.load(f"{MODELS_DIR}/best_regressor.pkl")
    features = joblib.load(f"{MODELS_DIR}/features.pkl")
    X = np.zeros((1, len(features)))
    pred = reg.predict(X)
    assert pred.shape == (1,)
