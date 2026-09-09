"""
src/training/train_regressor.py

Trains an XGBoost regressor to predict the exact magnitude value,
logs the run to MLflow and registers it as 'earthquake_regressor'.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import joblib
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

from src.training.train_classifier import get_training_data
from src.features.engineering import (
    clean_raw,
    add_features,
    ensure_ml_features,
    build_feature_matrix,
    FEATURES,
)
from src.data.prepare import load_dataset


RANDOM_STATE = 42
MLFLOW_EXPERIMENT = "earthquake_regression"
REGISTERED_MODEL_NAME = "earthquake_regressor"


def get_regression_data():
    df = load_dataset()

    if "magnitude" not in [c.lower() for c in df.columns]:
        raise ValueError("Could not find a magnitude column in the dataset.")

    if "magnitude_class" not in df.columns:
        # Dataset came from raw CSV / raw_earthquakes
        df.columns = [c.title() for c in df.columns]
        df = clean_raw(df)
        df, _ = add_features(df)
    else:
        # Dataset came from the dbt/DuckDB mart
        df, _ = ensure_ml_features(df)

    X = build_feature_matrix(df)

    y = df["Magnitude"]

    return X, y.values

def main():
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    X, y = get_regression_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    with mlflow.start_run(run_name="XGBRegressor"):
        reg = xgb.XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.05, random_state=RANDOM_STATE
        )
        reg.fit(X_train, y_train)
        pred = reg.predict(X_test)

        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)

        mlflow.log_params(reg.get_params())
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.xgboost.log_model(
            reg, artifact_path="model", registered_model_name=REGISTERED_MODEL_NAME
        )

        print(f"MAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.3f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(reg, "models/best_regressor.pkl")
    print("Saved local copy to models/best_regressor.pkl")


if __name__ == "__main__":
    main()
