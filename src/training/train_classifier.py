"""
src/training/train_classifier.py

Trains and compares several classifiers on magnitude_class, logs every
run to MLflow, and registers the best one under 'earthquake_classifier'
in the MLflow Model Registry.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import joblib
import mlflow
import mlflow.sklearn
import xgboost as xgb
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight

from src.data.prepare import load_dataset
from src.features.engineering import (
    clean_raw,
    add_features,
    ensure_ml_features,
    build_feature_matrix,
    FEATURES,
)

RANDOM_STATE = 42
MLFLOW_EXPERIMENT = "earthquake_classification"
REGISTERED_MODEL_NAME = "earthquake_classifier"


def get_training_data():
    df = load_dataset()

    if "magnitude_class" not in df.columns:
        # Dataset came from raw CSV / raw_earthquakes
        df.columns = [
            (
                c.strip().title()
                if c.lower()
                in (
                    "date",
                    "time",
                    "latitude",
                    "longitude",
                    "type",
                    "depth",
                    "magnitude",
                )
                else c
            )
            for c in df.columns
        ]

        df = clean_raw(df)
        df, type_encoder = add_features(df)

    else:
        # Dataset came from the dbt/DuckDB mart
        df, type_encoder = ensure_ml_features(df)

    X = build_feature_matrix(df)

    le_y = LabelEncoder()
    y = le_y.fit_transform(df["magnitude_class"])

    return X, y, le_y, type_encoder


def main():
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    X, y, le_y, type_encoder = get_training_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    weights = compute_sample_weight("balanced", y_train)

    models = {
        "LogReg": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=600,
            max_depth=7,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            verbose=-1,
        ),
    }

    best_f1 = -1
    best_name = None
    best_model = None

    for name, model in models.items():

        if name == "LogReg":
            Xtr = X_train_s
            Xte = X_test_s
        else:
            Xtr = X_train
            Xte = X_test

        with mlflow.start_run(run_name=name):

            model.fit(
                Xtr,
                y_train,
                sample_weight=weights,
            )

            pred = model.predict(Xte)

            acc = accuracy_score(y_test, pred)
            f1 = f1_score(
                y_test,
                pred,
                average="weighted",
            )
            rec = recall_score(
                y_test,
                pred,
                average="weighted",
            )
            prec = precision_score(
                y_test,
                pred,
                average="weighted",
            )

            mlflow.log_param(
                "model_type",
                name,
            )

            mlflow.log_params(model.get_params())

            mlflow.log_metric(
                "accuracy",
                acc,
            )

            mlflow.log_metric(
                "f1_weighted",
                f1,
            )

            mlflow.log_metric(
                "recall_weighted",
                rec,
            )

            mlflow.log_metric(
                "precision_weighted",
                prec,
            )

            # Log model with pickle to support XGBoost / LightGBM
            mlflow.sklearn.log_model(
                model,
                name="model",
                serialization_format="pickle",
            )

            print(f"{name}: " f"acc={acc:.3f} " f"f1={f1:.3f} " f"recall={rec:.3f}")

            if f1 > best_f1:
                best_f1 = f1
                best_name = name
                best_model = model

    print(f"\nBEST MODEL: " f"{best_name} " f"(f1={best_f1:.3f})")

    # Register the best model
    with mlflow.start_run(run_name=f"{best_name}_registered"):

        mlflow.log_param(
            "model_type",
            best_name,
        )

        mlflow.log_metric(
            "f1_weighted",
            best_f1,
        )

        mlflow.sklearn.log_model(
            best_model,
            name="model",
            serialization_format="pickle",
            registered_model_name=REGISTERED_MODEL_NAME,
        )

    # Save local copies
    os.makedirs(
        "models",
        exist_ok=True,
    )

    joblib.dump(
        best_model,
        "models/best_classifier.pkl",
    )

    joblib.dump(
        scaler,
        "models/scaler.pkl",
    )

    joblib.dump(
        le_y,
        "models/label_encoder.pkl",
    )

    joblib.dump(
        type_encoder,
        "models/type_encoder.pkl",
    )

    joblib.dump(
        FEATURES,
        "models/features.pkl",
    )

    print("Saved local copies to models/ " "(used as fallback by the API).")


if __name__ == "__main__":
    main()
