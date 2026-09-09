"""
src/features/engineering.py

Feature engineering logic, extracted from the original earthquake_model.py
so that training, the API and (conceptually) dbt all share ONE definition
of what a "feature" is. This avoids train/serve skew.

If you port this logic into dbt (int_earthquakes_clean.sql /
earthquake_features.sql), keep the two in sync.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

FEATURES = [
    "Latitude",
    "Longitude",
    "Depth",
    "Year",
    "Month",
    "Day",
    "Hour",
    "Type_enc",
    "Region_enc",
    "lat_bin",
    "lon_bin",
    "distance_center",
    "is_deep",
]


def region_enc(lat: float, lon: float) -> int:
    """Very coarse geographic bucketing used as a categorical feature."""
    if lat >= 0 and -30 <= lon <= 60:
        return 0
    if lat >= 0 and lon > 60:
        return 1
    if lat >= 0 and lon < -30:
        return 2
    if lat < 0 and -30 <= lon <= 60:
        return 3
    if lat < 0 and lon > 60:
        return 4
    return 5


def mag_class(m: float) -> str:
    if m < 5:
        return "Faible"
    elif m < 6.5:
        return "Moyen"
    return "Fort"


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Equivalent of the dbt staging + intermediate layers."""
    keep = ["Date", "Time", "Latitude", "Longitude", "Type", "Depth", "Magnitude"]
    df = df[keep].drop_duplicates()

    df["Datetime"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        errors="coerce",
        utc=True,
    )

    df = df.dropna(subset=["Datetime", "Magnitude", "Latitude", "Longitude", "Depth"])
    df["Datetime"] = df["Datetime"].dt.tz_convert(None)

    df = df[(df["Depth"] >= 0) & (df["Depth"] <= 800)]
    df = df[(df["Magnitude"] >= 0) & (df["Magnitude"] <= 10)]
    return df


def add_features(df: pd.DataFrame, type_encoder: LabelEncoder | None = None):
    """Equivalent of the dbt marts layer (earthquake_features)."""
    df = df.copy()
    df["Year"] = df["Datetime"].dt.year
    df["Month"] = df["Datetime"].dt.month
    df["Day"] = df["Datetime"].dt.day
    df["Hour"] = df["Datetime"].dt.hour

    df["Region_enc"] = [
        region_enc(a, b) for a, b in zip(df["Latitude"], df["Longitude"])
    ]
    df["lat_bin"] = np.floor((df["Latitude"] + 90) / 10).astype(int)
    df["lon_bin"] = np.floor((df["Longitude"] + 180) / 10).astype(int)

    if type_encoder is None:
        type_encoder = LabelEncoder()
        df["Type_enc"] = type_encoder.fit_transform(df["Type"].astype(str))
    else:
        # unseen categories at inference time fall back to 0
        classes = list(type_encoder.classes_)
        df["Type_enc"] = (
            df["Type"]
            .astype(str)
            .apply(lambda t: type_encoder.transform([t])[0] if t in classes else 0)
        )

    df["distance_center"] = np.sqrt(df["Latitude"] ** 2 + df["Longitude"] ** 2)

    df["is_deep"] = (df["Depth"] > 300).astype(int)

    if "Magnitude" in df.columns:
        df["magnitude_class"] = df["Magnitude"].apply(mag_class)

    return df, type_encoder


def ensure_ml_features(df: pd.DataFrame, type_encoder: LabelEncoder | None = None):
    """
    Complete ML-specific features when the input comes from the dbt mart.
    dbt provides the data features, while Type_enc is created here because
    it depends on the fitted ML encoder.
    """
    df = df.copy()

    # Normalize DuckDB/dbt lowercase columns to the names expected by ML
    rename_map = {
        "latitude": "Latitude",
        "longitude": "Longitude",
        "depth": "Depth",
        "magnitude": "Magnitude",
        "type": "Type",
        "year": "Year",
        "month": "Month",
        "day": "Day",
        "hour": "Hour",
        "region_enc": "Region_enc",
        "lat_bin": "lat_bin",
        "lon_bin": "lon_bin",
        "distance_center": "distance_center",
        "is_deep": "is_deep",
        "magnitude_class": "magnitude_class",
    }

    df = df.rename(columns=rename_map)

    if "Type" not in df.columns:
        raise ValueError("Missing 'Type' column required to create Type_enc.")

    if type_encoder is None:
        type_encoder = LabelEncoder()
        df["Type_enc"] = type_encoder.fit_transform(df["Type"].astype(str))
    else:
        classes = set(type_encoder.classes_)
        df["Type_enc"] = (
            df["Type"]
            .astype(str)
            .apply(lambda t: (type_encoder.transform([t])[0] if t in classes else 0))
        )

    return df, type_encoder


def build_feature_matrix(df: pd.DataFrame):
    X = df[FEATURES].fillna(0).values
    return X
