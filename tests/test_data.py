import os
import pytest
import pandas as pd

from src.features.engineering import clean_raw

# تخطي جميع الاختبارات في هذا الملف إذا لم يكن ملف البيانات موجوداً (مثل بيئة CI/CD)
pytestmark = pytest.mark.skipif(
    not os.path.exists("data/raw/database.csv"),
    reason="Raw data not available in CI environment",
)


def test_raw_data_has_required_columns():
    df = pd.read_csv("data/raw/database.csv")
    required = {"Date", "Time", "Latitude", "Longitude", "Type", "Depth", "Magnitude"}
    assert required.issubset(set(df.columns))


def test_clean_raw_respects_bounds():
    df = pd.read_csv("data/raw/database.csv")
    cleaned = clean_raw(df)
    assert cleaned["Depth"].between(0, 800).all()
    assert cleaned["Magnitude"].between(0, 10).all()
    assert cleaned["Latitude"].between(-90, 90).all()
    assert cleaned["Longitude"].between(-180, 180).all()


def test_no_nulls_in_key_columns():
    df = pd.read_csv("data/raw/database.csv")
    cleaned = clean_raw(df)
    assert (
        cleaned[["Latitude", "Longitude", "Depth", "Magnitude"]].isnull().sum().sum()
        == 0
    )
