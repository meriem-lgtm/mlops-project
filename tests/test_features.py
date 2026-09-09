import pandas as pd

from src.features.engineering import clean_raw, add_features, FEATURES, region_enc, mag_class


def test_add_features_creates_all_expected_columns():
    df = pd.read_csv("data/raw/database.csv")
    df = clean_raw(df)
    df, _ = add_features(df)
    for col in FEATURES:
        assert col in df.columns


def test_region_enc_is_deterministic():
    assert region_enc(10, 10) == region_enc(10, 10)
    assert isinstance(region_enc(-10, -50), int)


def test_mag_class_thresholds():
    assert mag_class(4.9) == "Faible"
    assert mag_class(5.0) == "Moyen"
    assert mag_class(6.4) == "Moyen"
    assert mag_class(6.5) == "Fort"
