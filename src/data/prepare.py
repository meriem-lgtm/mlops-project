"""
src/data/prepare.py

Loads earthquake data for training. Prefers reading from the DuckDB
warehouse (populated by dlt + dbt). Falls back to the raw CSV so the
project still works before the warehouse is built (Phase 3/4/5).
"""

import os
import pandas as pd

WAREHOUSE_PATH = os.path.join("data", "warehouse", "earthquakes.duckdb")
RAW_CSV_PATH = os.path.join("data", "raw", "database.csv")


def load_dataset() -> pd.DataFrame:
    try:
        import duckdb

        if os.path.exists(WAREHOUSE_PATH):
            con = duckdb.connect(WAREHOUSE_PATH, read_only=True)
            tables = [t[0] for t in con.execute("show tables").fetchall()]
            # Prefer the dbt mart if it has been built, else the raw dlt table
            for candidate in (
                "earthquake_features",
                "stg_earthquakes",
                "raw_earthquakes",
            ):
                if candidate in tables:
                    df = con.execute(f"select * from {candidate}").fetchdf()
                    con.close()
                    print(
                        f"[prepare] Loaded {len(df)} rows from DuckDB table '{candidate}'"
                    )
                    return df
            con.close()
    except Exception as e:
        print(f"[prepare] DuckDB unavailable, falling back to CSV ({e})")

    df = pd.read_csv(RAW_CSV_PATH)
    print(f"[prepare] Loaded {len(df)} rows from raw CSV")
    return df


if __name__ == "__main__":
    d = load_dataset()
    print(d.shape)
    print(d.head())
