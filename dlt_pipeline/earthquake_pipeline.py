"""
dlt_pipeline/earthquake_pipeline.py

Phase 3 of the project: replace `pd.read_csv("database.csv")` with a real
ingestion pipeline:

    database.csv (USGS export)
            |
           dlt
            |
         DuckDB  (data/warehouse/earthquakes.duckdb, table: raw_earthquakes)

Run it with:
    python dlt_pipeline/earthquake_pipeline.py

It is idempotent / relaunchable: it uses a `replace` write disposition
keyed on nothing (full refresh) which is fine for a static historical
CSV. If you later plug in the live USGS API instead of the CSV, switch
write_disposition to "merge" with primary_key="id".
"""

import os
import dlt
import pandas as pd

RAW_CSV_PATH = os.path.join("data", "raw", "database.csv")
WAREHOUSE_PATH = os.path.join("data", "warehouse", "earthquakes.duckdb")


def _check_source(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Source file not found: {path}. Put the USGS export at data/raw/database.csv"
        )
    if os.path.getsize(path) == 0:
        raise ValueError(f"Source file {path} is empty.")


@dlt.resource(name="raw_earthquakes", write_disposition="replace")
def earthquakes_source():
    """Reads the raw USGS earthquake export and yields rows for dlt."""
    _check_source(RAW_CSV_PATH)
    df = pd.read_csv(RAW_CSV_PATH)
    # normalize column names to snake_case-friendly identifiers for SQL/dbt
    df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]
    yield df.to_dict(orient="records")


def run() -> None:
    os.makedirs(os.path.dirname(WAREHOUSE_PATH), exist_ok=True)

    pipeline = dlt.pipeline(
        pipeline_name="earthquake_pipeline",
        destination=dlt.destinations.duckdb(WAREHOUSE_PATH),
        dataset_name="main",
    )

    load_info = pipeline.run(earthquakes_source())
    print(load_info)
    print(f"[dlt] Loaded data into {WAREHOUSE_PATH}, table 'raw_earthquakes'")


if __name__ == "__main__":
    run()
