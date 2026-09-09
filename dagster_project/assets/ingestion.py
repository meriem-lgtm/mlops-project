import subprocess
import sys

from dagster import asset, AssetExecutionContext


@asset
def raw_earthquakes(context: AssetExecutionContext) -> str:
    result = subprocess.run(
        [
            sys.executable,
            "dlt_pipeline/earthquake_pipeline.py",
        ],
        capture_output=True,
        text=True,
    )

    if result.stdout:
        context.log.info(result.stdout)

    if result.returncode != 0:
        if result.stderr:
            context.log.error(result.stderr)

        raise RuntimeError(
            f"dlt ingestion failed with exit code {result.returncode}"
        )

    return "data/warehouse/earthquakes.duckdb"