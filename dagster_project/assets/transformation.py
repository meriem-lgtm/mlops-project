import subprocess
from dagster import asset, AssetExecutionContext, AssetIn


@asset(ins={"raw_earthquakes": AssetIn()})
def earthquake_features(context: AssetExecutionContext, raw_earthquakes: str) -> str:
    """
    Runs dbt build (staging -> intermediate -> marts)
    on top of DuckDB.
    """
    result = subprocess.run(
        [
            "dbt",
            "build",
            "--project-dir",
            "dbt_project",
            "--profiles-dir",
            "dbt_project",
        ],
        capture_output=True,
        text=True,
    )

    if result.stdout:
        context.log.info(result.stdout)

    if result.returncode != 0:
        if result.stderr:
            context.log.error(result.stderr)
        raise RuntimeError("dbt build failed")

    return "earthquake_features"
