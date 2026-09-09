import subprocess

from dagster import asset, AssetExecutionContext, AssetIn, Failure


@asset(ins={"earthquake_features": AssetIn()})
def data_quality_check(
    context: AssetExecutionContext, earthquake_features: str
) -> bool:
    """
    Runs dbt tests against the transformed earthquake data.

    If any dbt test fails, the Dagster asset raises Failure
    and the training asset will not execute.
    """

    result = subprocess.run(
        [
            "dbt",
            "test",
            "--project-dir",
            "dbt_project",
            "--profiles-dir",
            "dbt_project",
        ],
        capture_output=True,
        text=True,
    )

    # Display dbt output in Dagster logs
    if result.stdout:
        context.log.info(result.stdout)

    if result.returncode != 0:
        if result.stderr:
            context.log.error(result.stderr)

        raise Failure(
            description="Data quality checks FAILED - training will not run.",
            metadata={"dbt_output": result.stdout[-2000:]},
        )

    context.log.info("Data quality checks PASSED.")

    return True
