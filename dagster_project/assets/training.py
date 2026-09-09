import subprocess
import sys

from dagster import asset, AssetExecutionContext, AssetIn


@asset(ins={"data_quality_check": AssetIn()})
def trained_models(context: AssetExecutionContext, data_quality_check: bool) -> str:

    scripts = [
        "src/training/train_classifier.py",
        "src/training/train_regressor.py",
    ]

    for script in scripts:

        context.log.info(f"Starting training: {script}")

        result = subprocess.run(
            [
                sys.executable,
                script,
            ],
            capture_output=True,
            text=True,
        )

        if result.stdout:
            context.log.info(result.stdout)

        if result.returncode != 0:

            if result.stderr:
                context.log.error(result.stderr)

            raise RuntimeError(f"{script} failed with exit code {result.returncode}")

    context.log.info("All ML models trained successfully.")

    return "models trained and logged to MLflow"
