from dagster import Definitions, load_assets_from_modules

from dagster_project.assets import ingestion, transformation, quality, training

all_assets = load_assets_from_modules([ingestion, transformation, quality, training])

defs = Definitions(assets=all_assets)

# Run locally with:
# dagster dev -f dagster_project/definitions.py
