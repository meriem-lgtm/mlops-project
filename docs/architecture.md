# Architecture

## Vue d'ensemble

```
database.csv (USGS)
      │
      ▼
     dlt  ──────────────►  DuckDB (data/warehouse/earthquakes.duckdb)
                                 │
                                 ▼
                                dbt
                     staging → intermediate → marts
                                 │
                                 ▼
                       earthquake_features
                                 │
                 ┌───────────────┼───────────────┐
                 ▼               ▼               ▼
           Data Quality     src/training     Dagster
          (dbt tests,      (classification,  orchestrates
           contract)        régression)      the whole DAG
                                 │
                                 ▼
                              MLflow
                     (tracking + model registry)
                                 │
                                 ▼
                             FastAPI  ◄──── SHAP explainability
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                                ▼
           Dashboard (Streamlit)             Monitoring
                                        (API, ML, data drift)
```

## Orchestration (Dagster)
```
INGESTION → TRANSFORMATION → DATA QUALITY → TRAIN MODEL → MLFLOW
                                    │
                          FAILED ──►│──► TRAINING STOPPED
```

## Déploiement (Docker)
- `api` : FastAPI service (serves the MLflow-registered model)
- `mlflow` : tracking server + model registry UI
- Optionnel : `dagster` webserver pour visualiser l'orchestration

## CI/CD (GitHub Actions)
```
push → install → lint → tests → build docker image → success/failure
```
