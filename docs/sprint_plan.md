# Sprint Plan (Agile)

## Sprint 1 — Fondations
- Cadrage (project_vision.md)
- Data Strategy (data_strategy.md)
- Organisation Agile (backlog, user stories)
- dlt (ingestion)
- DuckDB (stockage)

## Sprint 2 — Qualité des données
- dbt (staging, intermediate, marts)
- Data Contract
- Data Quality (tests dbt)
- Data Lineage (diagramme)
- Dagster : assets ingestion/transformation/quality

## Sprint 3 — Machine Learning
- Modularisation du script ML existant (src/)
- MLflow (tracking)
- Model Registry (best classifier / regressor versionnés)
- SHAP (explicabilité)
- Dagster : asset training branché sur MLflow

## Sprint 4 — Mise en production
- Tests (data, features, model, API)
- FastAPI (/health, /predict)
- Docker / docker-compose
- GitHub Actions (CI/CD)
- Monitoring (API, ML, data)
- Dashboard branché sur l'API (plus de chargement direct du .pkl)
- Documentation finale + rapport + soutenance
