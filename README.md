# EarthquakeSafe — Earthquake MLOps

## 1. Présentation
Projet MLOps complet de bout en bout : ingestion, stockage, transformation,
qualité des données, machine learning, suivi d'expériences, explicabilité,
déploiement, CI/CD et monitoring, appliqué à la prédiction de la magnitude
des séismes.

## 2. Problématique
Voir `docs/project_vision.md`.

## 3. Dataset
`data/raw/database.csv` — export historique de séismes (USGS) : Date, Time,
Latitude, Longitude, Type, Depth, Magnitude (+ métadonnées).

## 4. Architecture
Voir `docs/architecture.md` pour le schéma complet (ingestion → stockage →
transformation → qualité → ML → déploiement → monitoring).

## 5. Data Strategy
Voir `docs/data_strategy.md`.

## 6. Ingestion (dlt)
```bash
python dlt_pipeline/earthquake_pipeline.py
```
Charge `data/raw/database.csv` dans DuckDB (table `raw_earthquakes`).

## 7. DuckDB
Entrepôt local : `data/warehouse/earthquakes.duckdb`.

## 8. dbt
```bash
cd dbt_project
dbt build --profiles-dir .
```
staging → intermediate → marts (`earthquake_features`).

## 9. Data Quality
```bash
dbt test --profiles-dir .
```
Tests de complétude, validité (bornes) et unicité, définis dans les
`schema.yml` de chaque couche.

## 10. Data Contract
`docs/data_contract.yml` définit les garanties sur `earthquake_features`.

## 11. Data Lineage
`docs/data_lineage.png` (généré, voir schéma).

## 12. Dagster (orchestration)
```bash
dagster dev -f dagster_project/definitions.py
```
DAG : ingestion → transformation → data_quality_check → trained_models.
Si `data_quality_check` échoue, `trained_models` ne s'exécute pas.

## 13. Machine Learning
Modules dans `src/` : `data/prepare.py`, `features/engineering.py`,
`training/train_classifier.py`, `training/train_regressor.py`,
`evaluation/evaluate.py`.

## 14. MLflow
```bash
mlflow ui   # http://localhost:5000
python src/training/train_classifier.py
python src/training/train_regressor.py
```
Chaque run logge paramètres, métriques (accuracy/F1/recall pour la
classification, MAE/RMSE/R² pour la régression) et le modèle. Les meilleurs
modèles sont enregistrés dans le Model Registry sous `earthquake_classifier`
et `earthquake_regressor`.

## 15. SHAP
`src/explainability/shap_explainer.py` retourne les features les plus
influentes pour une prédiction donnée ; exposé via le champ `explanation`
de `/predict`.

## 16. FastAPI
```bash
uvicorn api.main:app --reload
```
- `GET /health` → `{"status": "healthy"}`
- `POST /predict` → classe, magnitude, confiance, explication SHAP

Documentation interactive : http://localhost:8000/docs

## 17. Docker
```bash
docker compose up --build
```
Lance l'API (port 8000) et un serveur MLflow (port 5000).

## 18. CI/CD
`.github/workflows/ci.yml` : install → lint → tests → build Docker, à
chaque push/PR sur `main`.

## 19. Monitoring
`monitoring/metrics.py` (requêtes, latence, erreurs, disponibilité,
distribution des prédictions, confiance) et `monitoring/drift.py` (PSI sur
latitude/longitude/depth, valeurs manquantes).

## 20. Installation
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 21. Utilisation
Ordre recommandé : dlt → dbt → Dagster (ou directement `src/training/*`) →
MLflow → API → Dashboard.

## 22. Tests
```bash
pytest tests/ -v
```
Couvre : validité des données, feature engineering, chargement des
modèles et prédiction, endpoints API (200 / 422).

## 23. Résultats
À compléter avec les métriques obtenues après entraînement (voir MLflow UI
pour les valeurs exactes d'accuracy/F1/MAE/RMSE par modèle).

---

## Structure du projet
```
earthquake-mlops/
├── data/{raw,processed,warehouse}
├── dlt_pipeline/
├── dbt_project/{models/{staging,intermediate,marts},tests,macros}
├── dagster_project/{assets,definitions.py}
├── src/{data,features,training,evaluation,explainability}
├── api/
├── monitoring/
├── tests/
├── models/
├── docs/
├── .github/workflows/
├── Dockerfile, docker-compose.yml, requirements.txt
```

## Fichiers hérités
`earthquake_model_legacy.py` et `dashboard_legacy.py` sont les scripts
originaux, conservés pour référence. La logique de `earthquake_model_legacy.py`
a été modularisée dans `src/`. `dashboard_legacy.py` doit être branché sur
l'API (`/predict`) plutôt que de charger directement les `.pkl`.
