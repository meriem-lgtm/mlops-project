# User Stories détaillées

## US-01 — Ingestion automatisée
**En tant que** data engineer
**Je veux** que les données USGS soient ingérées automatiquement via dlt
**Afin de** ne plus dépendre d'un simple `pd.read_csv` non traçable et non relançable.

**Critères d'acceptation**
- Le pipeline vérifie l'existence et la non-vacuité de la source
- Les données atterrissent dans DuckDB, table `raw_earthquakes`
- Le pipeline est relançable sans erreur (idempotent)

## US-02 — Qualité des données
**En tant que** data analyst
**Je veux** que les règles de validité (bornes, unicité, complétude) soient testées automatiquement
**Afin de** ne jamais entraîner un modèle sur des données corrompues.

**Critères d'acceptation**
- `dbt test` échoue si une règle du data contract est violée
- L'asset Dagster `quality` bloque `training` en cas d'échec

## US-03 — Suivi des expériences
**En tant que** data scientist
**Je veux** que chaque entraînement soit enregistré dans MLflow (paramètres, métriques, artefacts)
**Afin de** comparer objectivement les modèles et reproduire les résultats.

## US-04 — Explicabilité
**En tant qu'** utilisateur métier
**Je veux** voir les features qui ont le plus influencé une prédiction (SHAP)
**Afin de** faire confiance au modèle.

## US-05 — Exposition via API
**En tant qu'** utilisateur
**Je veux** un endpoint `/predict` qui renvoie classe, magnitude et explication
**Afin d'** intégrer le modèle dans d'autres applications (dashboard, tiers).
