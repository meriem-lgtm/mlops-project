# Project Vision — EarthquakeSafe

## Problématique
Les séismes constituent un phénomène naturel difficile à prévoir. L'objectif du
projet est d'utiliser les données historiques de séismes afin de construire des
modèles capables de classifier la magnitude et d'estimer sa valeur, tout en
fournissant une architecture MLOps reproductible permettant l'ingestion, la
transformation, l'entraînement et le déploiement du modèle.

## Objectifs
- Prédire la classe de magnitude (Faible / Moyen / Fort)
- Prédire la magnitude exacte (régression)
- Automatiser l'ingestion des données (dlt)
- Garantir la qualité des données (Data Contract, tests, lineage)
- Suivre les expériences ML (MLflow)
- Expliquer les prédictions (SHAP)
- Exposer le modèle via une API (FastAPI)
- Conteneuriser l'ensemble (Docker)
- Automatiser les tests et le déploiement (GitHub Actions)
- Monitorer l'API, le modèle et les données en production

## Périmètre
Le projet couvre l'ensemble de la chaîne MLOps : ingestion → stockage →
transformation → qualité → entraînement → suivi d'expériences → déploiement →
monitoring → documentation.

## Non-objectifs
- Prédiction en temps réel à partir de capteurs sismiques
- Alerte grand public (hors périmètre académique)
