# Data Strategy

## Flux de données

```
Source (USGS database.csv)
        ↓
       Raw (dlt → DuckDB, table raw_earthquakes)
        ↓
 Transformation (dbt: staging → intermediate → marts)
        ↓
     Features (earthquake_features)
        ↓
         ML (classification + régression)
```

## Variables sources principales
- Date
- Time
- Latitude
- Longitude
- Type
- Depth
- Magnitude

## Features ML dérivées
- Year, Month, Day, Hour (dérivées de Date + Time)
- Type_enc (encodage du type d'événement)
- Region_enc (bucket géographique grossier)
- lat_bin, lon_bin (discrétisation spatiale)
- distance_center (distance euclidienne à (0,0))
- depth_ratio (profondeur / (magnitude + 1))
- is_deep (profondeur > 300 km)

## Qualité et gouvernance
- Le contrat de données (`docs/data_contract.yml`) définit les bornes acceptables.
- Les tests dbt (`schema.yml`) vérifient complétude, validité, unicité.
- Le data lineage (`docs/data_lineage.png`) documente la traçabilité complète.

## Responsabilités
- Ingestion : dlt_pipeline/
- Stockage : DuckDB (data/warehouse/earthquakes.duckdb)
- Transformation : dbt_project/
- Qualité : dbt tests + Dagster asset checks
- Consommation : src/training, api/
