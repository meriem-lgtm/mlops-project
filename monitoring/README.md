# Monitoring

## API
- `request count`, `avg latency`, `error rate`, `uptime` — collected in-process by `metrics.py`
  and can be exposed via a `/metrics` endpoint (add it to `api/routes.py` if needed).

## ML
- `prediction distribution` (Faible / Moyen / Fort over time)
- `avg confidence` of the classifier

## Data / Model drift
- `drift.py` implements the Population Stability Index (PSI) between the
  training reference distribution and a recent sample of production inputs
  for latitude, longitude and depth.
- `check_missing_values` reports the missing-value rate per column.
- PSI > 0.2 is treated as significant drift and should trigger a retraining
  review (ideally as a Dagster sensor).

## How to run a drift check manually
```python
import pandas as pd
from monitoring.drift import drift_report

reference = pd.read_csv("data/raw/database.csv")
recent = pd.read_csv("data/processed/recent_requests.csv")  # logged production inputs
print(drift_report(reference, recent))
```
