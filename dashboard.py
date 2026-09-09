"""
dashboard.py

Phase 19: the dashboard no longer loads the .pkl files directly.
It calls the FastAPI service instead:

    Dashboard (Streamlit) --HTTP--> FastAPI (/predict) --> MLflow model --> SHAP

Run:
    uvicorn api.main:app --reload          # in one terminal
    streamlit run dashboard.py             # in another
"""

import os
import requests
import streamlit as st
import plotly.graph_objects as go

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="EarthquakeSafe", page_icon="🌐", layout="wide")

st.title("🌐 EarthquakeSafe — Prédiction sismique")
st.caption(f"Modèle servi via l'API FastAPI ({API_URL}), pas de chargement direct des .pkl.")

# Check API health
try:
    health = requests.get(f"{API_URL}/health", timeout=3).json()
    st.success(f"API status: {health.get('status')}")
except Exception:
    st.error(f"Impossible de contacter l'API sur {API_URL}. Lance `uvicorn api.main:app --reload`.")

st.divider()

col1, col2 = st.columns(2)
with col1:
    latitude = st.number_input("Latitude", -90.0, 90.0, 35.7)
    longitude = st.number_input("Longitude", -180.0, 180.0, -5.8)
    depth = st.number_input("Profondeur (km)", 0.0, 800.0, 10.0)
    eq_type = st.selectbox("Type", ["Earthquake", "Nuclear Explosion", "Explosion", "Rock Burst"])

with col2:
    year = st.number_input("Année", 1900, 2100, 2026)
    month = st.slider("Mois", 1, 12, 9)
    day = st.slider("Jour", 1, 31, 7)
    hour = st.slider("Heure", 0, 23, 14)

if st.button("Prédire", type="primary"):
    payload = {
        "latitude": latitude, "longitude": longitude, "depth": depth,
        "year": int(year), "month": int(month), "day": int(day), "hour": int(hour),
        "type": eq_type,
    }
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()

        c1, c2, c3 = st.columns(3)
        c1.metric("Classe prédite", result["predicted_class"])
        c2.metric("Magnitude estimée", f"{result['predicted_magnitude']:.2f}")
        c3.metric("Confiance", f"{result['confidence']*100:.1f}%")

        st.subheader("Explication (SHAP)")
        explanation = result.get("explanation", {})
        if explanation and "error" not in explanation:
            names = list(explanation.keys())
            impacts = [v["impact"] for v in explanation.values()]
            fig = go.Figure(go.Bar(x=impacts, y=names, orientation="h"))
            fig.update_layout(title="Contribution des features à la prédiction", height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(explanation.get("error", "Pas d'explication disponible."))

    except Exception as e:
        st.error(f"Erreur lors de l'appel à l'API : {e}")
