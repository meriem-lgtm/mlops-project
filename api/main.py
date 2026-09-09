from fastapi import FastAPI

from api.routes import router

app = FastAPI(
    title="EarthquakeSafe API",
    description="Predicts earthquake magnitude class and value, with SHAP explanations.",
    version="1.0.0",
)

app.include_router(router)


# Run with:
# uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
