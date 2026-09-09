"""
FastAPI Application Entrypoint for Customer Churn Decision Engine.
Implements non-blocking lifespan state loading and modular route registration for dual models.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import batch, compare, explain, health, predict
from src.config import (
    LR_EXPLAINER_PATH,
    LR_PIPELINE_PATH,
    METADATA_ARTIFACT_PATH,
    PIPELINE_ARTIFACT_PATH,
    RF_EXPLAINER_PATH,
    RF_PIPELINE_PATH,
    SHAP_EXPLAINER_ARTIFACT_PATH,
)
from src.explainability.shap_service import load_shap_explainer
from src.models.pipeline import load_pipeline_artifacts


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application Lifespan Event Handler.
    Pre-loads Random Forest and Logistic Regression pipelines, explainers, and metadata into app.state.
    """
    print("\n--- Initializing Customer Churn Decision Engine API (Dual Model Support) ---")

    models = {}
    explainers = {}

    # 1. Load Model Pipelines
    if RF_PIPELINE_PATH.exists() and METADATA_ARTIFACT_PATH.exists():
        rf_pipe, metadata = load_pipeline_artifacts(RF_PIPELINE_PATH, METADATA_ARTIFACT_PATH)
        models["random_forest"] = rf_pipe
        app.state.metadata = metadata
        print("Loaded Random Forest pipeline.")
    elif PIPELINE_ARTIFACT_PATH.exists() and METADATA_ARTIFACT_PATH.exists():
        rf_pipe, metadata = load_pipeline_artifacts(PIPELINE_ARTIFACT_PATH, METADATA_ARTIFACT_PATH)
        models["random_forest"] = rf_pipe
        app.state.metadata = metadata
        print("Loaded default champion pipeline as Random Forest.")
    else:
        app.state.metadata = None

    if LR_PIPELINE_PATH.exists():
        lr_pipe, _ = load_pipeline_artifacts(LR_PIPELINE_PATH, METADATA_ARTIFACT_PATH)
        models["logistic_regression"] = lr_pipe
        print("Loaded Logistic Regression pipeline.")

    # 2. Load SHAP Explainers
    if RF_EXPLAINER_PATH.exists():
        rf_exp = load_shap_explainer(RF_EXPLAINER_PATH)
        explainers["random_forest"] = rf_exp
        print("Loaded Random Forest TreeExplainer.")
    elif SHAP_EXPLAINER_ARTIFACT_PATH.exists():
        rf_exp = load_shap_explainer(SHAP_EXPLAINER_ARTIFACT_PATH)
        explainers["random_forest"] = rf_exp

    if LR_EXPLAINER_PATH.exists():
        lr_exp = load_shap_explainer(LR_EXPLAINER_PATH)
        explainers["logistic_regression"] = lr_exp
        print("Loaded Logistic Regression LinearExplainer.")

    # Set application state attributes
    app.state.models = models
    app.state.explainers = explainers
    app.state.pipeline = models.get("random_forest")
    app.state.explainer = explainers.get("random_forest")

    print(f"Active models in memory: {list(models.keys())}")
    print("API is ready to accept inference requests.\n")
    yield

    # Clean up on shutdown
    app.state.models.clear()
    app.state.explainers.clear()
    app.state.pipeline = None
    app.state.explainer = None
    app.state.metadata = None
    print("API server shutting down. Artifacts cleared from memory.")


app = FastAPI(
    title="Customer Churn Decision Engine API",
    description=(
        "Production REST API serving calibrated customer churn risk predictions, "
        "local SHAP root-cause attributions, batch CSV schema validation, and Next-Best-Action playbooks "
        "across Random Forest and Logistic Regression models."
    ),
    version="1.0.0",
    lifespan=lifespan
)

# Enable Cross-Origin Resource Sharing (CORS) for UI integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Endpoint Routers
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(explain.router)
app.include_router(batch.router)
app.include_router(compare.router)
