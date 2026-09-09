"""
FastAPI Application Entrypoint for Customer Churn Decision Engine.
Implements non-blocking lifespan state loading and modular route registration.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import batch, explain, health, predict
from src.config import (
    METADATA_ARTIFACT_PATH,
    PIPELINE_ARTIFACT_PATH,
    SHAP_EXPLAINER_ARTIFACT_PATH,
)
from src.explainability.shap_service import load_shap_explainer
from src.models.pipeline import load_pipeline_artifacts


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application Lifespan Event Handler.
    Pre-loads model pipeline, SHAP explainer, and metadata into app.state on startup.
    """
    print("\n--- Initializing Customer Churn Decision Engine API ---")

    # Load Model Pipeline & Metadata
    if PIPELINE_ARTIFACT_PATH.exists() and METADATA_ARTIFACT_PATH.exists():
        pipeline, metadata = load_pipeline_artifacts(
            PIPELINE_ARTIFACT_PATH, METADATA_ARTIFACT_PATH
        )
        app.state.pipeline = pipeline
        app.state.metadata = metadata
        print(f"Loaded champion pipeline: {metadata.get('model_name')}")
        print(f"Active decision threshold: {metadata.get('optimal_threshold')}")
    else:
        app.state.pipeline = None
        app.state.metadata = None
        print("WARNING: Model artifacts not found. Please train models first.")

    # Load SHAP Explainer
    if SHAP_EXPLAINER_ARTIFACT_PATH.exists():
        explainer = load_shap_explainer(SHAP_EXPLAINER_ARTIFACT_PATH)
        app.state.explainer = explainer
        print("Loaded SHAP TreeExplainer service.")
    else:
        app.state.explainer = None
        print("WARNING: SHAP explainer artifact not found.")

    print("API is ready to accept inference requests.\n")
    yield

    # Clean up on shutdown
    app.state.pipeline = None
    app.state.explainer = None
    app.state.metadata = None
    print("API server shutting down. Artifacts cleared from memory.")


app = FastAPI(
    title="Customer Churn Decision Engine API",
    description=(
        "Production REST API serving calibrated customer churn risk predictions, "
        "local SHAP root-cause attributions, batch CSV schema validation, and Next-Best-Action playbooks."
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
