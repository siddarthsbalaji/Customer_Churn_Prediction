"""
FastAPI Application Entrypoint for Customer Churn Decision Engine.
Implements non-blocking lifespan state loading and modular route registration for dual models.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import batch, compare, explain, health, predict
from src.config import (
    XGB_EXPLAINER_PATH,
    XGB_PIPELINE_PATH,
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
    Pre-loads Random Forest and XGBoost pipelines, explainers, and metadata into app.state.
    """
    print("\n--- Initializing Customer Churn Decision Engine API (Dual Model Support) ---")
    models={}
    explainers={}
    if RF_PIPELINE_PATH.exists() and METADATA_ARTIFACT_PATH.exists():
        rf_pipe, metadata=load_pipeline_artifacts(RF_PIPELINE_PATH, METADATA_ARTIFACT_PATH)
        models["random_forest"]=rf_pipe
        app.state.metadata=metadata
        print("Loaded Random Forest pipeline.")
    elif PIPELINE_ARTIFACT_PATH.exists() and METADATA_ARTIFACT_PATH.exists():
        rf_pipe, metadata=load_pipeline_artifacts(PIPELINE_ARTIFACT_PATH, METADATA_ARTIFACT_PATH)
        models["random_forest"]=rf_pipe
        app.state.metadata=metadata
        print("Loaded default champion pipeline as Random Forest.")
    else:
        app.state.metadata=None
    if XGB_PIPELINE_PATH.exists():
        xgb_pipe, _=load_pipeline_artifacts(XGB_PIPELINE_PATH, METADATA_ARTIFACT_PATH)
        models["xgboost"]=xgb_pipe
        print("Loaded XGBoost pipeline.")
    if RF_EXPLAINER_PATH.exists():
        rf_exp=load_shap_explainer(RF_EXPLAINER_PATH)
        explainers["random_forest"]=rf_exp
        print("Loaded Random Forest TreeExplainer.")
    elif SHAP_EXPLAINER_ARTIFACT_PATH.exists():
        rf_exp=load_shap_explainer(SHAP_EXPLAINER_ARTIFACT_PATH)
        explainers["random_forest"]=rf_exp
    if XGB_EXPLAINER_PATH.exists():
        xgb_exp=load_shap_explainer(XGB_EXPLAINER_PATH)
        explainers["xgboost"]=xgb_exp
        print("Loaded XGBoost TreeExplainer.")
    app.state.models=models
    app.state.explainers=explainers
    app.state.pipeline=models.get("random_forest")
    app.state.explainer=explainers.get("random_forest")
    print(f"Active models in memory: {list(models.keys())}")
    print("API is ready to accept inference requests.\n")
    yield
    app.state.models.clear()
    app.state.explainers.clear()
    app.state.pipeline=None
    app.state.explainer=None
    app.state.metadata=None
    print("API server shutting down. Artifacts cleared from memory.")
app=FastAPI(
    title="Customer Churn Decision Engine API",
    description=(
        "Production REST API serving calibrated customer churn risk predictions, "
        "local SHAP root-cause attributions, batch CSV schema validation, and Next-Best-Action playbooks "
        "across Random Forest and XGBoost models."
    ),
    version="1.0.0",
    lifespan=lifespan
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(explain.router)
app.include_router(batch.router)
app.include_router(compare.router)
