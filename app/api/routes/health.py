"""
Health and Model Metadata Endpoints with Multi-Model Status.
"""
from typing import Any, Dict
from fastapi import APIRouter, Request
from app.api.schemas import HealthResponse
router=APIRouter(tags=["System & Metadata"])
@router.get("/health", response_model=HealthResponse)
def get_health(request: Request):
    """Health check verifying dual model pipelines and explainers readiness in memory."""
    models=getattr(request.app.state, "models", {})
    explainers=getattr(request.app.state, "explainers", {})
    has_pipeline=hasattr(request.app.state, "pipeline") and request.app.state.pipeline is not None
    has_explainer=hasattr(request.app.state, "explainer") and request.app.state.explainer is not None
    available_models=list(models.keys()) if models else (["random_forest"] if has_pipeline else [])
    status = "healthy" if (len(available_models)>0 and (len(explainers)>0 or has_explainer)) else "degraded"
    return HealthResponse(
        status=status,
        service="Customer Churn Decision Engine API",
        version="1.0.0",
        available_models=available_models,
        model_loaded=len(available_models)>0 or has_pipeline,
        explainer_loaded=len(explainers)>0 or has_explainer
    )
@router.get("/metadata")
def get_model_metadata(request: Request) -> Dict[str, Any]:
    """Returns training metadata, tournament metrics, and multi-model thresholds."""
    if hasattr(request.app.state, "metadata") and request.app.state.metadata is not None:
        return request.app.state.metadata
    return {"message": "Metadata not loaded."}
