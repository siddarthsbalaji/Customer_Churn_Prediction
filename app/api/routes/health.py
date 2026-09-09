"""
Health and Model Metadata Endpoints.
"""
from typing import Any, Dict
from fastapi import APIRouter, Request

from app.api.schemas import HealthResponse

router = APIRouter(tags=["System & Metadata"])


@router.get("/health", response_model=HealthResponse)
def get_health(request: Request):
    """Health check verifying model and explainer readiness in memory."""
    has_pipeline = hasattr(request.app.state, "pipeline") and request.app.state.pipeline is not None
    has_explainer = hasattr(request.app.state, "explainer") and request.app.state.explainer is not None

    status = "healthy" if (has_pipeline and has_explainer) else "degraded"
    return HealthResponse(
        status=status,
        service="Customer Churn Decision Engine API",
        version="1.0.0",
        model_loaded=has_pipeline,
        explainer_loaded=has_explainer
    )


@router.get("/metadata")
def get_model_metadata(request: Request) -> Dict[str, Any]:
    """Returns training metadata, tournament metrics, and calibrated threshold."""
    if hasattr(request.app.state, "metadata") and request.app.state.metadata is not None:
        return request.app.state.metadata
    return {"message": "Metadata not loaded."}
