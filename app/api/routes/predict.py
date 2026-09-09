"""
Single-Customer Churn Prediction Endpoint.
"""
from fastapi import APIRouter, HTTPException, Request
import pandas as pd

from app.api.schemas import CustomerInput, PredictionResponse

router = APIRouter(tags=["Inference"])


@router.post("/predict", response_model=PredictionResponse)
def predict_single_customer(payload: CustomerInput, request: Request):
    """
    Evaluates calibrated churn risk probability for a single customer record.
    """
    pipeline = getattr(request.app.state, "pipeline", None)
    metadata = getattr(request.app.state, "metadata", {})

    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model pipeline is currently unavailable. Please verify startup logs."
        )

    threshold = metadata.get("optimal_threshold", 0.210)
    customer_dict = payload.model_dump()
    cust_id = customer_dict.get("customerID", "CUST-DEFAULT")

    # Ingest record as single-row DataFrame
    df = pd.DataFrame([customer_dict])

    try:
        prob = float(pipeline.predict_proba(df)[0][1])
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Inference pipeline execution error: {str(e)}"
        )

    # Risk Tier assignment
    if prob >= 0.70:
        tier = "CRITICAL"
    elif prob >= 0.40:
        tier = "MODERATE"
    else:
        tier = "LOW"

    return PredictionResponse(
        customer_id=cust_id,
        churn_probability=round(prob, 4),
        is_at_risk=prob >= threshold,
        risk_tier=tier,
        decision_threshold_applied=threshold
    )
