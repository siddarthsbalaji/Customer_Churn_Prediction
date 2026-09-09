"""
Single-Customer Churn Prediction Endpoint with Model Selection.
"""
from fastapi import APIRouter, HTTPException, Query, Request
import pandas as pd

from app.api.schemas import CustomerInput, PredictionResponse

router = APIRouter(tags=["Inference"])


@router.post("/predict", response_model=PredictionResponse)
def predict_single_customer(
    payload: CustomerInput,
    request: Request,
    model: str = Query(
        default="random_forest",
        pattern="^(random_forest|logistic_regression)$",
        description="Choose model: 'random_forest' or 'logistic_regression'"
    )
):
    """
    Evaluates calibrated churn risk probability for a single customer record
    using the specified model ('random_forest' or 'logistic_regression').
    """
    models = getattr(request.app.state, "models", {})
    metadata = getattr(request.app.state, "metadata", {})

    pipeline = models.get(model)
    if pipeline is None:
        pipeline = getattr(request.app.state, "pipeline", None)

    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model pipeline '{model}' is currently unavailable."
        )

    model_meta = metadata.get("models", {}).get(model, {})
    default_thresh = 0.210 if model == "random_forest" else 0.310
    threshold = model_meta.get("optimal_threshold", default_thresh)

    customer_dict = payload.model_dump()
    cust_id = customer_dict.get("customerID", "CUST-DEFAULT")
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
        model_used=model,
        churn_probability=round(prob, 4),
        is_at_risk=prob >= threshold,
        risk_tier=tier,
        decision_threshold_applied=threshold
    )
