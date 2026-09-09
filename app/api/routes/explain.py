"""
Single-Customer SHAP Explainability & Prescriptive Action Endpoint.
"""
from fastapi import APIRouter, HTTPException, Request
import pandas as pd

from app.api.schemas import CustomerInput, ExplanationResponse, PrescribedActionResponse
from src.decision_engine.rules import prescribe_retention_action
from src.explainability.shap_service import explain_single_customer

router = APIRouter(tags=["Explainability & Decision Engine"])


@router.post("/explain", response_model=ExplanationResponse)
def explain_single_customer_route(payload: CustomerInput, request: Request):
    """
    Computes local SHAP attributions and returns root-cause risk drivers
    alongside tailored Next-Best-Action retention playbooks.
    """
    pipeline = getattr(request.app.state, "pipeline", None)
    explainer = getattr(request.app.state, "explainer", None)
    metadata = getattr(request.app.state, "metadata", {})

    if pipeline is None or explainer is None:
        raise HTTPException(
            status_code=503,
            detail="Model pipeline or SHAP explainer service is currently unavailable."
        )

    customer_dict = payload.model_dump()
    cust_id = customer_dict.get("customerID", "CUST-DEFAULT")
    feature_names = metadata.get("encoded_feature_names", [])

    df = pd.DataFrame([customer_dict])

    try:
        explanation = explain_single_customer(
            pipeline=pipeline,
            explainer=explainer,
            customer_raw_df=df,
            feature_names=feature_names,
            top_k=5
        )
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"SHAP attribution computation failed: {str(e)}"
        )

    prob = explanation["prediction_probability"]
    tier = "CRITICAL" if prob >= 0.70 else ("MODERATE" if prob >= 0.40 else "LOW")

    # Determine Prescriptive Retention Action Playbook
    prescribed = prescribe_retention_action(
        churn_prob=prob,
        top_shap_drivers=explanation["risk_drivers"],
        customer_record=customer_dict
    )

    action_response = PrescribedActionResponse(
        action_code=prescribed["action_code"],
        action_title=prescribed["action_title"],
        priority=prescribed["priority"],
        incentive_type=prescribed["incentive_type"],
        estimated_cost=prescribed["estimated_cost"],
        recommended_channel=prescribed["recommended_channel"],
        playbook_details=prescribed["playbook_details"],
        trigger_rationale=prescribed["trigger_rationale"]
    )

    return ExplanationResponse(
        customer_id=cust_id,
        churn_probability=prob,
        base_value=explanation["base_value"],
        risk_tier=tier,
        risk_drivers=explanation["risk_drivers"],
        protective_factors=explanation["protective_factors"],
        prescribed_action=action_response
    )
