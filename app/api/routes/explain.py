"""
Single-Customer SHAP Explainability & Prescriptive Action Endpoint with Model Selection.
"""
from fastapi import APIRouter, HTTPException, Query, Request
import pandas as pd
from app.api.schemas import CustomerInput, ExplanationResponse, PrescribedActionResponse
from src.decision_engine.rules import prescribe_retention_action
from src.explainability.shap_service import explain_single_customer
router=APIRouter(tags=["Explainability & Decision Engine"])
@router.post("/explain", response_model=ExplanationResponse)
def explain_single_customer_route(
    payload: CustomerInput,
    request: Request,
    model: str=Query(
        default="random_forest",
        pattern="^(random_forest|logistic_regression)$",
        description="Choose model: 'random_forest' or 'logistic_regression'"
    )
):
    """
    Computes local SHAP attributions and returns root-cause risk drivers
    alongside tailored Next-Best-Action retention playbooks for the chosen model.
    """
    models=getattr(request.app.state, "models", {})
    explainers=getattr(request.app.state, "explainers", {})
    metadata=getattr(request.app.state, "metadata", {})
    pipeline=models.get(model)
    explainer=explainers.get(model)
    if pipeline is None:
        pipeline=getattr(request.app.state, "pipeline", None)
    if explainer is None:
        explainer=getattr(request.app.state, "explainer", None)
    if pipeline is None or explainer is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model pipeline or SHAP explainer for '{model}' is currently unavailable."
        )
    customer_dict=payload.model_dump()
    cust_id=customer_dict.get("customerID", "CUST-DEFAULT")
    feature_names=metadata.get("encoded_feature_names", [])
    df=pd.DataFrame([customer_dict])
    try:
        explanation=explain_single_customer(
            pipeline=pipeline,
            explainer=explainer,
            customer_raw_df=df,
            feature_names=feature_names,
            top_k=5
        )
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"SHAP attribution computation failed for '{model}': {str(e)}"
        )
    prob=explanation["prediction_probability"]
    tier = "CRITICAL" if prob>=0.70 else ("MODERATE" if prob>=0.40 else "LOW")
    prescribed=prescribe_retention_action(
        churn_prob=prob,
        top_shap_drivers=explanation["risk_drivers"],
        customer_record=customer_dict
    )
    action_response=PrescribedActionResponse(
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
        model_used=model,
        churn_probability=prob,
        base_value=explanation["base_value"],
        risk_tier=tier,
        risk_drivers=explanation["risk_drivers"],
        protective_factors=explanation["protective_factors"],
        prescribed_action=action_response
    )
