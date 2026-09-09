"""
Head-to-Head Multi-Model Comparison Endpoint.
"""
from fastapi import APIRouter, HTTPException, Request
import pandas as pd

from app.api.schemas import CustomerInput, ModelComparisonResponse, PredictionResponse
from src.explainability.shap_service import compare_customer_models

router = APIRouter(tags=["Model Comparison"])


@router.post("/compare", response_model=ModelComparisonResponse)
def compare_customer_models_route(payload: CustomerInput, request: Request):
    """
    Evaluates customer record across both Random Forest and Logistic Regression,
    providing head-to-head churn risk comparisons, probability deltas, and driver consensus.
    """
    models = getattr(request.app.state, "models", {})
    explainers = getattr(request.app.state, "explainers", {})
    metadata = getattr(request.app.state, "metadata", {})

    rf_pipe = models.get("random_forest")
    lr_pipe = models.get("logistic_regression")
    rf_exp = explainers.get("random_forest")
    lr_exp = explainers.get("logistic_regression")

    if rf_pipe is None or lr_pipe is None or rf_exp is None or lr_exp is None:
        raise HTTPException(
            status_code=503,
            detail="Dual model services are unavailable. Ensure both RF and LR pipelines are initialized."
        )

    customer_dict = payload.model_dump()
    cust_id = customer_dict.get("customerID", "CUST-DEFAULT")
    df = pd.DataFrame([customer_dict])

    rf_meta = metadata.get("models", {}).get("random_forest", {})
    lr_meta = metadata.get("models", {}).get("logistic_regression", {})
    thresholds = {
        "random_forest": rf_meta.get("optimal_threshold", 0.210),
        "logistic_regression": lr_meta.get("optimal_threshold", 0.310)
    }

    feature_names = metadata.get("encoded_feature_names", [])

    comparison = compare_customer_models(
        pipelines={"random_forest": rf_pipe, "logistic_regression": lr_pipe},
        explainers={"random_forest": rf_exp, "logistic_regression": lr_exp},
        customer_raw_df=df,
        feature_names=feature_names,
        thresholds=thresholds
    )

    rf_res = comparison["random_forest"]
    lr_res = comparison["logistic_regression"]

    return ModelComparisonResponse(
        customer_id=cust_id,
        random_forest=PredictionResponse(
            customer_id=cust_id,
            model_used="random_forest",
            churn_probability=rf_res["churn_probability"],
            is_at_risk=rf_res["is_at_risk"],
            risk_tier=rf_res["risk_tier"],
            decision_threshold_applied=thresholds["random_forest"]
        ),
        logistic_regression=PredictionResponse(
            customer_id=cust_id,
            model_used="logistic_regression",
            churn_probability=lr_res["churn_probability"],
            is_at_risk=lr_res["is_at_risk"],
            risk_tier=lr_res["risk_tier"],
            decision_threshold_applied=thresholds["logistic_regression"]
        ),
        probability_delta=comparison["probability_delta"],
        tier_agreement=comparison["tier_agreement"],
        agreed_risk_drivers=comparison["agreed_risk_drivers"]
    )
