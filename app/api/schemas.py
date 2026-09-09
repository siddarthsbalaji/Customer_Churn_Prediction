"""
Pydantic v2 Request & Response Schemas for Customer Churn Decision Engine API.
"""
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class CustomerInput(BaseModel):
    customerID: Optional[str] = Field(default="CUST-0001", description="Unique Customer Identifier")
    gender: str = Field(default="Female", description="Customer Gender (Male/Female)")
    SeniorCitizen: int = Field(default=0, ge=0, le=1, description="1 if Senior Citizen, else 0")
    Partner: str = Field(default="No", description="Yes/No")
    Dependents: str = Field(default="No", description="Yes/No")
    tenure: int = Field(default=1, ge=0, description="Customer tenure in months")
    PhoneService: str = Field(default="Yes", description="Yes/No")
    MultipleLines: str = Field(default="No", description="Yes/No/No phone service")
    InternetService: str = Field(default="DSL", description="DSL/Fiber optic/No")
    OnlineSecurity: str = Field(default="No", description="Yes/No/No internet service")
    OnlineBackup: str = Field(default="No", description="Yes/No/No internet service")
    DeviceProtection: str = Field(default="No", description="Yes/No/No internet service")
    TechSupport: str = Field(default="No", description="Yes/No/No internet service")
    StreamingTV: str = Field(default="No", description="Yes/No/No internet service")
    StreamingMovies: str = Field(default="No", description="Yes/No/No internet service")
    Contract: str = Field(default="Month-to-month", description="Month-to-month/One year/Two year")
    PaperlessBilling: str = Field(default="Yes", description="Yes/No")
    PaymentMethod: str = Field(
        default="Electronic check",
        description="Electronic check/Mailed check/Bank transfer (automatic)/Credit card (automatic)"
    )
    MonthlyCharges: float = Field(default=65.0, ge=0.0, description="Monthly recurring charge")
    TotalCharges: Union[float, str] = Field(default=65.0, description="Total cumulative charge or numeric string")

    model_config = {
        "json_schema_extra": {
            "example": {
                "customerID": "7590-VHVEG",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85
            }
        }
    }


class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    is_at_risk: bool
    risk_tier: str
    decision_threshold_applied: float


class DriverContribution(BaseModel):
    feature: str
    display_name: str
    shap_value: float
    feature_value: Optional[float] = None


class PrescribedActionResponse(BaseModel):
    action_code: str
    action_title: str
    priority: str
    incentive_type: str
    estimated_cost: str
    recommended_channel: str
    playbook_details: str
    trigger_rationale: str


class ExplanationResponse(BaseModel):
    customer_id: str
    churn_probability: float
    base_value: float
    risk_tier: str
    risk_drivers: List[DriverContribution]
    protective_factors: List[DriverContribution]
    prescribed_action: PrescribedActionResponse


class BatchCustomerResult(BaseModel):
    customer_id: str
    churn_probability: float
    is_at_risk: bool
    risk_tier: str
    primary_risk_driver: str
    prescribed_action_title: str
    monthly_charges: float


class BatchSummaryResponse(BaseModel):
    total_records: int
    at_risk_count: int
    critical_risk_count: int
    total_mrr_at_risk: float
    warnings: List[str]
    results: List[BatchCustomerResult]


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    model_loaded: bool
    explainer_loaded: bool
