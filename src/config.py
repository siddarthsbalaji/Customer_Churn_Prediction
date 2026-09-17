"""
Global Configuration & Constant Definitions for Customer Churn Decision Engine.
"""
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parent.parent
DATA_DIR=PROJECT_ROOT / "data"
RAW_DATA_DIR=DATA_DIR / "raw"
PROCESSED_DATA_DIR=DATA_DIR / "processed"
SAMPLE_UPLOADS_DIR=DATA_DIR / "sample_uploads"
MODELS_DIR=PROJECT_ROOT / "models"
PIPELINE_ARTIFACT_PATH=MODELS_DIR / "churn_pipeline.joblib"
SHAP_EXPLAINER_ARTIFACT_PATH=MODELS_DIR / "shap_explainer.joblib"
METADATA_ARTIFACT_PATH=MODELS_DIR / "model_metadata.json"
RF_PIPELINE_PATH=MODELS_DIR / "random_forest_pipeline.joblib"
LR_PIPELINE_PATH=MODELS_DIR / "logistic_regression_pipeline.joblib"
RF_EXPLAINER_PATH=MODELS_DIR / "random_forest_explainer.joblib"
LR_EXPLAINER_PATH=MODELS_DIR / "logistic_regression_explainer.joblib"
MODEL_REGISTRY={
    "random_forest": {
        "key": "random_forest",
        "display_name": "Random Forest (Champion Ensemble)",
        "pipeline_path": RF_PIPELINE_PATH,
        "explainer_path": RF_EXPLAINER_PATH,
        "default_threshold": 0.210,
        "description": "Non-linear ensemble capturing complex multi-service customer interactions."
    },
    "logistic_regression": {
        "key": "logistic_regression",
        "display_name": "Logistic Regression (Linear Baseline)",
        "pipeline_path": LR_PIPELINE_PATH,
        "explainer_path": LR_EXPLAINER_PATH,
        "default_threshold": 0.310,
        "description": "Interpretable linear model with transparent odds-ratio risk contributions."
    }
}
DEFAULT_MODEL_KEY = "random_forest"
ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"
NUMERIC_FEATURES=[
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]
CATEGORICAL_FEATURES=[
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]
ALL_FEATURES=NUMERIC_FEATURES+CATEGORICAL_FEATURES
DEFAULT_DECISION_THRESHOLD=0.45
RISK_TIERS={
    "CRITICAL": 0.70,   
    "MODERATE": 0.40,   
    "LOW": 0.00         
}
COLUMN_ALIASES={
    "customerID": ["customerid", "customer_id", "id", "account_id", "account_number"],
    "tenure": ["tenure", "tenure_months", "months_active", "customer_tenure"],
    "MonthlyCharges": ["monthlycharges", "monthly_charges", "monthly_spend", "monthly_bill", "monthly_fee", "mrr"],
    "TotalCharges": ["totalcharges", "total_charges", "total_spend", "cumulative_spend", "lifetime_bill"],
    "Contract": ["contract", "contract_type", "plan_type", "term"],
    "InternetService": ["internetservice", "internet_service", "connection_type", "broadband"],
    "PaymentMethod": ["paymentmethod", "payment_method", "billing_channel", "payment_type"],
    "TechSupport": ["techsupport", "tech_support", "support_tier"],
    "OnlineSecurity": ["onlinesecurity", "online_security", "security_package"],
    "OnlineBackup": ["onlinebackup", "online_backup", "cloud_backup"],
    "DeviceProtection": ["deviceprotection", "device_protection", "warranty"],
    "StreamingTV": ["streamingtv", "streaming_tv", "tv_service"],
    "StreamingMovies": ["streamingmovies", "streaming_movies", "movie_package"],
    "PaperlessBilling": ["paperlessbilling", "paperless_billing", "paperless"],
    "SeniorCitizen": ["seniorcitizen", "senior_citizen", "is_senior"],
    "Partner": ["partner", "has_partner"],
    "Dependents": ["dependents", "has_dependents"],
    "PhoneService": ["phoneservice", "phone_service", "has_phone"],
    "MultipleLines": ["multiplelines", "multiple_lines"]
}
