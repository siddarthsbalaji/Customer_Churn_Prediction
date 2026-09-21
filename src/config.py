"""
Global Configuration & Constant Definitions for Customer Churn Decision Engine.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT=Path(__file__).resolve().parent.parent
# Load .env file from project root
load_dotenv(PROJECT_ROOT / ".env")

# AI API Configurations
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
AI_PROVIDER = os.getenv("AI_PROVIDER", "gemini" if GEMINI_API_KEY else ("openai" if OPENAI_API_KEY else "gemini")).lower()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

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

# Calibrated Statistical Imputation Defaults (for missing custom metrics)
CANONICAL_STATISTICAL_DEFAULTS = {
    "tenure": 29.0,
    "MonthlyCharges": 64.76,
    "TotalCharges": 1397.47,
    "Contract": "Month-to-month",
    "InternetService": "DSL",
    "PaymentMethod": "Electronic check",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "PaperlessBilling": "Yes"
}

# Feature Taxonomy & Descriptions used by AI Classifier
FEATURE_SCHEMA_DESCRIPTIONS = {
    "customerID": {"role": "IDENTIFIER", "type": "string", "desc": "Unique customer / client / account identification code."},
    "tenure": {"role": "TENURE", "type": "numeric", "desc": "Number of months the customer has been active / subscribed."},
    "MonthlyCharges": {"role": "FINANCIAL_MONTHLY", "type": "numeric", "desc": "Monthly billing rate, recurring fee, or MRR."},
    "TotalCharges": {"role": "FINANCIAL_CUMULATIVE", "type": "numeric", "desc": "Total cumulative lifetime spend or billed balance."},
    "Contract": {"role": "CONTRACT_TYPE", "type": "categorical", "desc": "Contract duration or commitment terms (e.g. Month-to-month, One year, Two year)."},
    "PaymentMethod": {"role": "PAYMENT_METHOD", "type": "categorical", "desc": "Billing channel or payment method (Electronic check, Mailed check, Bank transfer, Credit card)."},
    "InternetService": {"role": "CORE_SERVICE", "type": "categorical", "desc": "Primary digital service or connection tier (DSL, Fiber optic, None)."},
    "TechSupport": {"role": "TECH_SUPPORT", "type": "categorical", "desc": "Dedicated premium technical support subscription (Yes, No)."},
    "OnlineSecurity": {"role": "ADDON_SERVICE", "type": "categorical", "desc": "Cybersecurity or antivirus add-on protection."},
    "OnlineBackup": {"role": "ADDON_SERVICE", "type": "categorical", "desc": "Cloud backup or storage add-on."},
    "DeviceProtection": {"role": "ADDON_SERVICE", "type": "categorical", "desc": "Hardware warranty or device coverage."},
    "StreamingTV": {"role": "ADDON_SERVICE", "type": "categorical", "desc": "Streaming television entertainment add-on."},
    "StreamingMovies": {"role": "ADDON_SERVICE", "type": "categorical", "desc": "Streaming movie package add-on."},
    "PhoneService": {"role": "ADDON_SERVICE", "type": "categorical", "desc": "Landline telephone service flag."},
    "MultipleLines": {"role": "ADDON_SERVICE", "type": "categorical", "desc": "Multiple telephone lines subscription."},
    "PaperlessBilling": {"role": "ADDON_SERVICE", "type": "categorical", "desc": "Electronic billing preference."},
    "gender": {"role": "DEMOGRAPHIC", "type": "categorical", "desc": "Customer gender (Male, Female)."},
    "SeniorCitizen": {"role": "DEMOGRAPHIC", "type": "numeric", "desc": "Senior citizen status flag (1, 0)."},
    "Partner": {"role": "DEMOGRAPHIC", "type": "categorical", "desc": "Whether customer has a spouse or partner (Yes, No)."},
    "Dependents": {"role": "DEMOGRAPHIC", "type": "categorical", "desc": "Whether customer has children or dependents (Yes, No)."},
    "Churn": {"role": "TARGET", "type": "binary", "desc": "Ground truth churn status (Yes, No, 1, 0)."}
}

