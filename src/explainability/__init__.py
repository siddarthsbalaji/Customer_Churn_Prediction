"""
Explainability package for SHAP attribution and diagnostics.
"""
from src.explainability.shap_service import (
    build_and_save_shap_explainer,
    load_shap_explainer,
    explain_single_customer,
    render_customer_waterfall_figure,
    humanize_feature_name,
)

__all__ = [
    "build_and_save_shap_explainer",
    "load_shap_explainer",
    "explain_single_customer",
    "render_customer_waterfall_figure",
    "humanize_feature_name",
]
