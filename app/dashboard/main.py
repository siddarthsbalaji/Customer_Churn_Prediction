"""
Executive Customer Churn Decision Engine — Master Streamlit Application.
Features dynamic model selection (Random Forest vs. Logistic Regression)
and side-by-side comparative diagnostics.
"""
import sys
from pathlib import Path

# Add project root to sys.path to enable clean absolute imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Invalidate cached src and dashboard modules so hot-reloading never uses stale definitions
for mod in list(sys.modules.keys()):
    if mod.startswith("src.") or mod.startswith("app.dashboard.components"):
        del sys.modules[mod]

import streamlit as st
from app.dashboard.components.comparison import render_comparison_component
from app.dashboard.components.diagnostics import render_diagnostics_component
from app.dashboard.components.priority_matrix import render_priority_matrix
from app.dashboard.components.simulator import render_simulator_component
from app.dashboard.components.uploader import render_uploader_component
from app.dashboard.styles import apply_custom_styles
from src.config import (
    LR_EXPLAINER_PATH,
    LR_PIPELINE_PATH,
    METADATA_ARTIFACT_PATH,
    PIPELINE_ARTIFACT_PATH,
    RF_EXPLAINER_PATH,
    RF_PIPELINE_PATH,
    SHAP_EXPLAINER_ARTIFACT_PATH,
)
from src.explainability.shap_service import load_shap_explainer
from src.models.pipeline import load_pipeline_artifacts

# Configure Page
st.set_page_config(
    page_title="Customer Churn Decision Engine",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Modern Executive Theme
apply_custom_styles()


@st.cache_resource
def load_application_resources():
    """Cached loader for dual model pipelines, metadata, and SHAP explainers."""
    if not METADATA_ARTIFACT_PATH.exists():
        st.error("Model metadata artifact not found. Please run training pipeline first.")
        st.stop()

    # Load Random Forest
    if RF_PIPELINE_PATH.exists():
        rf_pipeline, metadata = load_pipeline_artifacts(RF_PIPELINE_PATH, METADATA_ARTIFACT_PATH)
    else:
        rf_pipeline, metadata = load_pipeline_artifacts(PIPELINE_ARTIFACT_PATH, METADATA_ARTIFACT_PATH)

    # Load Logistic Regression
    if LR_PIPELINE_PATH.exists():
        lr_pipeline, _ = load_pipeline_artifacts(LR_PIPELINE_PATH, METADATA_ARTIFACT_PATH)
    else:
        lr_pipeline = rf_pipeline

    # Load Explainers
    rf_explainer = load_shap_explainer(RF_EXPLAINER_PATH if RF_EXPLAINER_PATH.exists() else SHAP_EXPLAINER_ARTIFACT_PATH)
    lr_explainer = load_shap_explainer(LR_EXPLAINER_PATH) if LR_EXPLAINER_PATH.exists() else rf_explainer

    pipelines = {
        "random_forest": rf_pipeline,
        "logistic_regression": lr_pipeline
    }
    explainers = {
        "random_forest": rf_explainer,
        "logistic_regression": lr_explainer
    }

    return pipelines, metadata, explainers


def main():
    # Load Models and Explainers
    pipelines, metadata, explainers = load_application_resources()

    # Executive Header
    st.markdown("""
    <div class="dashboard-header">
        <div class="dashboard-title">🎯 Customer Churn Decision Engine</div>
        <div class="dashboard-subtitle">
            Enterprise Retention Intelligence: Multi-Model Evaluation, SHAP Diagnostics & Prescriptive Interventions
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar: Model Selection & Intelligence
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=64)
        st.markdown("### 🤖 Active ML Model")

        active_model_key = st.radio(
            "Select inference model:",
            options=["random_forest", "logistic_regression"],
            format_func=lambda k: (
                "🌲 Random Forest (Champion Ensemble)"
                if k == "random_forest"
                else "📈 Logistic Regression (Linear Baseline)"
            ),
            help="Switch active model to compare risk scores, thresholds, and SHAP drivers."
        )

        active_pipeline = pipelines[active_model_key]
        active_explainer = explainers[active_model_key]
        alt_model_key = "logistic_regression" if active_model_key == "random_forest" else "random_forest"
        alt_pipeline = pipelines[alt_model_key]

        model_display_name = "Random Forest" if active_model_key == "random_forest" else "Logistic Regression"
        alt_display_name = "Logistic Regression" if active_model_key == "random_forest" else "Random Forest"

        model_meta = metadata.get("models", {}).get(active_model_key, {})
        default_thresh = 0.210 if active_model_key == "random_forest" else 0.310
        active_threshold = model_meta.get("optimal_threshold", default_thresh)

        st.markdown("---")
        st.markdown(f"**Model Type:** `{model_meta.get('model_type', model_display_name)}`")
        st.markdown(f"**Calibrated Threshold:** `τ* = {active_threshold:.3f}`")
        if "metrics_calibrated" in model_meta:
            recall = model_meta["metrics_calibrated"].get("recall", 0.90)
            f2 = model_meta["metrics_calibrated"].get("f2", 0.74)
            st.markdown(f"**Test Recall (τ*):** `{recall*100:.1f}%`")
            st.markdown(f"**Test F2 Score:** `{f2:.3f}`")

        st.markdown("---")
        st.markdown("### Workflow Navigation")
        st.markdown("""
        1. **Ingest Dataset:** Upload arbitrary customer CSV.
        2. **Priority Matrix:** Filter accounts by active model.
        3. **Diagnostics & What-If:** Audit SHAP drivers & simulate.
        4. **Head-to-Head Comparison:** Compare RF vs. LR divergence.
        """)

        st.markdown("---")
        st.caption("Powered by Scikit-Learn, SHAP & FastAPI.")

    # 1. Dynamic File Upload & Schema Validation
    aligned_df = render_uploader_component()

    if aligned_df is None or len(aligned_df) == 0:
        return

    st.markdown("---")

    # 2. Executive Priority Matrix (Scored with active model)
    ranked_df, selected_customer_id = render_priority_matrix(
        pipeline=active_pipeline,
        df=aligned_df,
        metadata=metadata,
        model_name=model_display_name,
        threshold=active_threshold
    )

    if selected_customer_id is None or selected_customer_id not in ranked_df["customerID"].values:
        selected_customer_id = ranked_df["customerID"].iloc[0]

    # Extract single customer record for deep-dive
    customer_sub_df = ranked_df[ranked_df["customerID"] == selected_customer_id].copy()
    customer_series = customer_sub_df.iloc[0]

    st.markdown("---")

    # Tabbed Interface for Diagnostics, What-If Simulator, and Model Comparison
    tab_diagnostics, tab_simulator, tab_comparison = st.tabs([
        f"🔬 Individual Customer Diagnostics ({model_display_name})",
        "🧪 'What-If' Counterfactual Retention Sandbox",
        "⚖️ Dual-Model Head-to-Head Comparison (RF vs. LR)"
    ])

    with tab_diagnostics:
        render_diagnostics_component(
            pipeline=active_pipeline,
            explainer=active_explainer,
            metadata=metadata,
            customer_df=customer_sub_df,
            customer_id=selected_customer_id,
            model_name=model_display_name
        )

    with tab_simulator:
        render_simulator_component(
            pipeline=active_pipeline,
            baseline_series=customer_series,
            customer_id=selected_customer_id,
            model_name=model_display_name,
            alt_pipeline=alt_pipeline,
            alt_model_name=alt_display_name
        )

    with tab_comparison:
        render_comparison_component(
            pipelines=pipelines,
            explainers=explainers,
            metadata=metadata,
            df=aligned_df,
            selected_customer_id=selected_customer_id
        )


if __name__ == "__main__":
    main()
