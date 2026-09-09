"""
Executive Customer Churn Decision Engine — Master Streamlit Application.
"""
import sys
from pathlib import Path

# Add project root to sys.path to enable clean absolute imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from app.dashboard.components.diagnostics import render_diagnostics_component
from app.dashboard.components.priority_matrix import render_priority_matrix
from app.dashboard.components.simulator import render_simulator_component
from app.dashboard.components.uploader import render_uploader_component
from app.dashboard.styles import apply_custom_styles
from src.config import (
    METADATA_ARTIFACT_PATH,
    PIPELINE_ARTIFACT_PATH,
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
    """Cached loader for model pipeline, metadata, and SHAP explainer."""
    if not PIPELINE_ARTIFACT_PATH.exists() or not METADATA_ARTIFACT_PATH.exists():
        st.error("Model artifacts not found. Please train models before launching dashboard.")
        st.stop()

    pipeline, metadata = load_pipeline_artifacts(
        PIPELINE_ARTIFACT_PATH, METADATA_ARTIFACT_PATH
    )

    if not SHAP_EXPLAINER_ARTIFACT_PATH.exists():
        st.error("SHAP explainer artifact not found. Please run SHAP service initialization.")
        st.stop()

    explainer = load_shap_explainer(SHAP_EXPLAINER_ARTIFACT_PATH)
    return pipeline, metadata, explainer


def main():
    # Load Models and Explainer
    pipeline, metadata, explainer = load_application_resources()

    # Executive Header
    st.markdown("""
    <div class="dashboard-header">
        <div class="dashboard-title">🎯 Customer Churn Decision Engine</div>
        <div class="dashboard-subtitle">
            Enterprise Retention Intelligence: Dynamic CSV Uploads, SHAP Root-Cause Diagnostics & Prescriptive Interventions
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar Information
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=64)
        st.markdown("### Operational Intelligence")
        st.markdown(f"**Champion Model:** `{metadata.get('model_name', 'RandomForest')}`")
        st.markdown(f"**Calibrated Decision Threshold:** `τ* = {metadata.get('optimal_threshold', 0.210)}`")
        st.markdown(f"**Engineered Features:** `{metadata.get('encoded_feature_count', 56)}`")
        st.markdown(f"**Training Sample Size:** `{metadata.get('training_records', 5634):,} rows`")

        st.markdown("---")
        st.markdown("### Workflow Navigation")
        st.markdown("""
        1. **Ingest Dataset:** Upload arbitrary customer CSV.
        2. **Executive Priority Matrix:** Review ranked accounts & MRR risk.
        3. **Root-Cause Diagnostics:** Audit local SHAP waterfall drivers.
        4. **Retention Sandbox:** Test counterfactual 'What-If' scenarios.
        """)

        st.markdown("---")
        st.caption("Powered by Scikit-Learn, SHAP & FastAPI.")

    # 1. Dynamic File Upload & Schema Validation
    aligned_df = render_uploader_component()

    if aligned_df is None or len(aligned_df) == 0:
        return

    st.markdown("---")

    # 2. Executive Priority Matrix
    ranked_df, selected_customer_id = render_priority_matrix(pipeline, aligned_df, metadata)

    if selected_customer_id is None or selected_customer_id not in ranked_df["customerID"].values:
        selected_customer_id = ranked_df["customerID"].iloc[0]

    # Extract single customer record for deep-dive
    customer_sub_df = ranked_df[ranked_df["customerID"] == selected_customer_id].copy()
    customer_series = customer_sub_df.iloc[0]

    st.markdown("---")

    # Tabbed Interface for Diagnostics and What-If Simulator
    tab_diagnostics, tab_simulator = st.tabs([
        "🔬 Individual Customer Diagnostics (SHAP Waterfall)",
        "🧪 'What-If' Counterfactual Retention Sandbox"
    ])

    with tab_diagnostics:
        render_diagnostics_component(
            pipeline=pipeline,
            explainer=explainer,
            metadata=metadata,
            customer_df=customer_sub_df,
            customer_id=selected_customer_id
        )

    with tab_simulator:
        render_simulator_component(
            pipeline=pipeline,
            baseline_series=customer_series,
            customer_id=selected_customer_id
        )


if __name__ == "__main__":
    main()
