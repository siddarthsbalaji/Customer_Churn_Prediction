"""
Dynamic CSV Uploader & AI Schema Validation Component for Streamlit.
Supports arbitrary datasets with custom metrics, AI classification via Gemini/OpenAI,
and automatic alignment to the churn decision engine.
"""
from typing import Dict, Optional, Tuple
import pandas as pd
import streamlit as st

from src.ai.metric_classifier import classify_dataset
from src.config import (
    ALL_FEATURES,
    GEMINI_API_KEY,
    OPENAI_API_KEY,
    SAMPLE_UPLOADS_DIR,
)
from src.data.adapter import adapt_custom_dataset


def render_uploader_component() -> Optional[pd.DataFrame]:
    """
    Renders dynamic file upload interface, AI metric classification review,
    and pre-staged sample dataset selector.
    """
    st.markdown("### 1. Ingest Customer Dataset")
    st.caption(
        "Upload arbitrary customer CSV datasets or select a sample. "
        "Our AI-powered schema classifier will automatically detect metric roles (e.g. nationality, MRR) "
        "and align them to the retention decision engine."
    )

    # Detect API Key Status
    session_key = st.session_state.get("custom_api_key", "").strip()
    active_key = session_key or GEMINI_API_KEY or OPENAI_API_KEY
    if active_key:
        provider_label = "Gemini" if (session_key or GEMINI_API_KEY) else "OpenAI"
        st.markdown(
            f"<div style='background-color: rgba(34, 197, 94, 0.1); border: 1px solid #22c55e; "
            f"border-radius: 8px; padding: 8px 14px; margin-bottom: 12px; font-size: 0.88rem; color: #166534;'>"
            f"✨ <strong>AI Metric Classifier Active:</strong> Using {provider_label} API key configured in <code>.env</code>."
            f"</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div style='background-color: rgba(245, 158, 11, 0.1); border: 1px solid #f59e0b; "
            "border-radius: 8px; padding: 8px 14px; margin-bottom: 12px; font-size: 0.88rem; color: #92400e;'>"
            "ℹ️ <strong>Semantic Heuristic Mode Active:</strong> Add <code>GEMINI_API_KEY</code> or "
            "<code>OPENAI_API_KEY</code> to <code>.env</code> for full AI metric classification."
            "</div>",
            unsafe_allow_html=True
        )

    col1, col2 = st.columns([3, 2])
    with col1:
        uploaded_file = st.file_uploader(
            "Upload Custom Customer Records (.csv)",
            type=["csv"],
            help="Upload raw customer records. Custom headers like nationality or MRR will be classified automatically."
        )

    with col2:
        st.markdown("**Or load a sample dataset fixture:**")
        sample_choice = st.selectbox(
            "Select sample fixture:",
            options=[
                "None",
                "Custom Banking with Nationality (custom_banking_churn.csv)",
                "Custom SaaS with MRR & Country (custom_saas_churn.csv)",
                "Standard Clean Sample (valid_sample.csv)",
                "Messy Aliased (messy_aliased_sample.csv)",
            ],
            index=1 if uploaded_file is None else 0
        )

    raw_df: Optional[pd.DataFrame] = None
    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
            st.toast(f"Uploaded file loaded: {raw_df.shape[0]} rows × {raw_df.shape[1]} columns", icon="📂")
        except Exception as e:
            st.error(f"Error parsing CSV file: {e}")
            return None
    elif sample_choice == "Custom Banking with Nationality (custom_banking_churn.csv)":
        banking_path = SAMPLE_UPLOADS_DIR / "custom_banking_churn.csv"
        if banking_path.exists():
            raw_df = pd.read_csv(banking_path)
    elif sample_choice == "Custom SaaS with MRR & Country (custom_saas_churn.csv)":
        saas_path = SAMPLE_UPLOADS_DIR / "custom_saas_churn.csv"
        if saas_path.exists():
            raw_df = pd.read_csv(saas_path)
    elif sample_choice == "Standard Clean Sample (valid_sample.csv)":
        clean_path = SAMPLE_UPLOADS_DIR / "valid_sample.csv"
        if clean_path.exists():
            raw_df = pd.read_csv(clean_path)
    elif sample_choice == "Messy Aliased (messy_aliased_sample.csv)":
        messy_path = SAMPLE_UPLOADS_DIR / "messy_aliased_sample.csv"
        if messy_path.exists():
            raw_df = pd.read_csv(messy_path)

    if raw_df is None:
        st.info("Awaiting customer data. Please upload a CSV file or select a sample dataset above.")
        return None

    # Run AI / Heuristic Metric Classification
    with st.spinner("🤖 Classifying dataset metrics and analyzing schema semantics..."):
        classification_result = classify_dataset(raw_df, api_key=active_key if active_key else None)

    # Interactive AI Classification & Mapping Review Expander
    with st.expander(
        f"🤖 AI Metric Classification & Mapping Review ({len(raw_df.columns)} Columns)",
        expanded=(uploaded_file is not None or "Custom" in sample_choice)
    ):
        st.markdown(
            f"**Classifier Engine:** `{classification_result.provider_used.upper()}` — "
            f"*{classification_result.status_message}*"
        )

        mapping_rows = []
        for col_name, item in classification_result.metrics.items():
            conf_pct = f"{int(item.confidence * 100)}%"
            mapping_target = item.canonical_mapping if item.canonical_mapping else "[Custom Attribute]"
            mapping_rows.append({
                "Metric Title": col_name,
                "Semantic Role": item.role,
                "Model Mapping": mapping_target,
                "Confidence": conf_pct,
                "Reasoning": item.reasoning
            })

        st.dataframe(pd.DataFrame(mapping_rows), use_container_width=True, hide_index=True)

        if classification_result.custom_attributes:
            st.markdown(
                f"💡 **Preserved Custom Attributes:** "
                + ", ".join([f"`{c}`" for c in classification_result.custom_attributes])
                + " *(Retained for customer diagnostics & priority view)*"
            )

    # Adapt dataset using classification results + statistical defaults
    aligned_df, adapter_meta = adapt_custom_dataset(raw_df, classification_result)

    # Display Verification Success & Notices
    st.success(
        f"✅ **Dataset Aligned & Verified:** {len(aligned_df)} customer accounts ready for multi-model inference."
    )

    if adapter_meta["warnings"]:
        with st.expander(f"⚠️ Imputation & Mapping Notices ({len(adapter_meta['warnings'])})", expanded=False):
            for w in adapter_meta["warnings"]:
                st.write(f"- {w}")

    with st.expander("👀 Preview Ingested Records (First 5 Rows)", expanded=False):
        st.dataframe(aligned_df.head(5), use_container_width=True)

    return aligned_df
