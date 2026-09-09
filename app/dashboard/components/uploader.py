"""
Dynamic CSV Uploader & Schema Validation Component for Streamlit.
"""
from typing import Optional
import pandas as pd
import streamlit as st

from src.config import SAMPLE_UPLOADS_DIR
from src.data.validator import validate_and_align_dataset


def render_uploader_component() -> Optional[pd.DataFrame]:
    """
    Renders dynamic file upload interface, schema verification alerts,
    and fallback sample dataset selector.
    """
    st.markdown("### 1. Ingest Customer Dataset")
    st.caption("Upload arbitrary customer CSV datasets or select a pre-staged sample to evaluate batch retention risk.")

    col1, col2 = st.columns([3, 2])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload Customer Records (.csv)",
            type=["csv"],
            help="Upload raw customer records. Column aliases will be automatically resolved."
        )

    with col2:
        st.markdown("**Or load a pre-staged sample dataset:**")
        sample_choice = st.selectbox(
            "Select sample fixture:",
            options=["None", "Clean Sample (valid_sample.csv)", "Messy Aliased (messy_aliased_sample.csv)"],
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
    elif sample_choice == "Clean Sample (valid_sample.csv)":
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

    # Execute Schema Alignment & Validation
    aligned_df, missing_cols, warnings = validate_and_align_dataset(raw_df)

    if missing_cols:
        st.error(
            f"🚫 **Schema Validation Failed!** The dataset is missing required mandatory columns: "
            f"`{missing_cols}`. Please verify your file or map columns accordingly."
        )
        return None

    # Render Validation Status
    st.success(f"✅ **Schema Verified:** {len(aligned_df)} customer accounts loaded and validated.")

    if warnings:
        with st.expander(f"⚠️ Column Resolution Notices ({len(warnings)})", expanded=False):
            for w in warnings:
                st.write(f"- {w}")

    with st.expander("👀 Preview Ingested Records (First 5 Rows)", expanded=False):
        st.dataframe(aligned_df.head(5), use_container_width=True)

    return aligned_df
