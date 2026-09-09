"""
Dual-Model Head-to-Head Comparison Component for Streamlit.
Enables side-by-side evaluation of Random Forest vs. Logistic Regression
for individual customer accounts and across the batch dataset.
"""
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline

from src.explainability.shap_service import (
    compare_customer_models,
    explain_single_customer,
)


def render_comparison_component(
    pipelines: Dict[str, Pipeline],
    explainers: Dict[str, Any],
    metadata: dict,
    df: pd.DataFrame,
    selected_customer_id: str
):
    """
    Renders side-by-side model comparison:
    1. Single Customer Head-to-Head (Prediction delta, tier concordance, driver overlap)
    2. Batch Population Concordance (Prediction correlation, tier disagreement, MRR delta)
    """
    st.markdown("### ⚖️ Head-to-Head Model Comparison")
    st.caption(
        "Evaluate prediction variance, tier concordance, and root-cause driver divergence "
        "between **Random Forest (Champion Ensemble)** and **Logistic Regression (Linear Baseline)**."
    )

    feature_names = metadata.get("encoded_feature_names", [])
    rf_meta = metadata.get("models", {}).get("random_forest", {})
    lr_meta = metadata.get("models", {}).get("logistic_regression", {})

    thresholds = {
        "random_forest": rf_meta.get("optimal_threshold", 0.210),
        "logistic_regression": lr_meta.get("optimal_threshold", 0.310)
    }

    # -------------------------------------------------------------
    # SECTION 1: Single Customer Head-to-Head
    # -------------------------------------------------------------
    st.markdown(f"#### 1. Account `{selected_customer_id}`: Cross-Model Diagnosis")

    cust_row = df[df["customerID"] == selected_customer_id]
    if cust_row.empty:
        cust_row = df.iloc[[0]]

    comparison = compare_customer_models(
        pipelines=pipelines,
        explainers=explainers,
        customer_raw_df=cust_row,
        feature_names=feature_names,
        thresholds=thresholds
    )

    rf_res = comparison["random_forest"]
    lr_res = comparison["logistic_regression"]
    delta_p = comparison["probability_delta"]

    # Comparative Metrics Columns
    col_rf, col_delta, col_lr = st.columns([3, 2, 3])

    with col_rf:
        rf_tier = rf_res["risk_tier"]
        rf_badge = f"badge-{rf_tier.lower()}"
        st.markdown(f"""
        <div class="metric-card" style="border-left: 4px solid #10b981;">
            <div class="metric-label">Random Forest (Ensemble)</div>
            <div class="metric-value">{rf_res['churn_probability']*100:.1f}%</div>
            <p><span class="{rf_badge}">{rf_tier} RISK</span> (Threshold: {thresholds['random_forest']:.3f})</p>
        </div>
        """, unsafe_allow_html=True)

    with col_delta:
        st.markdown(f"""
        <div class="metric-card" style="text-align: center;">
            <div class="metric-label">Probability Variance (ΔP)</div>
            <div class="metric-value" style="color: {'#e11d48' if abs(delta_p) > 0.1 else '#2563eb'};">
                {delta_p * 100:+.1f}%
            </div>
            <p><b>Tier Match:</b> {'✅ Identical' if comparison['tier_agreement'] else '⚠️ Divergent'}</p>
        </div>
        """, unsafe_allow_html=True)

    with col_lr:
        lr_tier = lr_res["risk_tier"]
        lr_badge = f"badge-{lr_tier.lower()}"
        st.markdown(f"""
        <div class="metric-card" style="border-left: 4px solid #3b82f6;">
            <div class="metric-label">Logistic Regression (Linear)</div>
            <div class="metric-value">{lr_res['churn_probability']*100:.1f}%</div>
            <p><span class="{lr_badge}">{lr_tier} RISK</span> (Threshold: {thresholds['logistic_regression']:.3f})</p>
        </div>
        """, unsafe_allow_html=True)

    # Top Drivers Comparison
    st.markdown("##### 🔍 Top Root-Cause Drivers Comparison")
    dcol1, dcol2 = st.columns(2)

    with dcol1:
        st.markdown("**Random Forest Key Drivers (Non-linear Splits):**")
        for d in rf_res["top_drivers"]:
            st.markdown(f"- **{d['display_name']}**: `+{d['shap_value']:.4f}`")

    with dcol2:
        st.markdown("**Logistic Regression Key Drivers (Linear Log-Odds):**")
        for d in lr_res["top_drivers"]:
            st.markdown(f"- **{d['display_name']}**: `+{d['shap_value']:.4f}`")

    if comparison["agreed_risk_drivers"]:
        st.success(f"🤝 **Consensus Drivers Across Both Models:** {', '.join(comparison['agreed_risk_drivers'])}")

    st.markdown("---")

    # -------------------------------------------------------------
    # SECTION 2: Batch Population Concordance & Disagreements
    # -------------------------------------------------------------
    st.markdown("#### 2. Batch Population Concordance Analysis")
    st.caption("Evaluate overall risk population agreement across all uploaded customer accounts.")

    rf_pipe = pipelines["random_forest"]
    lr_pipe = pipelines["logistic_regression"]

    rf_batch_probs = rf_pipe.predict_proba(df)[:, 1]
    lr_batch_probs = lr_pipe.predict_proba(df)[:, 1]

    batch_comp_df = df[["customerID", "MonthlyCharges", "tenure", "Contract"]].copy()
    batch_comp_df["rf_prob"] = rf_batch_probs.round(4)
    batch_comp_df["lr_prob"] = lr_batch_probs.round(4)
    batch_comp_df["delta_prob"] = (rf_batch_probs - lr_batch_probs).round(4)

    rf_thresh = thresholds["random_forest"]
    lr_thresh = thresholds["logistic_regression"]

    def assign_tier(p: float) -> str:
        if p >= 0.70:
            return "CRITICAL"
        elif p >= 0.40:
            return "MODERATE"
        return "LOW"

    batch_comp_df["rf_tier"] = batch_comp_df["rf_prob"].apply(assign_tier)
    batch_comp_df["lr_tier"] = batch_comp_df["lr_prob"].apply(assign_tier)
    batch_comp_df["tier_match"] = batch_comp_df["rf_tier"] == batch_comp_df["lr_tier"]

    # KPI Summary Cards for Batch
    rf_at_risk = int((batch_comp_df["rf_prob"] >= rf_thresh).sum())
    lr_at_risk = int((batch_comp_df["lr_prob"] >= lr_thresh).sum())
    disagreed_tiers = int((~batch_comp_df["tier_match"]).sum())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("RF At-Risk Accounts", f"{rf_at_risk:,}", f"{(rf_at_risk/len(df))*100:.1f}%")
    k2.metric("LR At-Risk Accounts", f"{lr_at_risk:,}", f"{(lr_at_risk/len(df))*100:.1f}%")
    k3.metric("Tier Disagreements", f"{disagreed_tiers:,}", f"{(disagreed_tiers/len(df))*100:.1f}% divergence")
    
    correlation = float(np.corrcoef(rf_batch_probs, lr_batch_probs)[0, 1])
    k4.metric("Pearson Model Correlation", f"{correlation:.3f}", "R-value")

    # Table of Largest Model Disagreements
    with st.expander("⚠️ View Accounts with Largest Model Disagreements (|ΔP| > 15%)", expanded=False):
        divergent_df = batch_comp_df.sort_values(by="delta_prob", key=abs, ascending=False).head(15)
        st.dataframe(
            divergent_df.rename(columns={
                "customerID": "Account ID",
                "MonthlyCharges": "MRR ($)",
                "tenure": "Tenure (Mo)",
                "Contract": "Contract",
                "rf_prob": "Random Forest Risk",
                "lr_prob": "Logistic Regression Risk",
                "delta_prob": "Variance (RF - LR)",
                "rf_tier": "RF Tier",
                "lr_tier": "LR Tier",
                "tier_match": "Tiers Match"
            }),
            use_container_width=True
        )
