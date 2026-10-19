"""
Individual Customer Root-Cause Diagnostics & SHAP Waterfall Rendering Component.
"""
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline
from src.ai.metric_classifier import generate_retention_outreach
from src.decision_engine.rules import prescribe_retention_action
from src.explainability.shap_service import (
    explain_single_customer,
    render_customer_waterfall_figure,
    render_customer_waterfall_plotly,
)
def render_diagnostics_component(
    pipeline: Pipeline,
    explainer: Any,
    metadata: dict,
    customer_df: pd.DataFrame,
    customer_id: str,
    model_name: str = "Random Forest"
):
    """
    Renders individual customer diagnostics:
    1. Account overview KPI metric cards
    2. Interactive SHAP Waterfall plot (Plotly / Matplotlib)
    3. Prescriptive Next-Best-Action retention playbook card
    4. AI-Generated Personalized Retention Outreach Copy
    """
    st.markdown(f"### 3. Root-Cause Diagnostics: Account `{customer_id}` (`{model_name}`)")
    st.caption(f"Local SHAP explainability audit evaluating feature attributions from **{model_name}**.")
    feature_names=metadata.get("encoded_feature_names", [])
    model_key = "xgboost" if "xgboost" in model_name.lower() else "random_forest"
    default_thresh = 0.210 if model_key == "random_forest" else 0.240
    threshold = metadata.get("models", {}).get(model_key, {}).get("optimal_threshold", default_thresh)
    explanation=explain_single_customer(
        pipeline=pipeline,
        explainer=explainer,
        customer_raw_df=customer_df,
        feature_names=feature_names,
        top_k=5
    )
    prob=explanation["prediction_probability"]
    tier = "CRITICAL" if prob>=0.70 else ("MODERATE" if prob>=threshold else "LOW")
    badge_class = f"badge-{tier.lower()}"
    c1, c2, c3, c4=st.columns(4)
    with c1:
        st.markdown(f"**Risk Evaluation:**<br><span class='{badge_class}'>{tier} RISK ({prob*100:.1f}%)</span>", unsafe_allow_html=True)
    with c2:
        mrr=float(customer_df.iloc[0].get("MonthlyCharges", 0.0))
        st.metric("Monthly Recurring Spend", f"${mrr:.2f}")
    with c3:
        tenure=int(customer_df.iloc[0].get("tenure", 0))
        st.metric("Tenure Active", f"{tenure} Months")
    with c4:
        contract=str(customer_df.iloc[0].get("Contract", "Month-to-month"))
        st.metric("Contract Type", contract)
    st.markdown("---")
    col_chart, col_playbook=st.columns([3, 2])
    with col_chart:
        st.markdown("#### 🔍 SHAP Local Attribution Waterfall")
        st.caption("Red bars push risk higher toward churn; blue bars push risk lower toward retention.")
        chart_mode = st.radio(
            "Visualization engine:",
            options=["Interactive (Plotly)", "Static Publication (Matplotlib)"],
            horizontal=True,
            key=f"chart_mode_{customer_id}"
        )
        if "Interactive" in chart_mode:
            fig_plotly = render_customer_waterfall_plotly(
                explanation_dict=explanation,
                customer_id=customer_id,
                model_name=model_name,
                max_display=8
            )
            st.plotly_chart(fig_plotly, use_container_width=True)
        else:
            with st.spinner("Generating SHAP waterfall plot..."):
                fig=render_customer_waterfall_figure(
                    pipeline=pipeline,
                    explainer=explainer,
                    customer_raw_df=customer_df,
                    feature_names=feature_names,
                    max_display=8,
                    customer_id=customer_id
                )
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
    with col_playbook:
        st.markdown("#### 🎯 Prescriptive Retention Playbook")
        st.caption("Automated Next-Best-Action tailored to this customer's acute risk drivers.")
        action=prescribe_retention_action(
            churn_prob=prob,
            top_shap_drivers=explanation["risk_drivers"],
            customer_record=customer_df.iloc[0].to_dict(),
            decision_threshold=threshold
        )
        st.markdown(f"""
        <div class="playbook-card">
            <div class="playbook-title">{action['action_title']}</div>
            <div class="playbook-meta">
                <b>Priority:</b> <span class="{badge_class}">{action['priority']}</span> &nbsp;|&nbsp; 
                <b>Est. Cost:</b> {action['estimated_cost']} &nbsp;|&nbsp; 
                <b>Incentive:</b> {action['incentive_type']}
            </div>
            <p><b>Recommended Channel:</b><br>{action['recommended_channel']}</p>
            <p><b>Execution Plan:</b><br>{action['playbook_details']}</p>
            <p><b>Trigger Rationale:</b><br><i>{action['trigger_rationale']}</i></p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("##### ✉️ Retention Outreach Automation")
        if st.button("✨ Draft Personalized Retention Email", key=f"draft_btn_{customer_id}"):
            with st.spinner("Generating customer retention outreach copy..."):
                outreach_text = generate_retention_outreach(
                    customer_id=customer_id,
                    customer_record=customer_df.iloc[0].to_dict(),
                    churn_prob=prob,
                    risk_drivers=explanation["risk_drivers"],
                    prescribed_action=action
                )
                st.text_area(
                    "Generated Outreach Copy (Ready for CRM / Email):",
                    value=outreach_text,
                    height=240,
                    key=f"outreach_text_{customer_id}"
                )
