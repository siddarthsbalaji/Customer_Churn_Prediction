"""
Interactive "What-If" Counterfactual Retention Simulator Component.
Supports single-model simulation and simultaneous dual-model evaluation.
"""
from typing import Optional
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline
def render_simulator_component(
    pipeline: Pipeline,
    baseline_series: pd.Series,
    customer_id: str,
    model_name: str = "Random Forest",
    alt_pipeline: Optional[Pipeline]=None,
    alt_model_name: str = "Logistic Regression"
):
    """
    Real-time counterfactual simulation sandbox enabling retention managers
    to test pricing discounts, contract commitments, and service add-ons.
    """
    st.markdown(f"### 4. Interactive 'What-If' Retention Simulator: Account `{customer_id}`")
    st.caption(
        f"Simulate adjustments and observe real-time churn risk reductions using **{model_name}**."
    )
    if "sim_active_id" not in st.session_state or st.session_state.sim_active_id!=customer_id:
        st.session_state.sim_active_id=customer_id
        st.session_state.sim_contract=str(baseline_series.get("Contract", "Month-to-month"))
        st.session_state.sim_payment=str(baseline_series.get("PaymentMethod", "Electronic check"))
        st.session_state.sim_mrr=float(baseline_series.get("MonthlyCharges", 65.0))
        st.session_state.sim_tech=str(baseline_series.get("TechSupport", "No"))
        st.session_state.sim_security=str(baseline_series.get("OnlineSecurity", "No"))
    col_ctrl1, col_ctrl2=st.columns(2)
    with col_ctrl1:
        contract_options=["Month-to-month", "One year", "Two year"]
        current_contract_idx=contract_options.index(st.session_state.sim_contract) if st.session_state.sim_contract in contract_options else 0
        sim_contract=st.selectbox(
            "Contract Term Commitment",
            options=contract_options,
            index=current_contract_idx,
            help="Simulate locking the customer into an annual commitment."
        )
        payment_options=[
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
        current_payment_idx=payment_options.index(st.session_state.sim_payment) if st.session_state.sim_payment in payment_options else 0
        sim_payment=st.selectbox(
            "Billing & Payment Channel",
            options=payment_options,
            index=current_payment_idx,
            help="Simulate switching from manual electronic check to automated billing."
        )
    with col_ctrl2:
        sim_mrr=st.slider(
            "Adjusted Monthly Rate ($)",
            min_value=18.0,
            max_value=130.0,
            value=float(st.session_state.sim_mrr),
            step=1.0,
            help="Simulate applying promotional loyalty discounts to reduce price sensitivity."
        )
        st_c1, st_c2=st.columns(2)
        with st_c1:
            tech_options=["No", "Yes"]
            current_tech_idx=tech_options.index(st.session_state.sim_tech) if st.session_state.sim_tech in tech_options else 0
            sim_tech=st.selectbox("VIP Tech Support", options=tech_options, index=current_tech_idx)
        with st_c2:
            sec_options=["No", "Yes"]
            current_sec_idx=sec_options.index(st.session_state.sim_security) if st.session_state.sim_security in sec_options else 0
            sim_security=st.selectbox("Online Security Suite", options=sec_options, index=current_sec_idx)
    base_df=pd.DataFrame([baseline_series.to_dict()])
    sim_df=base_df.copy()
    sim_df["Contract"]=sim_contract
    sim_df["PaymentMethod"]=sim_payment
    sim_df["MonthlyCharges"]=sim_mrr
    sim_df["TechSupport"]=sim_tech
    sim_df["OnlineSecurity"]=sim_security
    base_prob=float(pipeline.predict_proba(base_df)[0][1])
    sim_prob=float(pipeline.predict_proba(sim_df)[0][1])
    delta_prob=sim_prob-base_prob
    simulate_both=False
    if alt_pipeline is not None:
        simulate_both=st.checkbox(
            f"Compare simulation outcome against {alt_model_name} simultaneously",
            value=True,
            help="Evaluate how both linear and non-linear models react to the same intervention."
        )
    st.markdown("---")
    if simulate_both and alt_pipeline is not None:
        alt_base_prob=float(alt_pipeline.predict_proba(base_df)[0][1])
        alt_sim_prob=float(alt_pipeline.predict_proba(sim_df)[0][1])
        alt_delta=alt_sim_prob-alt_base_prob
        st.markdown("##### 🔬 Dual-Model Simulation Comparison")
        col_m1, col_m2=st.columns(2)
        with col_m1:
            st.markdown(f"**{model_name}:**")
            sub_c1, sub_c2=st.columns(2)
            sub_c1.metric("Baseline Risk", f"{base_prob * 100:.1f}%")
            sub_c2.metric("Simulated Risk", f"{sim_prob * 100:.1f}%", delta=f"{delta_prob * 100:.1f}%", delta_color="inverse")
        with col_m2:
            st.markdown(f"**{alt_model_name}:**")
            sub_a1, sub_a2=st.columns(2)
            sub_a1.metric("Baseline Risk", f"{alt_base_prob * 100:.1f}%")
            sub_a2.metric("Simulated Risk", f"{alt_sim_prob * 100:.1f}%", delta=f"{alt_delta * 100:.1f}%", delta_color="inverse")
    else:
        m1, m2=st.columns(2)
        m1.metric("Baseline Churn Risk", f"{base_prob * 100:.1f}%")
        m2.metric(
            "Simulated Churn Risk",
            f"{sim_prob * 100:.1f}%",
            delta=f"{delta_prob * 100:.1f}%",
            delta_color="inverse"
        )
    if sim_prob<0.40 and base_prob>=0.40:
        outcome_badge=(
            "<div style='background-color:#dcfce7; color:#166534; padding:0.8rem; border-radius:8px; font-weight:600; margin-top:0.5rem;'>"
            f"🟢 Strategy Highly Effective ({model_name}): Account successfully de-escalated to LOW RISK safe tier."
            "</div>"
        )
    elif sim_prob<base_prob:
        outcome_badge=(
            "<div style='background-color:#fef3c7; color:#92400e; padding:0.8rem; border-radius:8px; font-weight:600; margin-top:0.5rem;'>"
            f"🟡 Moderate Risk Reduction ({model_name}): Risk mitigated, but additional contractual incentives recommended."
            "</div>"
        )
    else:
        outcome_badge=(
            "<div style='background-color:#fee2e2; color:#991b1b; padding:0.8rem; border-radius:8px; font-weight:600; margin-top:0.5rem;'>"
            f"🔴 Strategy Ineffective ({model_name}): Risk remains elevated. Core underlying drivers are not addressed."
            "</div>"
        )
    st.markdown(outcome_badge, unsafe_allow_html=True)
