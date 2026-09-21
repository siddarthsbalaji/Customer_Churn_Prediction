"""
Executive Priority Matrix & Batch Risk Ranking Component.
"""
from typing import Optional, Tuple
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline
from src.decision_engine.rules import prescribe_retention_action
def render_priority_matrix(
    pipeline: Pipeline,
    df: pd.DataFrame,
    metadata: dict,
    model_name: str = "Random Forest",
    threshold: Optional[float]=None
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Computes batch predictions, displays KPI cards, renders sorted priority table,
    and enables one-click enriched CSV downloads for the selected model.
    """
    if threshold is None:
        threshold=metadata.get("optimal_threshold", 0.210)
    st.markdown(f"### 2. Executive Retention Priority Matrix — `{model_name}`")
    st.caption(f"Accounts evaluated via **{model_name}** using decision threshold `τ* = {threshold:.3f}`.")
    scored_df=df.copy()
    probs=pipeline.predict_proba(scored_df)[:, 1]
    scored_df["churn_probability"]=probs.round(4)
    scored_df["churn_risk_pct"]=(probs*100).round(1)
    scored_df["is_at_risk"]=scored_df["churn_probability"]>=threshold
    def assign_tier(p: float) -> str:
        if p>=0.70:
            return "CRITICAL"
        elif p>=0.40:
            return "MODERATE"
        return "LOW"
    scored_df["risk_tier"]=scored_df["churn_probability"].apply(assign_tier)
    prescribed_actions=[]
    primary_drivers=[]
    for _, row in scored_df.iterrows():
        p=float(row["churn_probability"])
        contract=str(row.get("Contract", ""))
        payment=str(row.get("PaymentMethod", ""))
        tech_support=str(row.get("TechSupport", ""))
        if contract == "Month-to-month":
            driver = "Month-to-month Contract (High Risk)"
            driver_key = "Contract_Month-to-month"
        elif payment == "Electronic check":
            driver = "Electronic Check Billing"
            driver_key = "PaymentMethod_Electronic check"
        elif tech_support == "No":
            driver = "Lacks Technical Support"
            driver_key = "TechSupport_No"
        else:
            driver = "Pricing & Spend Velocity"
            driver_key = "MonthlyCharges"
        action=prescribe_retention_action(
            churn_prob=p,
            top_shap_drivers=[{"feature": driver_key, "display_name": driver}]
        )
        prescribed_actions.append(action["action_title"])
        primary_drivers.append(driver)
    scored_df["primary_risk_driver"]=primary_drivers
    scored_df["prescribed_action"]=prescribed_actions
    ranked_df=scored_df.sort_values(by="churn_probability", ascending=False).reset_index(drop=True)
    total_customers=len(ranked_df)
    at_risk_count=int(ranked_df["is_at_risk"].sum())
    critical_count=int((ranked_df["risk_tier"] == "CRITICAL").sum())
    total_mrr_at_risk=float(ranked_df.loc[ranked_df["is_at_risk"], "MonthlyCharges"].sum())
    kpi1, kpi2, kpi3, kpi4=st.columns(4)
    kpi1.metric("Analyzed Customers", f"{total_customers:,}")
    kpi2.metric("At-Risk Population", f"{at_risk_count:,}", f"{(at_risk_count/max(1, total_customers))*100:.1f}% of total")
    kpi3.metric("Critical Risk Accounts", f"{critical_count:,}", f"{(critical_count/max(1, total_customers))*100:.1f}% tier")
    kpi4.metric("MRR at Risk", f"${total_mrr_at_risk:,.2f}", "Monthly Revenue")
    fcol1, fcol2, fcol3=st.columns([2, 2, 2])
    with fcol1:
        tier_filter=st.multiselect(
            "Filter by Risk Tier:",
            options=["CRITICAL", "MODERATE", "LOW"],
            default=["CRITICAL", "MODERATE", "LOW"]
        )
    with fcol2:
        search_query=st.text_input("Search Account ID:", placeholder="e.g., 7590-VHVEG")
    filtered_df=ranked_df[ranked_df["risk_tier"].isin(tier_filter)]
    if search_query.strip():
        filtered_df=filtered_df[filtered_df["customerID"].astype(str).str.contains(search_query.strip(), case=False)]
    standard_internal = set(ALL_FEATURES) | {
        "customerID", "churn_probability", "churn_risk_pct", "is_at_risk",
        "risk_tier", "primary_risk_driver", "prescribed_action", "Churn"
    }
    custom_cols = [c for c in filtered_df.columns if c not in standard_internal]

    display_cols = ["customerID"] + custom_cols + [
        "churn_risk_pct", "risk_tier", "MonthlyCharges",
        "tenure", "Contract", "primary_risk_driver", "prescribed_action"
    ]
    # Filter only available columns
    actual_display_cols = [c for c in display_cols if c in filtered_df.columns]

    rename_map = {
        "customerID": "Account ID",
        "churn_risk_pct": "Churn Risk (%)",
        "risk_tier": "Risk Tier",
        "MonthlyCharges": "MRR ($)",
        "tenure": "Tenure (Mo)",
        "Contract": "Contract",
        "primary_risk_driver": "Primary Risk Driver",
        "prescribed_action": "Prescribed Retention Action"
    }
    for c in custom_cols:
        rename_map[c] = f"🌐 {c.replace('_', ' ').title()}"

    st.dataframe(
        filtered_df[actual_display_cols].rename(columns=rename_map),
        use_container_width=True,
        height=320
    )
    csv_data=filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Export Prioritized Accounts (CSV)",
        data=csv_data,
        file_name="prioritized_customer_retention_list.csv",
        mime="text/csv",
        help="Download complete dataset enriched with churn scores, risk tiers, and playbooks."
    )
    customer_options=ranked_df["customerID"].tolist()

    def format_customer_label(cid: str) -> str:
        row = ranked_df[ranked_df["customerID"] == cid].iloc[0]
        context_parts = []
        for c in ["nationality", "country"]:
            if c in ranked_df.columns:
                context_parts.append(str(row[c]))
        risk_pct = row["churn_risk_pct"]
        ctx = f" ({', '.join(context_parts)})" if context_parts else ""
        return f"{cid}{ctx} — {risk_pct:.1f}% Risk [{row['risk_tier']}]"

    selected_customer_id=st.selectbox(
        "Select customer account for deep-dive root-cause diagnostics & What-If simulation:",
        options=customer_options,
        index=0 if customer_options else 0,
        format_func=format_customer_label
    )
    return ranked_df, selected_customer_id
