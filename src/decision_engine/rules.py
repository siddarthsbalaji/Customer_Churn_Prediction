"""
Prescriptive Action Engine (Next-Best-Action Rules).
Maps calibrated churn risk tiers and individual SHAP root-cause drivers to actionable retention playbooks.
"""
from typing import Any, Dict, List, Optional

# Standard Action Definitions
PLAYBOOKS = {
    "CONTRACT_UPGRADE": {
        "action_code": "ACT_CONTRACT_UPGRADE",
        "action_title": "1-Year Commitment Upgrade with 15% Monthly Discount Lock-in",
        "priority": "CRITICAL",
        "incentive_type": "Monthly Rate Lock Discount",
        "estimated_cost": "$10-$15/month",
        "recommended_channel": "Dedicated Retention Specialist (Direct Outreach)",
        "playbook_details": (
            "Customer is on a high-attrition Month-to-month contract. "
            "Offer a 1-year agreement locking in a 15% monthly rate reduction, "
            "converting flexible risk into contractually bound retention."
        )
    },
    "AUTOPAY_INCENTIVE": {
        "action_code": "ACT_AUTOPAY_INCENTIVE",
        "action_title": "Switch to Auto-Pay with $20 Instant Billing Credit",
        "priority": "HIGH",
        "incentive_type": "One-Time Account Credit",
        "estimated_cost": "$20 one-time",
        "recommended_channel": "Automated Email & Mobile Push Notification",
        "playbook_details": (
            "Customer utilizes manual billing (Electronic Check/Mailed Check), "
            "a leading driver of transaction friction. Incentivize enrollment into "
            "automatic bank direct debit or credit card payments."
        )
    },
    "TECH_SUPPORT_VIP": {
        "action_code": "ACT_TECH_SUPPORT_VIP",
        "action_title": "Complimentary 3-Month VIP TechSupport & Security Onboarding",
        "priority": "HIGH",
        "incentive_type": "Free Service Addon",
        "estimated_cost": "$0 marginal cost",
        "recommended_channel": "Customer Success Onboarding Specialist",
        "playbook_details": (
            "Customer lacks technical support and online security add-ons on a high-speed connection. "
            "Provide 90 days of white-glove technical assistance to resolve early service friction."
        )
    },
    "LOYALTY_BUNDLE_REPACK": {
        "action_code": "ACT_LOYALTY_BUNDLE_REPACK",
        "action_title": "Value Repackaging & Loyalty Speed Tier Upgrade",
        "priority": "MEDIUM",
        "incentive_type": "Service Tier Upgrade",
        "estimated_cost": "$5/month",
        "recommended_channel": "Account Manager Check-in",
        "playbook_details": (
            "High charge velocity relative to tenure indicates price sensitivity. "
            "Review customer usage and repackage services into a bundled discount tier."
        )
    },
    "PROACTIVE_CHECKIN": {
        "action_code": "ACT_PROACTIVE_CHECKIN",
        "action_title": "Proactive Satisfaction Survey & Relationship Check-in",
        "priority": "MEDIUM",
        "incentive_type": "Relationship Nurture",
        "estimated_cost": "$0",
        "recommended_channel": "Automated Relationship Survey / CSM Email",
        "playbook_details": (
            "Account is in the moderate risk tier without singular acute contractual drivers. "
            "Deploy a quick feedback inquiry to identify underlying satisfaction concerns."
        )
    },
    "ORGANIC_NURTURE": {
        "action_code": "ACT_ORGANIC_NURTURE",
        "action_title": "Maintain Standard Loyalty Stream & Cross-Sell Add-ons",
        "priority": "LOW",
        "incentive_type": "None (Upsell Candidate)",
        "estimated_cost": "$0",
        "recommended_channel": "Marketing Newsletter / Self-Service Portal",
        "playbook_details": (
            "Customer exhibits low churn probability and strong tenure stability. "
            "No defensive discounts needed; eligible for value-add feature cross-selling."
        )
    }
}


def prescribe_retention_action(
    churn_prob: float,
    top_shap_drivers: List[Dict[str, Any]],
    customer_record: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Evaluates churn probability and positive SHAP risk drivers to prescribe Next-Best-Action.

    Args:
        churn_prob: Calibrated churn prediction probability [0.0, 1.0].
        top_shap_drivers: Ordered list of top positive SHAP contributors, e.g.:
            [{'feature': 'Contract_Month-to-month', 'shap_value': 0.18, 'display_name': 'Month-to-month Contract'}, ...]
        customer_record: Optional original raw customer attributes dictionary.

    Returns:
        Structured prescriptive recommendation dictionary.
    """
    # 1. Low Risk Tier (P < 0.40)
    if churn_prob < 0.40:
        recommendation = PLAYBOOKS["ORGANIC_NURTURE"].copy()
        recommendation["trigger_rationale"] = (
            f"Low calibrated churn risk ({churn_prob * 100:.1f}%). Account is in the safe loyalty zone."
        )
        return recommendation

    # Extract driver names from top positive contributors
    driver_features = [d.get("feature", "").lower() for d in top_shap_drivers]
    primary_driver = top_shap_drivers[0]["display_name"] if top_shap_drivers else "General Risk"

    # 2. Critical Risk Tier (P >= 0.70)
    if churn_prob >= 0.70:
        # Check if contract is a dominant driver
        if any("contract" in f and "month" in f for f in driver_features):
            rec = PLAYBOOKS["CONTRACT_UPGRADE"].copy()
            rec["trigger_rationale"] = (
                f"Critical risk ({churn_prob * 100:.1f}%) predominantly fueled by {primary_driver}. "
                "Urgent contractual lock-in required before next billing cycle."
            )
            return rec

        # Check if payment method friction is dominant
        if any("payment" in f or "electronic" in f for f in driver_features):
            rec = PLAYBOOKS["AUTOPAY_INCENTIVE"].copy()
            rec["trigger_rationale"] = (
                f"Critical risk ({churn_prob * 100:.1f}%) driven by payment method friction ({primary_driver}). "
                "Switching to automated billing stabilizes retention."
            )
            return rec

        # Check if pricing / charge velocity is dominant
        if any("charge" in f or "monthly" in f for f in driver_features):
            rec = PLAYBOOKS["LOYALTY_BUNDLE_REPACK"].copy()
            rec["priority"] = "CRITICAL"
            rec["trigger_rationale"] = (
                f"Critical risk ({churn_prob * 100:.1f}%) driven by high pricing perception ({primary_driver})."
            )
            return rec

        # Fallback for critical tier
        rec = PLAYBOOKS["CONTRACT_UPGRADE"].copy()
        rec["trigger_rationale"] = (
            f"Critical risk ({churn_prob * 100:.1f}%) with primary risk driver: {primary_driver}."
        )
        return rec

    # 3. Moderate Risk Tier (0.40 <= P < 0.70)
    if any("techsupport" in f or "security" in f for f in driver_features):
        rec = PLAYBOOKS["TECH_SUPPORT_VIP"].copy()
        rec["trigger_rationale"] = (
            f"Moderate risk ({churn_prob * 100:.1f}%) linked to missing service security/support ({primary_driver}). "
            "Free trial creates stickiness."
        )
        return rec

    if any("contract" in f for f in driver_features):
        rec = PLAYBOOKS["CONTRACT_UPGRADE"].copy()
        rec["priority"] = "HIGH"
        rec["trigger_rationale"] = (
            f"Moderate risk ({churn_prob * 100:.1f}%) with Month-to-month vulnerability ({primary_driver})."
        )
        return rec

    # Default moderate tier action
    rec = PLAYBOOKS["PROACTIVE_CHECKIN"].copy()
    rec["trigger_rationale"] = (
        f"Moderate risk ({churn_prob * 100:.1f}%) driven by {primary_driver}. Relationship outreach recommended."
    )
    return rec
