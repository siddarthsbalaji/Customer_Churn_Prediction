"""
AI-Powered Metric Classification & Schema Understanding Service.
Inspects uploaded customer dataset headers, sample values, and data types
to classify metrics, map them to canonical churn features, and preserve custom context.
"""
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

from src.config import (
    AI_PROVIDER,
    ALL_FEATURES,
    CANONICAL_STATISTICAL_DEFAULTS,
    COLUMN_ALIASES,
    FEATURE_SCHEMA_DESCRIPTIONS,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    ID_COLUMN,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    TARGET_COLUMN,
)

logger = logging.getLogger(__name__)


@dataclass
class MetricClassification:
    """Classification record for an individual dataset column/metric."""
    original_name: str
    role: str  # IDENTIFIER, TENURE, FINANCIAL_MONTHLY, CONTRACT_TYPE, DEMOGRAPHIC, etc.
    canonical_mapping: Optional[str]  # e.g. "tenure", "MonthlyCharges", or None if custom
    confidence: float  # 0.0 to 1.0
    reasoning: str
    value_mapping: Optional[Dict[str, str]] = None  # e.g. {"12M": "One year"}


@dataclass
class AIMetricClassificationResult:
    """Comprehensive dataset classification and alignment report."""
    metrics: Dict[str, MetricClassification] = field(default_factory=dict)
    id_column: Optional[str] = None
    target_column: Optional[str] = None
    mapped_canonical: Dict[str, str] = field(default_factory=dict)  # canonical -> original
    custom_attributes: List[str] = field(default_factory=list)      # e.g. ['nationality', 'credit_score']
    missing_canonical: List[str] = field(default_factory=list)     # canonical features not present
    provider_used: str = "heuristic_fallback"
    status_message: str = "Heuristic classification applied."

    def get_summary(self) -> Dict[str, Any]:
        """Returns JSON-serializable summary of the classification result."""
        return {
            "provider_used": self.provider_used,
            "status_message": self.status_message,
            "id_column": self.id_column,
            "target_column": self.target_column,
            "mapped_canonical": self.mapped_canonical,
            "custom_attributes": self.custom_attributes,
            "missing_canonical": self.missing_canonical,
            "metrics": {
                k: {
                    "role": v.role,
                    "canonical_mapping": v.canonical_mapping,
                    "confidence": v.confidence,
                    "reasoning": v.reasoning,
                    "value_mapping": v.value_mapping,
                }
                for k, v in self.metrics.items()
            },
        }


def inspect_dataset_schema(df: pd.DataFrame, max_samples: int = 3) -> List[Dict[str, Any]]:
    """
    Extracts column headers, inferred data types, null rates, and sample values.
    """
    summary = []
    for col in df.columns:
        series = df[col]
        non_null_samples = series.dropna().astype(str).str.strip()
        samples = [v for v in non_null_samples.unique()[:max_samples] if v != ""]
        summary.append({
            "column_name": str(col),
            "inferred_type": str(series.dtype),
            "null_count": int(series.isna().sum()),
            "unique_count": int(series.nunique(dropna=True)),
            "sample_values": samples,
        })
    return summary


def _build_classification_prompt(schema_summary: List[Dict[str, Any]]) -> str:
    """Constructs prompt for LLM metric classification."""
    canonical_spec = json.dumps(FEATURE_SCHEMA_DESCRIPTIONS, indent=2)
    schema_json = json.dumps(schema_summary, indent=2)

    return f"""You are an expert Data Engineer & Machine Learning Solutions Architect.
A customer has uploaded an arbitrary dataset with custom metrics (column headers).
Your task is to analyze each metric title (and its sample values/data type) and classify it:
1. Identify the semantic role of each column:
   - "IDENTIFIER": Unique customer/account IDs (e.g. client_id, account_code, user_uuid).
   - "TENURE": Time length customer has been with company in months/days (e.g. account_age, months_active).
   - "FINANCIAL_MONTHLY": Monthly fee, billing rate, MRR, recurring spend.
   - "FINANCIAL_CUMULATIVE": Total lifetime spend, cumulative charges, account balance.
   - "CONTRACT_TYPE": Commitment terms (e.g. Month-to-month, Annual, 12M, Two year).
   - "PAYMENT_METHOD": Payment channel (e.g. credit card, bank transfer, check).
   - "CORE_SERVICE": Primary product or internet service tier.
   - "TECH_SUPPORT": Dedicated support tier or ticket indicator.
   - "DEMOGRAPHIC": Demographic customer information (e.g. nationality, country, gender, age, family).
   - "TARGET": Ground truth churn indicator (e.g. churn, exited, left, cancelled).
   - "CUSTOM_CONTEXT": Useful domain metrics not in canonical model (e.g. credit_score, balance, num_products, notes).

2. Canonical Feature Mapping:
   Determine if the metric directly maps to one of our canonical model features:
   {canonical_spec}

   Important rules:
   - If a metric is a custom demographic attribute like "nationality" or "country", classify role as "DEMOGRAPHIC", and set canonical_mapping to null (or "custom_demographic"). It will be preserved as a rich custom attribute!
   - If a metric is "credit_score" or "balance", classify role as "CUSTOM_CONTEXT" with canonical_mapping: null.
   - If it maps to a canonical feature (like account_age -> tenure, mrr -> MonthlyCharges), set canonical_mapping to that exact feature name.
   - Provide value_mapping if the metric has categorical values that should map to canonical terms (e.g. {{"12M": "One year", "1M": "Month-to-month"}} or {{"M": "Male", "F": "Female"}}).

Dataset Schema to Classify:
{schema_json}

Return ONLY a valid JSON object strictly matching this format:
{{
  "metrics": [
    {{
      "original_name": "column_name",
      "role": "ROLE_NAME",
      "canonical_mapping": "canonical_feature_name_or_null",
      "confidence": 0.95,
      "reasoning": "Brief one sentence explanation.",
      "value_mapping": null
    }}
  ]
}}
"""


def _extract_json_payload(response_text: str) -> Optional[Dict[str, Any]]:
    """Safely extracts JSON dict from LLM response text."""
    clean_text = response_text.strip()
    # Try direct parse
    try:
        return json.loads(clean_text)
    except json.JSONDecodeError:
        pass

    # Try extracting markdown fenced code blocks ```json ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding outermost braces { ... }
    match = re.search(r"(\{.*\})", clean_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    return None


def call_gemini_api(
    schema_summary: List[Dict[str, Any]],
    api_key: str,
    model: str = GEMINI_MODEL,
    timeout_sec: int = 15
) -> Optional[Dict[str, Any]]:
    """Calls Google Gemini REST API to classify metrics."""
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    prompt = _build_classification_prompt(schema_summary)

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json"
        }
    }

    try:
        response = requests.post(endpoint, json=payload, timeout=timeout_sec)
        if response.status_code != 200:
            logger.warning(f"Gemini API returned status {response.status_code}: {response.text}")
            return None
        data = response.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return _extract_json_payload(text)
    except Exception as e:
        logger.warning(f"Gemini API invocation failed: {e}")
        return None


def call_openai_api(
    schema_summary: List[Dict[str, Any]],
    api_key: str,
    model: str = OPENAI_MODEL,
    timeout_sec: int = 15
) -> Optional[Dict[str, Any]]:
    """Calls OpenAI Chat Completions REST API to classify metrics."""
    endpoint = "https://api.openai.com/v1/chat/completions"
    prompt = _build_classification_prompt(schema_summary)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "You are a specialized machine learning data schema alignment engine."},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=timeout_sec)
        if response.status_code != 200:
            logger.warning(f"OpenAI API returned status {response.status_code}: {response.text}")
            return None
        data = response.json()
        text = data["choices"][0]["message"]["content"]
        return _extract_json_payload(text)
    except Exception as e:
        logger.warning(f"OpenAI API invocation failed: {e}")
        return None


def heuristic_classify_metrics(schema_summary: List[Dict[str, Any]]) -> AIMetricClassificationResult:
    """
    Intelligent heuristic & regex-based schema classifier used as fallback
    when no API key is provided or when offline.
    """
    # Build reverse lookup for exact aliases
    alias_lookup: Dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        clean_canon = re.sub(r"[_\s\-]+", "", canonical.lower())
        alias_lookup[clean_canon] = canonical
        for a in aliases:
            clean_a = re.sub(r"[_\s\-]+", "", a.lower())
            alias_lookup[clean_a] = canonical

    metrics: Dict[str, MetricClassification] = {}
    mapped_canonical: Dict[str, str] = {}
    custom_attributes: List[str] = []
    id_col: Optional[str] = None
    target_col: Optional[str] = None

    for item in schema_summary:
        col = item["column_name"]
        norm = re.sub(r"[_\s\-]+", "", col.lower())

        # Check alias lookup first
        canonical = alias_lookup.get(norm)
        if canonical:
            if canonical == ID_COLUMN:
                role = "IDENTIFIER"
                id_col = col
            elif canonical == TARGET_COLUMN:
                role = "TARGET"
                target_col = col
            elif canonical in ["tenure"]:
                role = "TENURE"
            elif canonical in ["MonthlyCharges"]:
                role = "FINANCIAL_MONTHLY"
            elif canonical in ["TotalCharges"]:
                role = "FINANCIAL_CUMULATIVE"
            elif canonical in ["Contract"]:
                role = "CONTRACT_TYPE"
            elif canonical in ["PaymentMethod"]:
                role = "PAYMENT_METHOD"
            elif canonical in ["InternetService"]:
                role = "CORE_SERVICE"
            elif canonical in ["TechSupport"]:
                role = "TECH_SUPPORT"
            elif canonical in ["gender", "SeniorCitizen", "Partner", "Dependents"]:
                role = "DEMOGRAPHIC"
            else:
                role = "ADDON_SERVICE"

            classification = MetricClassification(
                original_name=col,
                role=role,
                canonical_mapping=canonical,
                confidence=0.92,
                reasoning=f"Matched known alias pattern for canonical feature '{canonical}'."
            )
            metrics[col] = classification
            mapped_canonical[canonical] = col
            continue

        # Heuristic rules for custom / domain metrics
        if any(term in norm for term in ["nationality", "country", "nation", "citizenship", "geography", "region"]):
            metrics[col] = MetricClassification(
                original_name=col,
                role="DEMOGRAPHIC",
                canonical_mapping=None,
                confidence=0.95,
                reasoning="Identified as customer geographic / nationality context."
            )
            custom_attributes.append(col)
        elif any(term in norm for term in ["clientid", "accountcode", "usercode", "custno", "customerid", "uuid"]):
            metrics[col] = MetricClassification(
                original_name=col,
                role="IDENTIFIER",
                canonical_mapping="customerID",
                confidence=0.88,
                reasoning="Identified as account / customer identifier."
            )
            if not id_col:
                id_col = col
                mapped_canonical["customerID"] = col
        elif any(term in norm for term in ["monthssubscribed", "accountage", "monthsactive", "tenuremonths"]):
            metrics[col] = MetricClassification(
                original_name=col,
                role="TENURE",
                canonical_mapping="tenure",
                confidence=0.89,
                reasoning="Identified as account longevity / tenure metric."
            )
            mapped_canonical["tenure"] = col
        elif any(term in norm for term in ["monthlyfee", "monthlyrate", "billingrate", "recurringspend", "mrr"]):
            metrics[col] = MetricClassification(
                original_name=col,
                role="FINANCIAL_MONTHLY",
                canonical_mapping="MonthlyCharges",
                confidence=0.90,
                reasoning="Identified as monthly billing / recurring revenue."
            )
            mapped_canonical["MonthlyCharges"] = col
        elif any(term in norm for term in ["balance", "cumulativespend", "lifetimespend", "totallifetime"]):
            metrics[col] = MetricClassification(
                original_name=col,
                role="FINANCIAL_CUMULATIVE",
                canonical_mapping="TotalCharges",
                confidence=0.85,
                reasoning="Identified as cumulative balance or spend volume."
            )
            mapped_canonical["TotalCharges"] = col
        elif any(term in norm for term in ["contractterm", "planduration", "termtype", "billingcadence"]):
            metrics[col] = MetricClassification(
                original_name=col,
                role="CONTRACT_TYPE",
                canonical_mapping="Contract",
                confidence=0.88,
                reasoning="Identified as contract commitment duration."
            )
            mapped_canonical["Contract"] = col
        elif any(term in norm for term in ["paymentchannel", "paymentmethod", "billingmethod", "cardtype"]):
            metrics[col] = MetricClassification(
                original_name=col,
                role="PAYMENT_METHOD",
                canonical_mapping="PaymentMethod",
                confidence=0.88,
                reasoning="Identified as payment channel."
            )
            mapped_canonical["PaymentMethod"] = col
        elif any(term in norm for term in ["churn", "churned", "exited", "status", "attrition", "left"]):
            metrics[col] = MetricClassification(
                original_name=col,
                role="TARGET",
                canonical_mapping="Churn",
                confidence=0.91,
                reasoning="Identified as target churn ground-truth indicator."
            )
            target_col = col
            mapped_canonical["Churn"] = col
        else:
            metrics[col] = MetricClassification(
                original_name=col,
                role="CUSTOM_CONTEXT",
                canonical_mapping=None,
                confidence=0.75,
                reasoning="Retained as custom domain attribute for display and diagnostic context."
            )
            custom_attributes.append(col)

    missing = [feat for feat in ALL_FEATURES if feat not in mapped_canonical]

    return AIMetricClassificationResult(
        metrics=metrics,
        id_column=id_col,
        target_column=target_col,
        mapped_canonical=mapped_canonical,
        custom_attributes=custom_attributes,
        missing_canonical=missing,
        provider_used="heuristic_fallback",
        status_message="Offline semantic heuristic classification applied."
    )


def classify_dataset(
    df: pd.DataFrame,
    api_key: Optional[str] = None,
    provider: Optional[str] = None
) -> AIMetricClassificationResult:
    """
    Master classification function.
    Inspects columns & sample data, checks for API key, and calls LLM or fallback.
    """
    schema_summary = inspect_dataset_schema(df)

    active_provider = (provider or AI_PROVIDER or "gemini").lower()
    key = api_key or (GEMINI_API_KEY if active_provider == "gemini" else OPENAI_API_KEY)

    # If no key for requested provider, check alternate
    if not key:
        if GEMINI_API_KEY:
            active_provider = "gemini"
            key = GEMINI_API_KEY
        elif OPENAI_API_KEY:
            active_provider = "openai"
            key = OPENAI_API_KEY

    # If key is available, call AI
    ai_raw = None
    if key:
        if active_provider == "gemini":
            ai_raw = call_gemini_api(schema_summary, api_key=key)
        elif active_provider == "openai":
            ai_raw = call_openai_api(schema_summary, api_key=key)

    if ai_raw and "metrics" in ai_raw and isinstance(ai_raw["metrics"], list):
        metrics_dict: Dict[str, MetricClassification] = {}
        mapped_canonical: Dict[str, str] = {}
        custom_attributes: List[str] = []
        id_col: Optional[str] = None
        target_col: Optional[str] = None

        for m in ai_raw["metrics"]:
            name = m.get("original_name")
            if not name or name not in df.columns:
                continue
            role = str(m.get("role", "CUSTOM_CONTEXT")).upper()
            mapping = m.get("canonical_mapping")
            if mapping in ["null", "None", "", None] or mapping not in FEATURE_SCHEMA_DESCRIPTIONS:
                mapping = None

            conf = float(m.get("confidence", 0.90))
            reason = str(m.get("reasoning", "AI classified metric role."))
            val_map = m.get("value_mapping")

            classification = MetricClassification(
                original_name=name,
                role=role,
                canonical_mapping=mapping,
                confidence=conf,
                reasoning=reason,
                value_mapping=val_map if isinstance(val_map, dict) else None
            )
            metrics_dict[name] = classification

            if mapping:
                mapped_canonical[mapping] = name
                if mapping == ID_COLUMN:
                    id_col = name
                elif mapping == TARGET_COLUMN:
                    target_col = name
            else:
                custom_attributes.append(name)

        # Catch any missing columns not returned by AI
        for item in schema_summary:
            c = item["column_name"]
            if c not in metrics_dict:
                metrics_dict[c] = MetricClassification(
                    original_name=c,
                    role="CUSTOM_CONTEXT",
                    canonical_mapping=None,
                    confidence=0.70,
                    reasoning="Preserved custom attribute."
                )
                custom_attributes.append(c)

        missing = [feat for feat in ALL_FEATURES if feat not in mapped_canonical]

        return AIMetricClassificationResult(
            metrics=metrics_dict,
            id_column=id_col,
            target_column=target_col,
            mapped_canonical=mapped_canonical,
            custom_attributes=custom_attributes,
            missing_canonical=missing,
            provider_used=f"ai_{active_provider}",
            status_message=f"AI-Powered Metric Classification successful via {active_provider.title()}."
        )

    # Fallback to intelligent heuristics
    logger.info("Using heuristic fallback for metric classification.")
    return heuristic_classify_metrics(schema_summary)
