"""
Batch CSV Upload, Schema Validation & Risk Ranking Endpoints with Model Selection.
"""
import io
from typing import Any, List
from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse
import pandas as pd
from starlette.concurrency import run_in_threadpool

from app.api.schemas import BatchCustomerResult, BatchSummaryResponse
from src.data.validator import validate_and_align_dataset
from src.decision_engine.rules import prescribe_retention_action

router = APIRouter(tags=["Batch Processing"])
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB limit


def _process_batch_sync(
    contents: bytes,
    pipeline: Any,
    threshold: float,
    model: str,
    allow_custom_adapter: bool
) -> BatchSummaryResponse:
    """Synchronous CPU-bound batch processing executed inside worker threadpool."""
    try:
        raw_df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse uploaded CSV: {str(e)}")

    aligned_df, missing_cols, warnings = validate_and_align_dataset(
        raw_df,
        allow_custom_adapter=allow_custom_adapter
    )
    if missing_cols or aligned_df is None:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Schema Validation Failed",
                "missing_required_columns": missing_cols,
                "message": f"Uploaded file is missing required core columns: {missing_cols}"
            }
        )

    try:
        probs = pipeline.predict_proba(aligned_df)[:, 1]
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Batch model prediction failed: {str(e)}")

    aligned_df["churn_probability"] = probs.round(4)
    aligned_df["is_at_risk"] = aligned_df["churn_probability"] >= threshold

    def assign_tier(p: float) -> str:
        if p >= 0.70:
            return "CRITICAL"
        elif p >= threshold:
            return "MODERATE"
        return "LOW"

    aligned_df["risk_tier"] = aligned_df["churn_probability"].apply(assign_tier)
    results: List[BatchCustomerResult] = []

    for _, row in aligned_df.iterrows():
        p = float(row["churn_probability"])
        cust_id = str(row.get("customerID", "CUST-UNKNOWN"))
        raw_mrr = row.get("MonthlyCharges", 0.0)
        mrr = float(raw_mrr) if pd.notna(raw_mrr) else 0.0

        if str(row.get("Contract", "")) == "Month-to-month":
            primary_driver = "Month-to-month Contract (High Risk)"
            driver_key = "Contract_Month-to-month"
        elif str(row.get("PaymentMethod", "")) == "Electronic check":
            primary_driver = "Electronic Check Billing"
            driver_key = "PaymentMethod_Electronic check"
        elif str(row.get("TechSupport", "")) == "No":
            primary_driver = "Lacks Technical Support"
            driver_key = "TechSupport_No"
        else:
            primary_driver = "Pricing & Spend Velocity"
            driver_key = "MonthlyCharges"

        action = prescribe_retention_action(
            churn_prob=p,
            top_shap_drivers=[{"feature": driver_key, "display_name": primary_driver}],
            decision_threshold=threshold
        )
        results.append(BatchCustomerResult(
            customer_id=cust_id,
            churn_probability=p,
            is_at_risk=p >= threshold,
            risk_tier=assign_tier(p),
            primary_risk_driver=primary_driver,
            prescribed_action_title=action["action_title"],
            monthly_charges=round(mrr, 2)
        ))

    results.sort(key=lambda x: x.churn_probability, reverse=True)
    at_risk_count = sum(1 for r in results if r.is_at_risk)
    critical_count = sum(1 for r in results if r.risk_tier == "CRITICAL")
    total_mrr_at_risk = sum(r.monthly_charges for r in results if r.is_at_risk)

    return BatchSummaryResponse(
        model_used=model,
        decision_threshold_applied=threshold,
        total_records=len(results),
        at_risk_count=at_risk_count,
        critical_risk_count=critical_count,
        total_mrr_at_risk=round(total_mrr_at_risk, 2),
        warnings=warnings,
        results=results
    )


@router.post("/batch-risk", response_model=BatchSummaryResponse)
async def process_batch_risk(
    request: Request,
    file: UploadFile = File(...),
    model: str = Query(
        default="random_forest",
        pattern="^(random_forest|xgboost)$",
        description="Choose model: 'random_forest' or 'xgboost'"
    ),
    allow_custom_adapter: bool = Query(
        default=False,
        description="Enable automatic schema alignment & statistical imputation for non-standard industry datasets."
    )
):
    """
    Ingests dynamic user CSV file, validates columns, executes batch model scoring
    with the selected model in a worker threadpool, and returns prioritized customer accounts ranked by churn probability.
    """
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Please upload a valid CSV file (.csv)."
        )
    models = getattr(request.app.state, "models", {})
    metadata = getattr(request.app.state, "metadata", {})
    pipeline = models.get(model)
    if pipeline is None:
        pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail=f"Model pipeline '{model}' is currently unavailable.")

    model_meta = metadata.get("models", {}).get(model, {})
    default_thresh = 0.210 if model == "random_forest" else 0.240
    threshold = model_meta.get("optimal_threshold", default_thresh)

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Maximum supported upload size is 50MB.")

    return await run_in_threadpool(
        _process_batch_sync,
        contents,
        pipeline,
        threshold,
        model,
        allow_custom_adapter
    )


@router.post("/batch-risk/export-csv")
async def export_batch_risk_csv(
    request: Request,
    file: UploadFile = File(...),
    model: str = Query(
        default="random_forest",
        pattern="^(random_forest|xgboost)$",
        description="Choose model: 'random_forest' or 'xgboost'"
    ),
    allow_custom_adapter: bool = Query(
        default=False,
        description="Enable automatic schema alignment for non-standard industry datasets."
    )
):
    """
    Executes batch inference with chosen model and streams an enriched CSV download
    including risk probabilities, risk tiers, and prescribed actions.
    """
    summary = await process_batch_risk(
        request,
        file,
        model=model,
        allow_custom_adapter=allow_custom_adapter
    )
    export_df = pd.DataFrame([r.model_dump() for r in summary.results])
    output = io.StringIO()
    export_df.to_csv(output, index=False)
    output.seek(0)
    response = StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv"
    )
    response.headers["Content-Disposition"] = f"attachment; filename=prioritized_retention_accounts_{model}.csv"
    return response
