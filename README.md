# Customer Churn Decision Engine

An enterprise-grade machine learning decision engine that transforms customer churn prediction from static retrospective analysis into an actionable, real-time retention application.

Built with **Scikit-Learn**, **FastAPI**, **SHAP**, and **Streamlit**, this project enables business stakeholders to ingest dynamic customer datasets, evaluate calibrated churn risk, diagnose root causes with local explainability, and simulate counterfactual retention interventions.

---

## System Architecture

```text
Dynamic CSV Upload ──► Schema Validation & Alias Matching
                               │
                               ▼
                   Unified Scikit-Learn Pipeline
                   (Cleaning, Scaling, One-Hot Encoding)
                               │
                               ▼
                   Champion Model (Random Forest)
                   & Cost-Calibrated Thresholding (F2)
                               │
                               ▼
                   Local SHAP Explainability Engine
                               │
                               ▼
                   Prescriptive Action Engine (Next-Best-Action)
                               │
        ┌──────────────────────┴──────────────────────┐
        ▼                                             ▼
FastAPI REST Service                           Streamlit Dashboard
(/predict, /explain, /batch-risk)              - Executive Priority Matrix
                                               - Waterfall Diagnostics
                                               - "What-If" Sandbox
```

---

## Directory Structure

```text
├── data/
│   ├── raw/                  # Original IBM Telco dataset
│   ├── processed/            # Train/test split datasets
│   └── sample_uploads/       # Test CSV files for dynamic UI upload
├── notebooks/                # Exploratory Data Analysis & experiments
├── models/                   # Serialized pipeline & explainer artifacts
├── src/
│   ├── config.py             # Feature schemas, paths, risk thresholds
│   ├── data/                 # Custom transformers, cleaners, validators
│   ├── models/               # Pipeline composition, training & evaluation
│   ├── explainability/       # SHAP diagnostics & TreeExplainer service
│   ├── decision_engine/      # Prescriptive Next-Best-Action rules
│   └── utils/                # Logging & file I/O helpers
├── app/
│   ├── api/                  # FastAPI inference backend
│   └── dashboard/            # Streamlit interactive application
├── tests/                    # Pytest validation test suite
├── PROJECT_BLUEPRINT.md      # Comprehensive technical execution blueprint
├── requirements.txt          # Python dependencies
└── Makefile                  # Development workflow automation
```

---

## Quickstart

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Launch FastAPI Inference Server
```bash
make run-api
# API documentation available at: http://localhost:8000/docs
```

### 3. Launch Streamlit Executive Dashboard
```bash
make run-ui
# Dashboard available at: http://localhost:8501
```

---

## Development Roadmap
- [x] **Step 1:** Project scaffolding, repository layout, dependency specification, and configuration setup.
- [ ] **Step 2 (Phase 1):** IBM Telco dataset ingestion, custom Scikit-Learn transformers (data cleaning, feature engineering), and pipeline serialization.
- [ ] **Step 3 (Phase 2):** Model tournament (Logistic Regression vs. Random Forest), 5-fold Stratified CV, and cost-benefit threshold calibration.
- [ ] **Step 4 (Phase 3):** SHAP explainability audit and Prescriptive Action Engine rule matrix.
- [ ] **Step 5 (Phase 4):** FastAPI endpoints (`/predict`, `/explain`, `/batch-risk`).
- [ ] **Step 6 (Phase 5):** Streamlit dashboard implementation (Dynamic uploader, priority table, waterfall plots, "What-If" simulator).
