# Multi-stage production Dockerfile for Customer Churn Decision Engine
FROM python:3.11-slim as base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system build dependencies and curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -e .

# Copy application and model artifacts
COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/
COPY data/sample_uploads/ ./data/sample_uploads/
COPY Makefile ./

# Target 1: FastAPI REST API Service
FROM base as api
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Target 2: Streamlit Executive Dashboard
FROM base as dashboard
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "app/dashboard/main.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
