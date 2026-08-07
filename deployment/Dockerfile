# EXAI-ResumeIntel: FastAPI backend Docker image
# Author: Mithin Sagar S
# GitHub: https://github.com/mithinsagar

FROM python:3.11-slim AS builder

WORKDIR /build

# System dependencies for numpy/scipy/scikit-learn
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# ---------- runtime stage ----------
FROM python:3.11-slim

LABEL org.opencontainers.image.title="EXAI-ResumeIntel"
LABEL org.opencontainers.image.description="Explainable AI Framework for Automated Resume Analysis"
LABEL org.opencontainers.image.authors="Mithin Sagar S <mithinsagar@gmail.com>"
LABEL org.opencontainers.image.source="https://github.com/mithinsagar/EXAI-ResumeIntel"
LABEL org.opencontainers.image.licenses="MIT"

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

WORKDIR /app

# Copy application code
COPY core/       ./core/
COPY xai/        ./xai/
COPY api/        ./api/
COPY config/     ./config/

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    API_HOST=0.0.0.0 \
    API_PORT=8765

EXPOSE 8765

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request, sys; \
        urllib.request.urlopen('http://localhost:8765/health', timeout=3); \
        sys.exit(0)" || exit 1

# Non-root user for security
RUN useradd --create-home --shell /bin/bash exai && chown -R exai:exai /app
USER exai

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8765"]
