#!/usr/bin/env bash
# EXAI-ResumeIntel: Launch the Streamlit application
# Author: Mithin Sagar S

set -euo pipefail

PORT="${STREAMLIT_SERVER_PORT:-8501}"

echo "[EXAI] Starting Streamlit application on http://localhost:$PORT"
echo "        Requires: models/job_df.pkl and models/job_embeddings.memmap"
echo

exec streamlit run app/main.py \
    --server.port "$PORT" \
    --server.address "${STREAMLIT_SERVER_ADDRESS:-0.0.0.0}"
