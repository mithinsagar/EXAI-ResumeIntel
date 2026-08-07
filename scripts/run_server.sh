#!/usr/bin/env bash
# EXAI-ResumeIntel: Start the FastAPI backend
# Author: Mithin Sagar S

set -euo pipefail

HOST="${API_HOST:-0.0.0.0}"
PORT="${API_PORT:-8765}"
RELOAD="${API_RELOAD:-true}"

echo "[EXAI] Starting FastAPI server"
echo "        Host: $HOST"
echo "        Port: $PORT"
echo "        Docs: http://localhost:$PORT/docs"
echo

if [ "$RELOAD" = "true" ]; then
    exec uvicorn api.server:app --host "$HOST" --port "$PORT" --reload
else
    exec uvicorn api.server:app --host "$HOST" --port "$PORT"
fi
