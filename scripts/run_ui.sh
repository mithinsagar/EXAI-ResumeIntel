#!/usr/bin/env bash
# EXAI-ResumeIntel: Serve the static web dashboard
# Author: Mithin Sagar S

set -euo pipefail

PORT="${UI_PORT:-8080}"

echo "[EXAI] Starting static UI on http://localhost:$PORT"
echo "        Landing page:  http://localhost:$PORT/landing.html"
echo "        Analyzer:      http://localhost:$PORT/index.html"
echo "        Requires:      FastAPI running on port 8765"
echo

cd "$(dirname "$0")/../ui"
exec python -m http.server "$PORT"
