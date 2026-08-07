#!/usr/bin/env bash
# EXAI-ResumeIntel: Train the semantic embedding engine
# Author: Mithin Sagar S

set -euo pipefail

echo "[EXAI] Training semantic embedding engine"
echo "        Corpus: data/raw/clean_resume_data.csv"
echo "        Output: models/embedding_engine.pkl"
echo

cd "$(dirname "$0")/.."
exec python -m training.train
