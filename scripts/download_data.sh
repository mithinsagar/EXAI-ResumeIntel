#!/usr/bin/env bash
# EXAI-ResumeIntel: Download datasets and models from Hugging Face Hub
# Author: Mithin Sagar S
# GitHub: https://github.com/mithinsagar

set -euo pipefail

# Colours
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
RESET='\033[0m'

# ─── Configuration ─────────────────────────────────────────────────
HF_DATA_REPO="${HF_DATA_REPO:-mithinsagar/exai-resumeintel-data}"
HF_MODEL_REPO="${HF_MODEL_REPO:-mithinsagar/exai-resumeintel-models}"

DATA_DIR="data/raw"
MODELS_DIR="models"

# ─── Helpers ───────────────────────────────────────────────────────
log()  { echo -e "${BLUE}[EXAI]${RESET} $*"; }
ok()   { echo -e "${GREEN}[OK]${RESET}  $*"; }
warn() { echo -e "${YELLOW}[WARN]${RESET} $*"; }
die()  { echo -e "${RED}[ERROR]${RESET} $*" >&2; exit 1; }

# ─── Preflight ─────────────────────────────────────────────────────
log "EXAI-ResumeIntel data downloader"
log "Data repo:  $HF_DATA_REPO"
log "Model repo: $HF_MODEL_REPO"
echo

if ! command -v huggingface-cli >/dev/null 2>&1; then
    warn "huggingface-cli not installed. Installing..."
    pip install --quiet huggingface_hub || die "Failed to install huggingface_hub"
fi

mkdir -p "$DATA_DIR" "$MODELS_DIR"

# ─── Download datasets ─────────────────────────────────────────────
log "Downloading datasets from $HF_DATA_REPO"
if huggingface-cli download "$HF_DATA_REPO" \
     --repo-type dataset \
     --local-dir "$DATA_DIR" \
     --local-dir-use-symlinks False 2>/dev/null; then
    ok "Datasets downloaded to $DATA_DIR"
else
    warn "Dataset download failed. The dataset may not be published yet."
    warn "You can manually place clean_resume_data.csv and jobs_dataset_with_features.csv into $DATA_DIR/"
fi

echo

# ─── Download models ───────────────────────────────────────────────
log "Downloading models from $HF_MODEL_REPO"
if huggingface-cli download "$HF_MODEL_REPO" \
     --repo-type model \
     --local-dir "$MODELS_DIR" \
     --local-dir-use-symlinks False 2>/dev/null; then
    ok "Models downloaded to $MODELS_DIR"
else
    warn "Model download failed. The model may not be published yet."
    warn "You can generate the model locally by running: make train"
fi

echo
ok "Download complete."
