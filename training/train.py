"""
EXAI-ResumeIntel: Train the semantic embedding engine
======================================================

Loads the resume corpus, fits the custom TF-IDF vectoriser (k=10000
features, 1-3 grams), then applies Truncated SVD (k=150 components,
7 power iterations) to produce 150-dimensional semantic embeddings.

Usage:
    python -m training.train

Outputs:
    models/embedding_engine.pkl

Module: training.train
Author: Mithin Sagar S
GitHub: https://github.com/mithinsagar
Institution: Vellore Institute of Technology (VIT)
"""

from __future__ import annotations

import sys
from pathlib import Path

from core.constants import (
    EMBEDDING_MODEL_PATH,
    MODELS_DIR,
    RESUME_CORPUS_PATH,
)
from core.scorer import ModelTrainer


def main() -> None:
    """Train and save the embedding engine."""
    print("=" * 60)
    print("  EXAI-ResumeIntel : Semantic Embedding Engine Training")
    print("  Author: Mithin Sagar S")
    print("=" * 60)

    if not RESUME_CORPUS_PATH.exists():
        print(f"[ERROR] Resume corpus not found: {RESUME_CORPUS_PATH}")
        print()
        print("Download the dataset from Hugging Face:")
        print("  bash scripts/download_data.sh")
        print()
        print("Or place `clean_resume_data.csv` in data/raw/ manually.")
        sys.exit(1)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    engine = ModelTrainer.train_from_csv(
        str(RESUME_CORPUS_PATH), str(EMBEDDING_MODEL_PATH)
    )

    print()
    print(f"[OK] Model trained and saved to {EMBEDDING_MODEL_PATH}")
    print(f"     Vocabulary size:  {len(engine.tfidf.vocabulary_)}")
    print(f"     Corpus size:      {len(engine.corpus_labels)}")
    print(f"     SVD components:   {engine.n_components}")
    print(f"     Explained var:    {engine.svd.explained_variance_ratio_.sum():.3f}")


if __name__ == "__main__":
    main()
