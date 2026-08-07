"""
EXAI-ResumeIntel: SBERT and precomputed job embedding loader
============================================================

Module: app.backend.model_loader
Author: Mithin Sagar S
GitHub: https://github.com/mithinsagar
Institution: Vellore Institute of Technology (VIT)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


APP_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = APP_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


def _infer_embedding_shape(memmap_path: Path, n_rows: int) -> Tuple[int, int]:
    """
    Infer (n_rows, dim) for a float32 memmap from file size and dataframe rows.
    """
    file_size = memmap_path.stat().st_size
    bytes_per_row = file_size // max(n_rows, 1)
    if bytes_per_row == 0:
        raise ValueError("Memmap file appears to be empty.")
    dim = bytes_per_row // 4  # float32
    if dim * 4 * n_rows != file_size:
        # Fallback: assume dim=768 (all-mpnet-base-v2) if shape doesn't align perfectly
        dim = 768
    return n_rows, dim


def load_sbert_model() -> SentenceTransformer:
    """
    Load the SBERT model (cached by Streamlit in the caller).

    The model is not reloaded across interactions thanks to @st.cache_resource
    on the caller side.
    """
    model = SentenceTransformer(MODEL_NAME)
    return model


def load_job_data() -> tuple[pd.DataFrame, np.memmap]:
    """
    Load job dataframe and precomputed embeddings as a memory-mapped array.

    Returns
    -------
    (job_df, job_embeddings_memmap)
    """
    job_df_path = MODELS_DIR / "job_df.pkl"
    memmap_path = MODELS_DIR / "job_embeddings.memmap"

    if not job_df_path.exists():
        raise FileNotFoundError(f"job_df.pkl not found at {job_df_path}")
    if not memmap_path.exists():
        raise FileNotFoundError(f"job_embeddings.memmap not found at {memmap_path}")

    job_df = pd.read_pickle(job_df_path)
    if "Role" not in job_df.columns:
        raise KeyError("job_df.pkl must contain a 'Role' column.")

    n_rows = len(job_df)
    rows, dim = _infer_embedding_shape(memmap_path, n_rows)

    job_embeddings = np.memmap(
        memmap_path,
        dtype=np.float32,
        mode="r",
        shape=(rows, dim),
        order="C",
    )
    return job_df, job_embeddings

