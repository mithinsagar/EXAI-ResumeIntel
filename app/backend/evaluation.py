"""
EXAI-ResumeIntel: Role-vs-resume semantic evaluation
====================================================

Module: app.backend.evaluation
Author: Mithin Sagar S
GitHub: https://github.com/mithinsagar
Institution: Vellore Institute of Technology (VIT)
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from .similarity import chunked_cosine_similarity


def evaluate_against_role(
    resume_text: str,
    target_role: str,
    job_df: pd.DataFrame,
    job_embeddings: np.memmap,
    model: SentenceTransformer,
) -> Dict[str, Any]:
    """
    Core evaluation logic (semantic similarity only).

    This function name is preserved for compatibility with existing pipelines.
    """
    # Robust role matching: normalize whitespace and case
    norm_roles = job_df["Role"].astype(str).str.strip().str.lower()
    target_norm = target_role.strip().lower()
    mask = norm_roles == target_norm
    if not mask.any():
        raise KeyError(f"Role '{target_role}' not found in job_df.")

    # Encode resume
    resume_vec = model.encode(
        [resume_text],
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=False,
    )[0]

    # Chunked cosine similarity against all jobs
    sims = chunked_cosine_similarity(resume_vec, job_embeddings, chunk_size=1024)

    # Restrict to rows matching the role (in case multiple postings per role)

    role_indices = np.where(mask.values)[0]
    role_sims = sims[role_indices]
    best_idx_local = int(role_sims.argmax())
    best_idx_global = int(role_indices[best_idx_local])

    best_similarity = float(role_sims[best_idx_local])
    match_percentage = max(min(best_similarity * 100.0, 100.0), 0.0)

    # Missing skills placeholder: expect an upstream process to attach them,
    # but here we simply leave it empty so that XAI module can derive them.
    result: Dict[str, Any] = {
        "match_percentage": match_percentage,
        "semantic_similarity": best_similarity,
        "best_job_index": best_idx_global,
        "role": target_role,
        "missing_skills": [],
    }
    return result


def evaluate_resume_for_role(
    resume_text: str,
    target_role: str,
    job_df: pd.DataFrame,
    job_embeddings: np.memmap,
    model: SentenceTransformer,
) -> Dict[str, Any]:
    """
    Thin wrapper around `evaluate_against_role` to keep a clean external API
    for the UI application.
    """
    return evaluate_against_role(
        resume_text=resume_text,
        target_role=target_role,
        job_df=job_df,
        job_embeddings=job_embeddings,
        model=model,
    )

