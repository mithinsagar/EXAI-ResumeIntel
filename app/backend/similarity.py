"""
EXAI-ResumeIntel: Cosine similarity and fuzzy role autocomplete
===============================================================

Module: app.backend.similarity
Author: Mithin Sagar S
GitHub: https://github.com/mithinsagar
Institution: Vellore Institute of Technology (VIT)
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
from rapidfuzz import fuzz, process as rf_process


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def chunked_cosine_similarity(
    query_vec: np.ndarray,
    memmap_matrix: np.memmap,
    chunk_size: int = 1024,
) -> np.ndarray:
    """
    Compute cosine similarity between query_vec and each row of memmap_matrix
    in chunks to avoid loading the entire matrix into RAM.
    """
    n_rows = memmap_matrix.shape[0]
    sims = np.empty(n_rows, dtype=np.float32)

    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)
        batch = np.array(memmap_matrix[start:end], copy=False)
        num = np.dot(batch, query_vec)
        denom = (
            np.linalg.norm(batch, axis=1) * (np.linalg.norm(query_vec) + 1e-8)
        ) + 1e-8
        sims[start:end] = num / denom
    return sims


def suggest_roles(
    query: str, roles: Iterable[str], max_suggestions: int = 1000
) -> List[str]:
    """
    Intelligent autocomplete for roles using partial and fuzzy matching.
    """
    q = query.strip().lower()
    if not q:
        return []

    # First pass: simple case-insensitive substring filter
    filtered = [r for r in roles if q in r.lower()]

    # If not enough, augment with fuzzy matches
    if len(filtered) < max_suggestions:
        candidates = list(set(roles) - set(filtered))
        if candidates:
            fuzzy = rf_process.extract(
                query,
                candidates,
                scorer=fuzz.token_set_ratio,
                limit=max_suggestions,
            )
            for role, score, _ in fuzzy:
                if score >= 60 and role not in filtered:
                    filtered.append(role)

    # De-duplicate and truncate
    seen = set()
    deduped: List[str] = []
    for r in filtered:
        if r not in seen:
            deduped.append(r)
            seen.add(r)
        if len(deduped) >= max_suggestions:
            break
    return deduped

