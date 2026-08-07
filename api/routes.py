"""
EXAI-ResumeIntel: API route definitions
========================================

Defines the endpoints of the FastAPI backend:
    GET  /health   -- health check
    GET  /roles    -- list supported job roles
    POST /analyze  -- full XAI analysis

Module: api.routes
Author: Mithin Sagar S
GitHub: https://github.com/mithinsagar
Institution: Vellore Institute of Technology (VIT)
"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException

from api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    HealthResponse,
    RolesResponse,
)
from core.constants import APP_VERSION, EMBEDDING_MODEL_PATH
from core.embeddings import SemanticEmbeddingEngine
from core.ontology import get_engine
from core.scorer import ScoringEngine

log = logging.getLogger("exai.api")

router = APIRouter()

# ─── Lazy engine loading ───────────────────────────────────────────
_ontology = None
_embeddings: Optional[SemanticEmbeddingEngine] = None
_scorer: Optional[ScoringEngine] = None


def _get_scorer() -> ScoringEngine:
    """Lazily construct the scoring engine singleton."""
    global _ontology, _embeddings, _scorer

    if _scorer is not None:
        return _scorer

    log.info("Initialising ontology engine")
    _ontology = get_engine()

    if Path(EMBEDDING_MODEL_PATH).exists():
        log.info("Loading embedding engine from %s", EMBEDDING_MODEL_PATH)
        try:
            _embeddings = SemanticEmbeddingEngine.load(str(EMBEDDING_MODEL_PATH))
        except Exception as exc:
            log.warning("Failed to load embedding engine: %s", exc)
            _embeddings = None
    else:
        log.warning(
            "Embedding model not found at %s -- semantic/corpus scores will "
            "default to 0.5. Run `python -m training.train` to train.",
            EMBEDDING_MODEL_PATH,
        )
        _embeddings = None

    _scorer = ScoringEngine(_ontology, _embeddings)
    log.info("Scoring engine ready")
    return _scorer


@router.get("/health", response_model=HealthResponse, tags=["Meta"])
def health() -> HealthResponse:
    """Return the health status and whether the embedding model was loaded."""
    scorer = _get_scorer()
    return HealthResponse(
        status="ok",
        model_loaded=scorer.embeddings is not None
        and scorer.embeddings.is_fitted,
        version=APP_VERSION,
    )


@router.get("/roles", response_model=RolesResponse, tags=["Meta"])
def list_roles() -> RolesResponse:
    """List all supported job roles."""
    scorer = _get_scorer()
    roles = scorer.ontology.get_all_roles()
    return RolesResponse(roles=roles, count=len(roles))


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    tags=["Analysis"],
    summary="Analyze a resume against a target role",
)
def analyze(payload: AnalyzeRequest) -> AnalyzeResponse:
    """
    Run the full EXAI pipeline on a resume/role pair.

    Returns the complete XAI bundle including SHAP, LIME, counterfactuals,
    attention heatmap, skill breakdown, and natural-language insights.
    """
    scorer = _get_scorer()
    try:
        result = scorer.score(payload.resume_text, payload.role)
    except Exception as exc:
        log.error("Analysis failed: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return AnalyzeResponse(**result.xai_bundle)
