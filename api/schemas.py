"""
EXAI-ResumeIntel: Pydantic request/response schemas
====================================================

Defines the JSON schemas for all API endpoints. Auto-generates the Swagger
documentation at /docs and the ReDoc documentation at /redoc.

Module: api.schemas
Author: Mithin Sagar S
GitHub: https://github.com/mithinsagar
Institution: Vellore Institute of Technology (VIT)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ─── Health ────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Overall health status", examples=["ok"])
    model_loaded: bool = Field(
        ..., description="Whether the embedding model was loaded successfully"
    )
    version: str = Field(..., description="Application version")


# ─── Roles ─────────────────────────────────────────────────────────
class RolesResponse(BaseModel):
    """List of supported job roles."""

    roles: List[str] = Field(..., description="Sorted list of supported role slugs")
    count: int = Field(..., description="Total number of supported roles")


# ─── Analyze ───────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    """Payload for POST /analyze."""

    resume_text: str = Field(
        ...,
        min_length=50,
        description="Full resume text (minimum 50 characters)",
    )
    role: str = Field(
        ...,
        min_length=1,
        description="Target job role (must be a supported role slug)",
        examples=["MACHINE LEARNING ENGINEER"],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "resume_text": (
                    "Machine Learning Engineer with 5 years of experience in "
                    "PyTorch, YOLOv8, MLOps..."
                ),
                "role": "MACHINE LEARNING ENGINEER",
            }
        }
    }


class ScoreComponents(BaseModel):
    """Four-component score breakdown (paper eq. 7)."""

    ontology_score: float
    semantic_score: float
    depth_score: float
    corpus_score: float
    weights: Dict[str, float]


class SHAPSummaryItem(BaseModel):
    """A single row of the SHAP feature attribution summary."""

    skill: str
    display_name: str
    shapley_value: float
    contribution_pct: float
    your_score: float
    required_score: float
    gap: float
    direction: str
    impact_label: str


class LIMEResultItem(BaseModel):
    """A single row of the LIME local explanation."""

    skill: str
    display_name: str
    lime_weight: float
    your_score: float
    direction: str


class CounterfactualScenario(BaseModel):
    """A single counterfactual "what if" scenario."""

    type: Optional[str] = None
    skills_to_add: List[str]
    score_before: float
    score_after: float
    gain: float
    description: str
    priority: str


class ParsedResume(BaseModel):
    """Structured fields extracted from the raw resume."""

    years_experience: float
    education_level: int
    word_count: int
    has_quantified_achievements: bool


class AnalyzeResponse(BaseModel):
    """Full XAI bundle response for POST /analyze."""

    overall_score: float
    overall_score_raw: float
    score_components: ScoreComponents
    parsed_resume: ParsedResume
    shap_summary: List[SHAPSummaryItem]
    shap_values_raw: Dict[str, float]
    lime_results: List[LIMEResultItem]
    token_heatmap: List[Dict[str, Any]]
    highlighted_sentences: List[Dict[str, Any]]
    counterfactuals: List[CounterfactualScenario]
    feature_interactions: List[Dict[str, Any]]
    skill_breakdown: List[Dict[str, Any]]
    strong_skills: List[str]
    missing_skills: List[str]
    partial_skills: List[str]
    nl_overall: str
    nl_shap_insights: List[str]
    nl_lime_insight: str
    nl_counterfactual_insights: List[str]
    role: str
    n_skills_detected: int
    n_skills_required: int


class ErrorResponse(BaseModel):
    """Standard error envelope."""

    error: str
    detail: Optional[str] = None
