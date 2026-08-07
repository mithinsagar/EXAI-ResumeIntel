"""
EXAI-ResumeIntel: Scoring engine tests
=======================================

Verifies:
    - ScoringEngine end-to-end pipeline
    - Component score bounds
    - Overall score computation
    - Parser field extraction

Module: tests.test_scorer
Author: Mithin Sagar S
GitHub: https://github.com/mithinsagar
"""

from __future__ import annotations

import pytest

from core.parser import ResumeParser
from core.scorer import ScoringEngine, ScoringResult


@pytest.fixture
def scorer(ontology):
    """Return a ScoringEngine without embeddings (semantic defaults to 0.5)."""
    return ScoringEngine(ontology, embedding_engine=None)


def test_parser_extracts_experience():
    """Parser should extract explicit years of experience."""
    parser = ResumeParser()
    result = parser.parse("5 years of experience with Python and PyTorch")
    assert result["years_experience"] == 5.0


def test_parser_default_experience():
    """Parser should default to 2 years when no years found."""
    parser = ResumeParser()
    result = parser.parse("Python developer with strong ML skills")
    assert result["years_experience"] == 2.0


def test_parser_detects_education():
    """Parser should detect and rank education levels."""
    parser = ResumeParser()
    result = parser.parse("PhD Computer Science and MBA Finance")
    assert result["education_level"] == 5  # PhD trumps MBA


def test_parser_detects_quantified_achievements():
    """Parser should detect quantified achievements."""
    parser = ResumeParser()
    result = parser.parse("Increased accuracy by 25% and reduced latency 65%")
    assert result["has_quantified_achievements"] is True


def test_scoring_end_to_end(scorer, ml_resume):
    """ScoringEngine.score should return a ScoringResult with all fields set."""
    result = scorer.score(ml_resume, "MACHINE LEARNING ENGINEER")
    assert isinstance(result, ScoringResult)
    assert 0.0 <= result.overall_score <= 1.0
    assert 0.0 <= result.ontology_score <= 1.0
    assert 0.0 <= result.semantic_score <= 1.0
    assert 0.0 <= result.depth_score <= 1.0
    assert 0.0 <= result.corpus_score <= 1.0


def test_scoring_bundle_contains_xai(scorer, ml_resume):
    """Result should include a complete XAI bundle."""
    result = scorer.score(ml_resume, "MACHINE LEARNING ENGINEER")
    bundle = result.xai_bundle
    assert "shap_summary" in bundle
    assert "lime_results" in bundle
    assert "counterfactuals" in bundle
    assert "feature_interactions" in bundle
    assert "nl_overall" in bundle
    assert "skill_breakdown" in bundle


def test_ml_resume_scores_high_for_ml_role(scorer, ml_resume):
    """An ML resume should score meaningfully well against the ML role."""
    result = scorer.score(ml_resume, "MACHINE LEARNING ENGINEER")
    assert result.overall_score > 0.30


def test_ml_resume_scores_low_for_finance_role(scorer, ml_resume):
    """An ML resume should score much lower against Finance."""
    result_ml = scorer.score(ml_resume, "MACHINE LEARNING ENGINEER")
    result_fin = scorer.score(ml_resume, "FINANCE")
    assert result_ml.overall_score > result_fin.overall_score


def test_score_components_sum_to_overall(scorer, ml_resume):
    """Weighted sum of components should equal the overall score."""
    result = scorer.score(ml_resume, "MACHINE LEARNING ENGINEER")
    weights = scorer.weights
    computed = (
        weights["ontology"] * result.ontology_score
        + weights["semantic"] * result.semantic_score
        + weights["depth"] * result.depth_score
        + weights["corpus"] * result.corpus_score
    )
    # Allow small float tolerance + cap
    assert abs(min(computed, 0.97) - result.overall_score) < 1e-3
