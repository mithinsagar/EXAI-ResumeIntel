"""
EXAI-ResumeIntel: XAI layer tests
==================================

Verifies:
    - Shapley value efficiency axiom (sum = grand coalition - empty)
    - LIME weight bounded response
    - Counterfactual gain positivity for missing skills
    - Interaction detection returns pairs
    - NL generator produces non-empty strings

Module: tests.test_explainer
Author: Mithin Sagar S
GitHub: https://github.com/mithinsagar
"""

from __future__ import annotations

import pytest

from xai.counterfactual import CounterfactualExplainer
from xai.explainer import XAICoordinator, weighted_match_score
from xai.interactions import FeatureInteractionDetector
from xai.lime_explainer import LIMEExplainer
from xai.nl_generator import NLExplainer
from xai.shapley import ShapleyExplainer


SKILLS = {
    "python": 0.9,
    "machine_learning": 0.85,
    "computer_vision": 0.75,
    "sql": 0.6,
    "leadership": 0.4,
}

REQS = {
    "python": 1.0,
    "machine_learning": 0.9,
    "computer_vision": 0.8,
    "sql": 0.5,
    "leadership": 0.3,
}


def score_fn(vec):
    return weighted_match_score(vec, REQS)


def test_shapley_efficiency_axiom():
    """Sum of Shapley values must equal v(F) - v(empty)."""
    explainer = ShapleyExplainer(exact_threshold=10)
    values = explainer.explain(SKILLS, score_fn, baseline=0.0)
    total = sum(values.values())
    v_full = score_fn(SKILLS)
    v_empty = score_fn({})
    assert abs(total - (v_full - v_empty)) < 1e-4


def test_shapley_summary_shape():
    """SHAP summary should return one entry per feature with required fields."""
    explainer = ShapleyExplainer(exact_threshold=10)
    values = explainer.explain(SKILLS, score_fn)
    summary = explainer.compute_shap_summary(values, SKILLS, REQS)
    assert len(summary) == len(SKILLS)
    for entry in summary:
        assert "shapley_value" in entry
        assert "display_name" in entry
        assert "impact_label" in entry


def test_lime_returns_top_n():
    """LIME should return exactly n_features entries."""
    explainer = LIMEExplainer(n_samples=100)
    results = explainer.explain(SKILLS, score_fn, n_features=3)
    assert len(results) == 3
    for entry in results:
        assert "lime_weight" in entry
        assert "direction" in entry


def test_counterfactual_positive_gain_for_missing():
    """Adding a fully missing skill should give a non-negative gain."""
    reduced_skills = {"python": 0.9}  # missing everything else
    explainer = CounterfactualExplainer()
    scenarios = explainer.explain(reduced_skills, REQS, score_fn)
    assert len(scenarios) > 0
    for scenario in scenarios[:3]:
        assert scenario["gain"] >= 0


def test_interactions_returns_pairs():
    """FeatureInteractionDetector should return pair labels."""
    detector = FeatureInteractionDetector()
    pairs = detector.detect(SKILLS, score_fn, top_pairs=3)
    assert len(pairs) <= 3
    for pair in pairs:
        assert "skill_a" in pair
        assert "skill_b" in pair
        assert pair["skill_a"] != pair["skill_b"]


def test_nl_generator_overall():
    """NL generator overall summary should be a non-empty string."""
    nl = NLExplainer()
    summary = nl.generate_overall(
        score=0.65,
        role="MACHINE LEARNING ENGINEER",
        strong_skills=["python", "machine_learning"],
        missing_skills=["kubernetes"],
        shap_top=[{
            "display_name": "Python",
            "shapley_value": 0.1,
            "contribution_pct": 25.0,
            "your_score": 90.0,
            "required_score": 100.0,
        }],
    )
    assert isinstance(summary, str)
    assert len(summary) > 20
    assert "Machine Learning Engineer" in summary


def test_xai_coordinator_bundle(ontology):
    """XAICoordinator should return a complete bundle with all keys."""
    coord = XAICoordinator()
    bundle = coord.explain(
        resume_text="python machine learning pytorch",
        extracted_skills=SKILLS,
        role_requirements=REQS,
        overall_score=0.55,
        role="MACHINE LEARNING ENGINEER",
        ontology_engine=ontology,
    )
    expected_keys = [
        "shap_summary", "lime_results", "counterfactuals",
        "feature_interactions", "skill_breakdown",
        "nl_overall", "nl_shap_insights", "role",
    ]
    for key in expected_keys:
        assert key in bundle
