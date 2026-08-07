"""
EXAI-ResumeIntel: Unified XAI coordinator
==========================================

Orchestrates all six XAI components (Shapley, LIME, attention, counterfactual,
interactions, natural language) and returns a single unified explanation
bundle for downstream consumers (API, UI, notebooks).

Module: xai.explainer
Author: Mithin Sagar S
GitHub: https://github.com/mithinsagar
Institution: Vellore Institute of Technology (VIT)
"""

from __future__ import annotations

from typing import Dict

from xai.attention import AttentionHeatmap
from xai.counterfactual import CounterfactualExplainer
from xai.interactions import FeatureInteractionDetector
from xai.lime_explainer import LIMEExplainer
from xai.nl_generator import NLExplainer
from xai.shapley import ShapleyExplainer


def weighted_match_score(
    skill_vec: Dict[str, float], requirements: Dict[str, float]
) -> float:
    """
    Core scoring function used by all XAI methods.

    Applies a non-linear coverage adjustment (paper eq. 8):

        S_ont = sum_s w_s * (1 - (1 - cov(s))^1.5) / sum_s w_s

    Being 80% there gives approximately 85% of the credit -- rewards near
    complete coverage disproportionately.
    """
    if not requirements:
        return 0.0
    total_weight = sum(requirements.values())
    score = 0.0
    for skill, req_weight in requirements.items():
        candidate_val = skill_vec.get(skill, 0.0)
        coverage = min(1.0, candidate_val / max(req_weight, 0.01))
        coverage_adj = 1.0 - (1.0 - coverage) ** 1.5
        score += req_weight * coverage_adj
    return score / total_weight


# Backward-compatibility alias for anyone importing the private name
_weighted_match_score = weighted_match_score


class XAICoordinator:
    """
    Orchestrates all XAI components and returns unified explanation bundle.

    This is the main entry point for the XAI layer.
    """

    def __init__(self):
        self.shapley = ShapleyExplainer(n_samples=512, exact_threshold=12)
        self.lime = LIMEExplainer(n_samples=300, kernel_width=0.25)
        self.heatmap = AttentionHeatmap()
        self.counterfactual = CounterfactualExplainer()
        self.interaction = FeatureInteractionDetector()
        self.nl = NLExplainer()

    def set_vectorizer(self, vectorizer) -> None:
        """Inject a fitted TF-IDF vectoriser for the attention heatmap."""
        self.heatmap.vectorizer = vectorizer

    def explain(
        self,
        resume_text: str,
        extracted_skills: Dict[str, float],
        role_requirements: Dict[str, float],
        overall_score: float,
        role: str,
        ontology_engine,
    ) -> Dict:
        """
        Build a full XAI explanation bundle for a single resume/role pair.

        Returns
        -------
        dict
            Contains SHAP summary + raw values, LIME results, attention heatmap,
            counterfactuals, feature interactions, natural language narrative,
            and per-skill breakdown.
        """

        def score_fn(skill_vec: Dict[str, float]) -> float:
            return weighted_match_score(skill_vec, role_requirements)

        # --- Shapley ---
        shap_vals = self.shapley.explain(extracted_skills, score_fn, baseline=0.0)
        shap_summary = self.shapley.compute_shap_summary(
            shap_vals, extracted_skills, role_requirements
        )

        # --- LIME ---
        lime_results = self.lime.explain(extracted_skills, score_fn, n_features=10)

        # --- Attention heatmap ---
        token_scores = self.heatmap.compute(
            resume_text, role_requirements, ontology_engine
        )
        highlighted_sentences = self.heatmap.get_highlighted_sentences(
            resume_text, token_scores, top_n=5
        )

        # --- Counterfactuals ---
        counterfactuals = self.counterfactual.explain(
            extracted_skills,
            role_requirements,
            score_fn,
            target_increase=0.1,
            max_skills_to_add=5,
        )

        # --- Interactions ---
        interactions = self.interaction.detect(extracted_skills, score_fn, top_pairs=5)

        # --- Strong / missing / partial skills ---
        strong_skills = [
            s for s, v in extracted_skills.items()
            if v >= role_requirements.get(s, 0.0) * 0.75
            and role_requirements.get(s, 0) > 0
        ]
        missing_skills = [
            s for s, w in role_requirements.items()
            if extracted_skills.get(s, 0.0) < w * 0.4
        ]
        partial_skills = [
            s for s, w in role_requirements.items()
            if 0.4 <= extracted_skills.get(s, 0.0) / max(w, 0.01) < 0.75
        ]

        # --- Natural language ---
        nl_overall = self.nl.generate_overall(
            overall_score, role, strong_skills, missing_skills, shap_summary
        )
        nl_shap = [self.nl.generate_shap_insight(s) for s in shap_summary[:5]]
        nl_lime = self.nl.generate_lime_insight(lime_results)
        nl_cf = [
            self.nl.generate_counterfactual_insight(cf) for cf in counterfactuals[:3]
        ]

        # --- Skill breakdown ---
        skill_breakdown = []
        for skill, req_weight in sorted(role_requirements.items(), key=lambda x: -x[1]):
            your_val = extracted_skills.get(skill, 0.0)
            skill_breakdown.append({
                "skill": skill,
                "display_name": skill.replace("_", " ").title(),
                "your_score": round(your_val * 100, 1),
                "required_score": round(req_weight * 100, 1),
                "gap": round(max(0, req_weight - your_val) * 100, 1),
                "coverage": round(min(1.0, your_val / max(req_weight, 0.01)) * 100, 1),
                "status": (
                    "strong" if your_val >= req_weight * 0.75
                    else "partial" if your_val >= req_weight * 0.4
                    else "missing"
                ),
                "shapley": round(shap_vals.get(skill, 0.0), 4),
            })

        return {
            "overall_score": round(overall_score * 100, 1),
            "overall_score_raw": overall_score,
            "shap_summary": shap_summary,
            "shap_values_raw": {k: round(v, 4) for k, v in shap_vals.items()},
            "lime_results": lime_results,
            "token_heatmap": token_scores[:100],
            "highlighted_sentences": highlighted_sentences,
            "counterfactuals": counterfactuals,
            "feature_interactions": interactions,
            "skill_breakdown": skill_breakdown,
            "strong_skills": strong_skills,
            "missing_skills": missing_skills,
            "partial_skills": partial_skills,
            "nl_overall": nl_overall,
            "nl_shap_insights": nl_shap,
            "nl_lime_insight": nl_lime,
            "nl_counterfactual_insights": nl_cf,
            "role": role,
            "n_skills_detected": len(extracted_skills),
            "n_skills_required": len(role_requirements),
        }
