"""
EXAI-ResumeIntel: Counterfactual explanations for skill acquisition
====================================================================

Answers the actionable question: "What skills should I add to increase
my match score by X%?" Quantifies the exact score gain for each missing
or partially-covered skill and returns a prioritised list of scenarios.

Corresponds to paper Section III.F (XAI Layer -- Counterfactual Explainer).

Module: xai.counterfactual
Author: Mithin Sagar S
GitHub: https://github.com/mithinsagar
Institution: Vellore Institute of Technology (VIT)
"""

from __future__ import annotations

from typing import Callable, Dict, List


def _priority_label(gain: float) -> str:
    """Map a score gain to a priority label (ASCII-safe)."""
    if gain > 0.15:
        return "Top Priority"
    if gain > 0.08:
        return "High Priority"
    if gain > 0.04:
        return "Medium Priority"
    return "Low Priority"


class CounterfactualExplainer:
    """
    Generates counterfactual "what if" scenarios showing how much your
    score would increase if you acquired specific missing skills.
    """

    def explain(
        self,
        current_skills: Dict[str, float],
        role_requirements: Dict[str, float],
        score_fn: Callable[[Dict[str, float]], float],
        target_increase: float = 0.1,
        max_skills_to_add: int = 5,
    ) -> List[Dict]:
        """
        Return a prioritised list of counterfactual scenarios.

        Parameters
        ----------
        current_skills : dict
            Current skill confidence vector.
        role_requirements : dict
            Skill weights for the target role.
        score_fn : callable
            Scoring function (skill dict -> float in [0, 1]).
        target_increase : float
            Target score increase (informational only).
        max_skills_to_add : int
            Maximum number of scenarios to return.
        """
        current_score = score_fn(current_skills)
        missing_skills = {
            skill: weight
            for skill, weight in role_requirements.items()
            if current_skills.get(skill, 0.0) < weight * 0.5
        }

        scenarios: List[Dict] = []

        # Single-skill additions
        for skill, weight in missing_skills.items():
            augmented = dict(current_skills)
            augmented[skill] = weight
            new_score = score_fn(augmented)
            gain = new_score - current_score
            scenarios.append({
                "type": "add_single",
                "skills_to_add": [skill],
                "score_before": round(current_score * 100, 1),
                "score_after": round(new_score * 100, 1),
                "gain": round(gain * 100, 1),
                "description": f"Learn {skill.replace('_', ' ').title()}",
                "priority": _priority_label(gain),
            })

        # Two-skill combinations for the top four missing skills
        missing_sorted = sorted(missing_skills.keys(), key=lambda s: -missing_skills[s])
        for i in range(min(len(missing_sorted), 4)):
            for j in range(i + 1, min(len(missing_sorted), 4)):
                s1, s2 = missing_sorted[i], missing_sorted[j]
                augmented = dict(current_skills)
                augmented[s1] = missing_skills[s1]
                augmented[s2] = missing_skills[s2]
                new_score = score_fn(augmented)
                gain = new_score - current_score
                scenarios.append({
                    "type": "add_pair",
                    "skills_to_add": [s1, s2],
                    "score_before": round(current_score * 100, 1),
                    "score_after": round(new_score * 100, 1),
                    "gain": round(gain * 100, 1),
                    "description": (
                        f"Learn {s1.replace('_', ' ').title()} + "
                        f"{s2.replace('_', ' ').title()}"
                    ),
                    "priority": _priority_label(gain),
                })

        scenarios.sort(key=lambda x: -x["gain"])
        return scenarios[:max_skills_to_add]
