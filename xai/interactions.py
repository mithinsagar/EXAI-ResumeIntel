"""
EXAI-ResumeIntel: Feature interaction detector
===============================================

Detects synergies between skills using Shapley interaction values:
pairs of skills that together contribute more (positive synergy) or less
(negative synergy) than the sum of their individual contributions.

    phi_ij = v(both) - v(only i) - v(only j) + v(neither)

Module: xai.interactions
Author: Mithin Sagar S
GitHub: https://github.com/mithinsagar
Institution: Vellore Institute of Technology (VIT)
"""

from __future__ import annotations

from typing import Callable, Dict, List


class FeatureInteractionDetector:
    """
    Detects synergies between skills (skills that together boost score more
    than individually) using Shapley interaction values.
    """

    def detect(
        self,
        skills: Dict[str, float],
        score_fn: Callable[[Dict[str, float]], float],
        top_pairs: int = 5,
        max_skills: int = 8,
    ) -> List[Dict]:
        """
        Return top-K skill pairs by absolute interaction magnitude.

        Parameters
        ----------
        skills : dict
            Current skill confidence vector.
        score_fn : callable
            Scoring function.
        top_pairs : int
            Maximum number of pairs to return.
        max_skills : int
            Only the first max_skills features are considered (cost control).
        """
        skill_names = list(skills.keys())
        n = len(skill_names)
        baseline_with_all = score_fn(skills)

        interactions: List[Dict] = []

        for i in range(min(n, max_skills)):
            for j in range(i + 1, min(n, max_skills)):
                s1, s2 = skill_names[i], skill_names[j]
                without_both = {k: v for k, v in skills.items() if k not in [s1, s2]}
                without_s1 = {k: v for k, v in skills.items() if k != s1}
                without_s2 = {k: v for k, v in skills.items() if k != s2}

                v_none = score_fn(without_both)
                v_s1 = score_fn(without_s2)   # only s1 present
                v_s2 = score_fn(without_s1)   # only s2 present
                v_both = baseline_with_all

                interaction = v_both - v_s1 - v_s2 + v_none
                interactions.append({
                    "skill_a": s1,
                    "skill_b": s2,
                    "interaction": round(interaction, 4),
                    "synergy": (
                        "positive" if interaction > 0.01
                        else ("negative" if interaction < -0.01 else "neutral")
                    ),
                    "label": (
                        f"{s1.replace('_', ' ').title()} x "
                        f"{s2.replace('_', ' ').title()}"
                    ),
                })

        interactions.sort(key=lambda x: -abs(x["interaction"]))
        return interactions[:top_pairs]
