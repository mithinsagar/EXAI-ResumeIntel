"""
EXAI-ResumeIntel: Exact and Kernel Shapley value computation
=============================================================

Implements exact Shapley values for small feature sets (|F| <= 12) and
Kernel SHAP (sampling-based approximation) for larger sets. Satisfies all
four game-theoretic axioms: Efficiency, Symmetry, Dummy, and Additivity.

Corresponds to paper Section III.F (XAI Layer -- Shapley Value Attribution).

Module: xai.shapley
Author: Mithin Sagar S
GitHub: https://github.com/mithinsagar
Institution: Vellore Institute of Technology (VIT)
"""

from __future__ import annotations

import math
from collections import defaultdict
from itertools import combinations
from typing import Callable, Dict, List

import numpy as np


def _impact_label(val: float) -> str:
    """Map a Shapley value to a qualitative impact label."""
    if val > 0.15:
        return "Very High Impact"
    if val > 0.08:
        return "High Impact"
    if val > 0.03:
        return "Medium Impact"
    if val > 0.005:
        return "Low Impact"
    if val < -0.05:
        return "Hurting Score"
    return "Negligible"


class ShapleyExplainer:
    """
    Computes exact Shapley values for skill features.

    Shapley value for feature i = average marginal contribution of i across
    all possible subsets of features:

        phi_i = sum_{S} [|S|!(|F|-|S|-1)!/|F|!] * [v(S+i) - v(S)]

    For large feature sets, uses Kernel SHAP sampling approximation.

    Parameters
    ----------
    n_samples : int
        Number of permutations for Kernel SHAP approximation.
    exact_threshold : int
        Use exact SHAP when the number of features is at most this value.
    """

    def __init__(self, n_samples: int = 512, exact_threshold: int = 10):
        self.n_samples = n_samples
        self.exact_threshold = exact_threshold

    def explain(
        self,
        features: Dict[str, float],
        value_fn: Callable[[Dict[str, float]], float],
        baseline: float = 0.0,
    ) -> Dict[str, float]:
        """
        Compute Shapley values for each feature.

        Parameters
        ----------
        features : dict
            Feature -> score contribution mapping.
        value_fn : callable
            Scoring function taking a feature dict and returning a float.
        baseline : float
            Baseline score for the empty coalition.

        Returns
        -------
        dict
            Feature -> Shapley value mapping.
        """
        feature_names = list(features.keys())
        n = len(feature_names)
        if n == 0:
            return {}
        if n <= self.exact_threshold:
            return self._exact_shapley(feature_names, features, value_fn, baseline)
        return self._kernel_shapley(feature_names, features, value_fn, baseline)

    def _exact_shapley(
        self,
        feature_names: List[str],
        features: Dict[str, float],
        value_fn: Callable,
        baseline: float,
    ) -> Dict[str, float]:
        n = len(feature_names)
        shapley = {f: 0.0 for f in feature_names}

        for feat_i in feature_names:
            others = [f for f in feature_names if f != feat_i]
            for size in range(len(others) + 1):
                for subset in combinations(others, size):
                    s_without = {f: features[f] for f in subset}
                    s_with = {f: features[f] for f in subset}
                    s_with[feat_i] = features[feat_i]

                    v_without = value_fn(s_without)
                    v_with = value_fn(s_with)
                    marginal = v_with - v_without

                    s_size = len(subset)
                    weight = (
                        math.factorial(s_size)
                        * math.factorial(n - s_size - 1)
                        / math.factorial(n)
                    )
                    shapley[feat_i] += weight * marginal

        return shapley

    def _kernel_shapley(
        self,
        feature_names: List[str],
        features: Dict[str, float],
        value_fn: Callable,
        baseline: float,
    ) -> Dict[str, float]:
        """Sampling-based Shapley approximation (Kernel SHAP style)."""
        n = len(feature_names)
        shapley: Dict[str, float] = defaultdict(float)
        counts: Dict[str, int] = defaultdict(int)
        rng = np.random.RandomState(42)

        for _ in range(self.n_samples):
            perm = rng.permutation(n)
            current_features: Dict[str, float] = {}
            prev_value = value_fn(current_features)

            for idx in perm:
                feat = feature_names[idx]
                current_features[feat] = features[feat]
                new_value = value_fn(current_features)
                marginal = new_value - prev_value
                shapley[feat] += marginal
                counts[feat] += 1
                prev_value = new_value

        return {feat: shapley[feat] / max(counts[feat], 1) for feat in feature_names}

    def compute_shap_summary(
        self,
        shapley_values: Dict[str, float],
        features: Dict[str, float],
        role_requirements: Dict[str, float],
    ) -> List[Dict]:
        """
        Build a rich SHAP summary for display.

        Returns a sorted list of feature attributions with metadata:
        display name, contribution percentage, gap, direction, impact label.
        """
        total_positive = sum(v for v in shapley_values.values() if v > 0) or 1.0
        summary = []

        for skill, shap_val in shapley_values.items():
            skill_score = features.get(skill, 0.0)
            required = role_requirements.get(skill, 0.0)
            gap = max(0.0, required - skill_score)

            summary.append({
                "skill": skill,
                "display_name": skill.replace("_", " ").title(),
                "shapley_value": round(shap_val, 4),
                "contribution_pct": (
                    round((shap_val / total_positive) * 100, 1)
                    if total_positive > 0
                    else 0
                ),
                "your_score": round(skill_score * 100, 1),
                "required_score": round(required * 100, 1),
                "gap": round(gap * 100, 1),
                "direction": (
                    "positive" if shap_val > 0.01
                    else ("negative" if shap_val < -0.01 else "neutral")
                ),
                "impact_label": _impact_label(shap_val),
            })

        summary.sort(key=lambda x: abs(x["shapley_value"]), reverse=True)
        return summary
