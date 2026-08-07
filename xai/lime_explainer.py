"""
EXAI-ResumeIntel: LIME local interpretable model-agnostic explanation
======================================================================

Perturbs the skill presence vector around the input, scores each
perturbation, fits a local weighted ridge regression, and returns the
linear coefficients as local feature importance.

Corresponds to paper Section III.F (XAI Layer -- LIME Local Explanation).

Module: xai.lime_explainer
Author: Mithin Sagar S
GitHub: https://github.com/mithinsagar
Institution: Vellore Institute of Technology (VIT)
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List

import numpy as np


class LIMEExplainer:
    """
    LIME (Local Interpretable Model-Agnostic Explanations) for resume
    skill vectors.

    Generates N perturbations of the skill vector, weights them by proximity
    to the original, and fits a weighted ridge regression whose coefficients
    constitute the local explanation.

    Parameters
    ----------
    n_samples : int
        Number of perturbations.
    kernel_width : float
        Bandwidth of the exponential proximity kernel.
    ridge_alpha : float
        Regularisation strength for the local ridge regression.
    """

    def __init__(
        self,
        n_samples: int = 300,
        kernel_width: float = 0.25,
        ridge_alpha: float = 0.01,
    ):
        self.n_samples = n_samples
        self.kernel_width = kernel_width
        self.ridge_alpha = ridge_alpha

    def explain(
        self,
        skill_vector: Dict[str, float],
        score_fn: Callable[[Dict[str, float]], float],
        n_features: int = 10,
    ) -> List[Dict]:
        """
        Return the top-n features with their LIME weights.

        Parameters
        ----------
        skill_vector : dict
            Mapping from skill name to confidence.
        score_fn : callable
            Scoring function producing a scalar from a skill vector.
        n_features : int
            Number of top features to return.
        """
        feature_names = list(skill_vector.keys())
        n = len(feature_names)
        values = np.array([skill_vector[f] for f in feature_names], dtype=np.float32)

        if n == 0:
            return []

        rng = np.random.RandomState(42)

        X_perturb: List[np.ndarray] = []
        y_scores: List[float] = []
        weights: List[float] = []

        for _ in range(self.n_samples):
            mask = rng.binomial(1, 0.5, size=n).astype(np.float32)
            perturbed_values = values * mask
            perturbed_dict = {
                f: float(perturbed_values[i]) for i, f in enumerate(feature_names)
            }
            score = score_fn(perturbed_dict)
            distance = np.sqrt(np.sum((mask - np.ones(n)) ** 2))
            weight = self._kernel(distance)

            X_perturb.append(mask)
            y_scores.append(score)
            weights.append(weight)

        X = np.array(X_perturb)
        y = np.array(y_scores)
        w = np.array(weights)

        coefficients = self._weighted_ridge(X, y, w, alpha=self.ridge_alpha)

        explanations: List[Dict] = []
        for i, feat in enumerate(feature_names):
            explanations.append({
                "skill": feat,
                "display_name": feat.replace("_", " ").title(),
                "lime_weight": round(float(coefficients[i]), 4),
                "your_score": round(float(values[i]) * 100, 1),
                "direction": "positive" if coefficients[i] > 0 else "negative",
            })

        explanations.sort(key=lambda x: abs(x["lime_weight"]), reverse=True)
        return explanations[:n_features]

    def _kernel(self, distance: float) -> float:
        """Exponential proximity kernel."""
        return math.exp(-(distance ** 2) / (self.kernel_width ** 2))

    @staticmethod
    def _weighted_ridge(
        X: np.ndarray, y: np.ndarray, weights: np.ndarray, alpha: float = 0.01
    ) -> np.ndarray:
        """Closed-form weighted ridge: beta = (X^T W X + alpha I)^-1 X^T W y."""
        W = np.diag(weights)
        XtW = X.T @ W
        A = XtW @ X + alpha * np.eye(X.shape[1])
        b = XtW @ y
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(A, b, rcond=None)[0]
