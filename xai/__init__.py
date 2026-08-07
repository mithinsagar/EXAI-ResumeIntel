"""
EXAI-ResumeIntel: Explainable AI layer
=======================================

Six-component XAI package:
    - shapley:         Exact and Kernel SHAP feature attribution
    - lime_explainer:  LIME local perturbation explanation
    - counterfactual:  What-if score scenarios
    - attention:       Token-level importance heatmap
    - interactions:    Feature interaction detection
    - nl_generator:    Natural language insight generator
    - explainer:       Unified coordinator returning the full XAI bundle

Author: Mithin Sagar S
GitHub: https://github.com/mithinsagar
Institution: Vellore Institute of Technology (VIT)
"""

from xai.attention import AttentionHeatmap
from xai.counterfactual import CounterfactualExplainer
from xai.explainer import XAICoordinator, weighted_match_score
from xai.interactions import FeatureInteractionDetector
from xai.lime_explainer import LIMEExplainer
from xai.nl_generator import NLExplainer
from xai.shapley import ShapleyExplainer

__all__ = [
    "AttentionHeatmap",
    "CounterfactualExplainer",
    "XAICoordinator",
    "weighted_match_score",
    "FeatureInteractionDetector",
    "LIMEExplainer",
    "NLExplainer",
    "ShapleyExplainer",
]
