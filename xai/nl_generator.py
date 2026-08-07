"""
EXAI-ResumeIntel: Natural-language explanation generator
=========================================================

Generates human-readable narrative explanations from the computed SHAP,
LIME, and counterfactual values. Powers the "in plain English" panels of
the dashboard.

Module: xai.nl_generator
Author: Mithin Sagar S
GitHub: https://github.com/mithinsagar
Institution: Vellore Institute of Technology (VIT)
"""

from __future__ import annotations

from typing import Dict, List


class NLExplainer:
    """Generates human-readable XAI explanations from computed values."""

    def generate_overall(
        self,
        score: float,
        role: str,
        strong_skills: List[str],
        missing_skills: List[str],
        shap_top: List[Dict],
    ) -> str:
        """Return the top-level narrative summary for the dashboard."""
        score_pct = round(score * 100, 1)
        role_clean = role.replace("-", " ").replace("_", " ").title()

        if score_pct >= 80:
            strength = "an excellent"
        elif score_pct >= 65:
            strength = "a strong"
        elif score_pct >= 50:
            strength = "a moderate"
        elif score_pct >= 35:
            strength = "a developing"
        else:
            strength = "an early-stage"

        top_skill = shap_top[0]["display_name"] if shap_top else "your skills"
        top_contrib = shap_top[0]["contribution_pct"] if shap_top else 0

        lines = [
            f"You have {strength} fit ({score_pct}%) for the {role_clean} role.",
            "",
            (
                f"Your strongest contributor is {top_skill}, accounting for "
                f"{top_contrib:.1f}% of your match score."
            ),
        ]

        if strong_skills:
            names = ", ".join(s.replace("_", " ").title() for s in strong_skills[:4])
            lines.append(f"Skills working in your favor: {names}.")

        if missing_skills:
            names = ", ".join(s.replace("_", " ").title() for s in missing_skills[:4])
            lines.append(f"Key gaps to address: {names}.")

        return "\n".join(lines)

    def generate_shap_insight(self, shap_entry: Dict) -> str:
        """Return a one-sentence insight for a single SHAP entry."""
        name = shap_entry["display_name"]
        val = shap_entry["shapley_value"]
        pct = shap_entry["contribution_pct"]
        your = shap_entry["your_score"]
        req = shap_entry["required_score"]

        if val > 0.05:
            return (
                f"{name} is a strong asset -- it contributes +{pct:.1f}% to your score "
                f"(you: {your:.0f}%, required: {req:.0f}%)."
            )
        if val > 0:
            return (
                f"{name} contributes modestly (+{pct:.1f}%) -- "
                "there is room to strengthen it further."
            )
        if val < -0.02:
            return (
                f"{name} is below the threshold for this role -- "
                "improving it could recover score."
            )
        return f"{name} has minimal impact on your current score."

    def generate_lime_insight(self, lime_results: List[Dict]) -> str:
        """Return a summary sentence combining top positive and negative LIME features."""
        pos = [r for r in lime_results if r["direction"] == "positive"][:3]
        neg = [r for r in lime_results if r["direction"] == "negative"][:2]
        parts = []
        if pos:
            names = ", ".join(r["display_name"] for r in pos)
            parts.append(f"Locally, {names} have the strongest positive influence.")
        if neg:
            names = ", ".join(r["display_name"] for r in neg)
            parts.append(
                f"{names} may be dragging the local score -- consider addressing them."
            )
        return " ".join(parts)

    def generate_counterfactual_insight(self, cf: Dict) -> str:
        """Return an action sentence for a single counterfactual scenario."""
        skills = " + ".join(s.replace("_", " ").title() for s in cf["skills_to_add"])
        return (
            f"If you add {skills}, your score would rise from "
            f"{cf['score_before']}% -> {cf['score_after']}% "
            f"(+{cf['gain']}%). {cf['priority']}"
        )
