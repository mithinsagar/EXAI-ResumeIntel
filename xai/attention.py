"""
EXAI-ResumeIntel: Token-level attention heatmap generator
==========================================================

Generates a token-level importance map from the ontology alias matches and
TF-IDF weights. Highlights the exact parts of the resume text that
contribute most to the match score.

Module: xai.attention
Author: Mithin Sagar S
GitHub: https://github.com/mithinsagar
Institution: Vellore Institute of Technology (VIT)
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional


SKILL_ONTOLOGY_WEIGHTS: Dict[str, float] = {
    "machine_learning": 1.0,
    "computer_vision": 0.95,
    "nlp": 0.95,
    "deep_learning_frameworks": 0.9,
    "data_science": 1.0,
    "python": 0.9,
    "software_engineering": 1.0,
    "web_development": 0.9,
    "databases": 0.85,
    "cybersecurity": 1.0,
    "model_deployment": 0.9,
    "accounting": 0.95,
    "finance": 1.0,
    "banking": 0.95,
    "healthcare": 1.0,
    "legal": 1.0,
    "hr": 1.0,
    "sales": 1.0,
    "marketing": 1.0,
    "mechanical_engineering": 1.0,
    "civil_engineering": 1.0,
    "design": 1.0,
    "leadership": 0.7,
    "communication": 0.6,
    "project_management": 0.75,
    "feature_engineering": 0.85,
    "recommendation_systems": 0.85,
    "time_series": 0.85,
}


def _importance_level(score: float) -> str:
    """Map an importance score to a qualitative level."""
    if score > 0.7:
        return "critical"
    if score > 0.4:
        return "high"
    if score > 0.2:
        return "medium"
    if score > 0.05:
        return "low"
    return "none"


class AttentionHeatmap:
    """
    Produces a token-level attention/importance map from ontology matches
    and TF-IDF weights.

    Parameters
    ----------
    tfidf_vectorizer : object, optional
        Fitted TF-IDF vectoriser exposing `idf_` and `vocabulary_`.
    """

    def __init__(self, tfidf_vectorizer: Optional[object] = None):
        self.vectorizer = tfidf_vectorizer

    def compute(
        self,
        resume_text: str,
        role_skill_weights: Dict[str, float],
        ontology_engine,
    ) -> List[Dict]:
        """
        Return a list of token importance records for the resume text.

        Each record contains: token, position, importance, level.
        """
        text = re.sub(r"[^a-z0-9\s]", " ", resume_text.lower())
        tokens = text.split()
        token_scores: Dict[int, float] = {}

        # Score tokens by ontology match (up to trigrams)
        for i, _token in enumerate(tokens):
            for n in [1, 2, 3]:
                if i + n <= len(tokens):
                    phrase = " ".join(tokens[i : i + n])
                    if phrase in ontology_engine.alias_map:
                        skill = ontology_engine.alias_map[phrase]
                        role_weight = role_skill_weights.get(skill, 0.0)
                        for j in range(i, i + n):
                            token_scores[j] = max(
                                token_scores.get(j, 0.0),
                                role_weight * SKILL_ONTOLOGY_WEIGHTS.get(skill, 0.7),
                            )

        # TF-IDF fallback for unmatched tokens
        if self.vectorizer and hasattr(self.vectorizer, "idf_"):
            max_idf = self.vectorizer.idf_.max()
            for i, token in enumerate(tokens):
                if i not in token_scores and token in self.vectorizer.vocabulary_:
                    idx = self.vectorizer.vocabulary_[token]
                    idf = self.vectorizer.idf_[idx]
                    token_scores[i] = min(1.0, idf / max_idf * 0.3)

        result = []
        for i, token in enumerate(tokens):
            if len(token) > 2:
                importance = token_scores.get(i, 0.0)
                result.append({
                    "token": token,
                    "position": i,
                    "importance": round(importance, 3),
                    "level": _importance_level(importance),
                })

        return result

    def get_highlighted_sentences(
        self,
        resume_text: str,
        token_scores: List[Dict],
        top_n: int = 5,
    ) -> List[Dict]:
        """Return the top-N most impactful sentences from the resume."""
        sentences = re.split(r"[.!?;]", resume_text)
        scored_sentences = []
        score_map = {t["token"]: t["importance"] for t in token_scores}

        for sent in sentences:
            tokens = sent.lower().split()
            if len(tokens) < 3:
                continue
            score = sum(score_map.get(t, 0.0) for t in tokens) / max(len(tokens), 1)
            scored_sentences.append({
                "sentence": sent.strip()[:200],
                "score": round(score, 3),
                "level": _importance_level(score),
            })

        scored_sentences.sort(key=lambda x: -x["score"])
        return scored_sentences[:top_n]
