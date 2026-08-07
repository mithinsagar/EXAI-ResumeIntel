"""
EXAI-ResumeIntel: Multi-dimensional scoring engine
===================================================

Combines four interpretable signals into a single match score:
    - Ontology skill extraction score  (weight 0.45)
    - Semantic LSA similarity          (weight 0.30)
    - Skill depth / experience         (weight 0.15)
    - Corpus comparison to top resumes (weight 0.10)

Corresponds to paper Section III.E (Multi-Dimensional Scoring Engine).

Module: core.scorer
Author: Mithin Sagar S
GitHub: https://github.com/mithinsagar
Institution: Vellore Institute of Technology (VIT)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from core.constants import DEFAULT_DEPTH_WEIGHTS, DEFAULT_SCORING_WEIGHTS, SCORE_CAP
from core.embeddings import SemanticEmbeddingEngine
from core.ontology import SKILL_ONTOLOGY, OntologyEngine
from core.parser import ResumeParser
from xai.explainer import XAICoordinator, weighted_match_score


@dataclass
class ScoringResult:
    """Container for a single scoring run."""

    overall_score: float
    ontology_score: float
    semantic_score: float
    depth_score: float
    corpus_score: float
    extracted_skills: Dict[str, float]
    role_requirements: Dict[str, float]
    xai_bundle: Dict
    role: str
    resume_text: str


class ScoringEngine:
    """
    Main scoring engine combining ontology, semantic, depth, and corpus signals.

    Parameters
    ----------
    ontology_engine : OntologyEngine
        Skill ontology and extraction engine.
    embedding_engine : SemanticEmbeddingEngine, optional
        Trained TF-IDF + SVD embedding engine. If not provided, semantic
        and corpus scores fall back to defaults.
    weights : dict, optional
        Component weights. Defaults to paper eq. 7 values.
    """

    def __init__(
        self,
        ontology_engine: OntologyEngine,
        embedding_engine: Optional[SemanticEmbeddingEngine] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.ontology = ontology_engine
        self.embeddings = embedding_engine
        self.parser = ResumeParser()
        self.xai = XAICoordinator()
        self.weights = weights or DEFAULT_SCORING_WEIGHTS

        if embedding_engine and embedding_engine.is_fitted:
            self.xai.set_vectorizer(embedding_engine.tfidf)

    def score(self, resume_text: str, role: str) -> ScoringResult:
        """
        Full scoring pipeline for a resume against a target role.

        Parameters
        ----------
        resume_text : str
            Raw resume text.
        role : str
            Target job role (must be a key in ROLE_SKILL_WEIGHTS).

        Returns
        -------
        ScoringResult
            All component scores plus a complete XAI explanation bundle.
        """
        parsed = self.parser.parse(resume_text)
        clean_text = parsed["clean_text"]

        # 1. Ontology skill extraction
        extracted_skills = self.ontology.extract_skills(clean_text)
        role_requirements = self.ontology.get_role_requirements(role)

        # Adjust skill scores by experience
        exp_multiplier = min(1.0, 0.6 + parsed["years_experience"] * 0.05)
        adjusted_skills = {
            k: min(1.0, v * exp_multiplier) for k, v in extracted_skills.items()
        }

        # 2. Ontology component
        ontology_score = weighted_match_score(adjusted_skills, role_requirements)

        # 3. Semantic component
        semantic_score = 0.5
        if self.embeddings and self.embeddings.is_fitted:
            sem_sim = self.embeddings.semantic_role_similarity(clean_text, role)
            semantic_score = max(0.0, min(1.0, (sem_sim + 1) / 2))

        # 4. Depth component
        depth_score = self._compute_depth_score(clean_text, role_requirements, parsed)

        # 5. Corpus component
        corpus_score = 0.5
        if self.embeddings and self.embeddings.is_fitted:
            similar = self.embeddings.most_similar_in_corpus(
                clean_text, top_k=5, filter_label=role
            )
            if similar:
                corpus_score = float(np.mean([s[1] for s in similar]))
                corpus_score = max(0.0, min(1.0, (corpus_score + 1) / 2))

        # 6. Weighted overall
        overall_score = (
            self.weights["ontology"] * ontology_score
            + self.weights["semantic"] * semantic_score
            + self.weights["depth"] * depth_score
            + self.weights["corpus"] * corpus_score
        )
        overall_score = max(0.0, min(SCORE_CAP, overall_score))

        # 7. XAI bundle
        xai_bundle = self.xai.explain(
            resume_text=clean_text,
            extracted_skills=adjusted_skills,
            role_requirements=role_requirements,
            overall_score=overall_score,
            role=role,
            ontology_engine=self.ontology,
        )

        # Enrich with component breakdown
        xai_bundle["score_components"] = {
            "ontology_score": round(ontology_score * 100, 1),
            "semantic_score": round(semantic_score * 100, 1),
            "depth_score": round(depth_score * 100, 1),
            "corpus_score": round(corpus_score * 100, 1),
            "weights": self.weights,
        }
        xai_bundle["parsed_resume"] = {
            "years_experience": parsed["years_experience"],
            "education_level": parsed["education_level"],
            "word_count": parsed["word_count"],
            "has_quantified_achievements": parsed["has_quantified_achievements"],
        }

        return ScoringResult(
            overall_score=overall_score,
            ontology_score=ontology_score,
            semantic_score=semantic_score,
            depth_score=depth_score,
            corpus_score=corpus_score,
            extracted_skills=adjusted_skills,
            role_requirements=role_requirements,
            xai_bundle=xai_bundle,
            role=role,
            resume_text=resume_text,
        )

    def _compute_depth_score(
        self,
        text: str,
        role_requirements: Dict[str, float],
        parsed: Dict,
    ) -> float:
        """
        Depth measures how deeply the candidate knows their skills.

        Signals: experience years, education level, quantified achievements,
        number of related tools/frameworks mentioned.
        """
        text_lower = text.lower()

        exp_score = min(1.0, parsed["years_experience"] / 10.0)
        edu_score = parsed["education_level"] / 5.0
        quant_score = 0.8 if parsed["has_quantified_achievements"] else 0.3

        tool_mentions = 0
        total_tools = 0
        for skill, _weight in role_requirements.items():
            aliases = SKILL_ONTOLOGY.get(skill, {}).get("aliases", [])
            skill_tools = [a for a in aliases if len(a.split()) >= 1]
            total_tools += max(len(skill_tools), 1)
            mentions = sum(1 for t in skill_tools if t in text_lower)
            tool_mentions += min(mentions, 5)

        tool_density = min(1.0, tool_mentions / max(total_tools, 1) * 5)

        weights = DEFAULT_DEPTH_WEIGHTS
        score = (
            weights["experience"] * exp_score
            + weights["education"] * edu_score
            + weights["quantified"] * quant_score
            + weights["tool_density"] * tool_density
        )
        return round(score, 4)


class ModelTrainer:
    """
    Trains the semantic embedding engine on the resume corpus.
    """

    @staticmethod
    def train_from_csv(csv_path: str, model_save_path: str) -> SemanticEmbeddingEngine:
        """Load a corpus CSV, fit the embedding engine, and save it."""
        import csv

        print(f"[Trainer] Loading corpus from {csv_path}")
        documents = []
        labels = []

        with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row.get("Feature", row.get("Resume", "")).strip()
                label = row.get("Category", "").strip().upper()
                if text and label:
                    documents.append(text)
                    labels.append(label)

        print(f"[Trainer] {len(documents)} documents, {len(set(labels))} categories")

        engine = SemanticEmbeddingEngine(n_components=150)
        engine.fit(documents, labels)
        engine.save(model_save_path)
        return engine
