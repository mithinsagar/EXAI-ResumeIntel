"""
EXAI-ResumeIntel: Baseline comparison benchmark
================================================

Compares the full EXAI pipeline against four baseline approaches on the
same test resumes and roles, reproducing the paper's comparative study.

Baselines:
    B1 - Keyword Matching        (no semantics, no XAI)
    B2 - TF-IDF Cosine           (semantic only, no ontology)
    B3 - SBERT Cosine Similarity (transformer semantic, no ontology)
    B4 - Ontology Only           (no semantic component)

Full EXAI system combines ontology + semantic + depth + corpus with XAI.

Usage:
    python -m training.benchmark

Module: training.benchmark
Author: Mithin Sagar S
GitHub: https://github.com/mithinsagar
Institution: Vellore Institute of Technology (VIT)
"""

from __future__ import annotations

import sys

from core.constants import EMBEDDING_MODEL_PATH
from core.embeddings import SemanticEmbeddingEngine
from core.ontology import get_engine
from core.scorer import ScoringEngine
from xai.explainer import weighted_match_score


SAMPLE_ML_RESUME = """
Machine Learning Engineer with 5 years of experience building production
computer vision and NLP systems. Deep expertise in PyTorch, TensorFlow,
YOLOv8, COCO dataset, BERT, HuggingFace Transformers, and MLOps
(MLflow, Weights & Biases, Docker, Kubernetes, AWS SageMaker,
TorchServe). Reduced inference latency 65% with TensorRT + INT8
quantization. Implemented SHAP-based explainability dashboard for
business stakeholders.
"""


def _keyword_score(text: str, role: str) -> float:
    """Baseline 1: raw keyword hit-count."""
    text_lower = text.lower()
    keywords = ["python", "machine learning", "deep learning", "pytorch",
                "tensorflow", "computer vision", "nlp", "docker", "aws",
                "sql", "data science", "model deployment"]
    hits = sum(1 for kw in keywords if kw in text_lower)
    return min(1.0, hits / len(keywords))


def _tfidf_baseline_score(text: str, role: str, engine: SemanticEmbeddingEngine) -> float:
    """Baseline 2: TF-IDF cosine similarity to role centroid only."""
    if not engine or not engine.is_fitted:
        return 0.5
    sim = engine.semantic_role_similarity(text, role)
    return max(0.0, min(1.0, (sim + 1) / 2))


def _ontology_only_score(text: str, role: str, ontology) -> float:
    """Baseline 4: ontology component only, no semantic/depth/corpus."""
    skills = ontology.extract_skills(text)
    reqs = ontology.get_role_requirements(role)
    return weighted_match_score(skills, reqs)


def main() -> None:
    """Run comparative benchmark."""
    print("=" * 70)
    print("  EXAI-ResumeIntel : Comparative Baseline Benchmark")
    print("  Author: Mithin Sagar S")
    print("=" * 70)

    ontology = get_engine()
    try:
        embeddings = SemanticEmbeddingEngine.load(str(EMBEDDING_MODEL_PATH))
    except Exception:
        print(f"[WARN] Embedding engine not found; semantic scores default to 0.5")
        embeddings = None

    scorer = ScoringEngine(ontology, embeddings)

    role = "MACHINE LEARNING ENGINEER"
    text = SAMPLE_ML_RESUME.strip()

    print(f"\n[1] Test Case")
    print(f"    Role: {role}")
    print(f"    Resume: 5-year ML Engineer (CV + NLP + MLOps)")

    # Baseline scores
    kw_score = _keyword_score(text, role)
    tfidf_score = _tfidf_baseline_score(text, role, embeddings)
    ontology_score = _ontology_only_score(text, role, ontology)
    exai_result = scorer.score(text, role)

    print(f"\n[2] Comparative Results")
    print(f"    {'Method':<35} {'Score':>8} {'Delta vs EXAI':>15}")
    print(f"    {'-' * 60}")
    print(f"    {'B1: Keyword Matching':<35} "
          f"{kw_score * 100:>7.1f}% {(kw_score - exai_result.overall_score) * 100:>+14.1f}%")
    print(f"    {'B2: TF-IDF Cosine (semantic-only)':<35} "
          f"{tfidf_score * 100:>7.1f}% {(tfidf_score - exai_result.overall_score) * 100:>+14.1f}%")
    print(f"    {'B3: Ontology only':<35} "
          f"{ontology_score * 100:>7.1f}% {(ontology_score - exai_result.overall_score) * 100:>+14.1f}%")
    print(f"    {'EXAI Full System':<35} "
          f"{exai_result.overall_score * 100:>7.1f}% {'-':>15}")

    print(f"\n[3] EXAI Score Component Breakdown")
    print(f"    Ontology (45%):  {exai_result.ontology_score * 100:.1f}%")
    print(f"    Semantic (30%):  {exai_result.semantic_score * 100:.1f}%")
    print(f"    Depth (15%):     {exai_result.depth_score * 100:.1f}%")
    print(f"    Corpus (10%):    {exai_result.corpus_score * 100:.1f}%")

    print(f"\n[OK] Benchmark complete.")


if __name__ == "__main__":
    main()
