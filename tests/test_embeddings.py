"""
EXAI-ResumeIntel: Embedding engine tests
=========================================

Verifies:
    - TFIDFVectorizer fit and transform shapes
    - TruncatedSVD dimensionality reduction
    - SemanticEmbeddingEngine end-to-end pipeline
    - Cosine similarity monotonicity

Module: tests.test_embeddings
Author: Mithin Sagar S
GitHub: https://github.com/mithinsagar
"""

from __future__ import annotations

import numpy as np
import pytest

from core.embeddings import SemanticEmbeddingEngine, TFIDFVectorizer, TruncatedSVD


CORPUS = [
    "Python machine learning PyTorch deep learning computer vision",
    "SQL databases postgresql mongodb data engineering pipelines",
    "React nodejs javascript typescript frontend web development",
    "Finance DCF valuation equity research bloomberg financial modeling",
    "Machine learning NLP BERT transformers HuggingFace language models",
    "Python data science pandas numpy scikit-learn analytics",
]

LABELS = [
    "MACHINE LEARNING ENGINEER",
    "DATA ENGINEER",
    "FRONTEND DEVELOPER",
    "FINANCE",
    "MACHINE LEARNING ENGINEER",
    "DATA SCIENTIST",
]


def test_tfidf_fit_and_transform():
    """TFIDFVectorizer should produce L2-normalised sparse vectors."""
    vec = TFIDFVectorizer(max_features=100, ngram_range=(1, 2), min_df=1)
    X = vec.fit_transform(CORPUS)
    assert X.shape[0] == len(CORPUS)
    assert X.shape[1] <= 100
    # Rows are L2 normalised
    norms = np.linalg.norm(X, axis=1)
    assert np.allclose(norms[norms > 0], 1.0, atol=1e-4)


def test_svd_reduces_dimensionality():
    """TruncatedSVD should reduce feature count to n_components."""
    rng = np.random.RandomState(42)
    X = rng.randn(20, 100).astype(np.float32)
    svd = TruncatedSVD(n_components=10, n_iter=3)
    X_reduced = svd.fit_transform(X)
    assert X_reduced.shape == (20, 10)


def test_full_engine_pipeline():
    """SemanticEmbeddingEngine.fit + embed should return shape (n_components,)."""
    engine = SemanticEmbeddingEngine(n_components=5)
    engine.tfidf = TFIDFVectorizer(max_features=50, ngram_range=(1, 1), min_df=1)
    engine.svd = TruncatedSVD(n_components=5, n_iter=3)
    engine.fit(CORPUS, LABELS)
    assert engine.is_fitted

    vec = engine.embed("Python machine learning deep learning")
    assert vec.shape == (5,)


def test_cosine_similarity_symmetric():
    """Cosine similarity should be symmetric and in [-1, 1]."""
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    sim_ab = SemanticEmbeddingEngine.cosine_similarity(a, b)
    sim_ba = SemanticEmbeddingEngine.cosine_similarity(b, a)
    assert sim_ab == sim_ba
    assert -1.0 <= sim_ab <= 1.0


def test_cosine_self_similarity_is_one():
    """A vector should have cosine similarity 1 with itself."""
    v = np.array([1.0, 2.0, 3.0])
    assert abs(SemanticEmbeddingEngine.cosine_similarity(v, v) - 1.0) < 1e-6


def test_zero_vector_similarity():
    """Cosine similarity with a zero vector should return 0."""
    a = np.zeros(5)
    b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert SemanticEmbeddingEngine.cosine_similarity(a, b) == 0.0
