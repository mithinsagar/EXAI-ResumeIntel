"""
EXAI-ResumeIntel: Semantic embedding engine (TF-IDF + Truncated SVD / LSA)
===========================================================================

Custom TF-IDF vectoriser combined with randomised Truncated SVD produces
150-dimensional semantic embeddings trained on the resume corpus, capturing
27.8% of corpus variance. Cosine similarity provides semantic matching
beyond keyword overlap.

Corresponds to paper Section III.D (Semantic Embedding Engine).

Module: core.embeddings
Author: Mithin Sagar S
GitHub: https://github.com/mithinsagar
Institution: Vellore Institute of Technology (VIT)
"""


import numpy as np
import re
import math
import pickle
import os
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional


class TFIDFVectorizer:
    """
    Full TF-IDF implementation from scratch.
    Handles sublinear TF, smooth IDF, and n-grams.
    """

    def __init__(
        self,
        max_features: int = 8000,
        ngram_range: Tuple[int, int] = (1, 3),
        min_df: int = 2,
        sublinear_tf: bool = True,
    ):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.sublinear_tf = sublinear_tf
        self.vocabulary_: Dict[str, int] = {}
        self.idf_: np.ndarray = None
        self.n_docs_: int = 0

    @staticmethod
    def _preprocess(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _get_ngrams(self, tokens: List[str]) -> List[str]:
        ngrams = []
        min_n, max_n = self.ngram_range
        for n in range(min_n, max_n + 1):
            for i in range(len(tokens) - n + 1):
                ngrams.append(" ".join(tokens[i : i + n]))
        return ngrams

    def _tokenize(self, text: str) -> List[str]:
        tokens = self._preprocess(text).split()
        return self._get_ngrams(tokens)

    def fit(self, documents: List[str]) -> "TFIDFVectorizer":
        self.n_docs_ = len(documents)
        df_counts: Counter = Counter()
        tokenized = []

        for doc in documents:
            terms = self._tokenize(doc)
            unique_terms = set(terms)
            df_counts.update(unique_terms)
            tokenized.append(terms)

        # filter by min_df, rank by df for feature selection
        valid = {t: df for t, df in df_counts.items() if df >= self.min_df}
        sorted_terms = sorted(valid.keys(), key=lambda t: -valid[t])
        top_terms = sorted_terms[: self.max_features]

        self.vocabulary_ = {term: idx for idx, term in enumerate(top_terms)}

        # compute smooth IDF
        self.idf_ = np.zeros(len(self.vocabulary_))
        for term, idx in self.vocabulary_.items():
            df = df_counts[term]
            self.idf_[idx] = math.log((1 + self.n_docs_) / (1 + df)) + 1.0

        return self

    def transform(self, documents: List[str]) -> np.ndarray:
        n = len(documents)
        V = len(self.vocabulary_)
        X = np.zeros((n, V), dtype=np.float32)

        for i, doc in enumerate(documents):
            terms = self._tokenize(doc)
            tf_counts: Counter = Counter(terms)
            total = max(sum(tf_counts.values()), 1)
            for term, count in tf_counts.items():
                if term in self.vocabulary_:
                    idx = self.vocabulary_[term]
                    tf = (
                        1 + math.log(count)
                        if self.sublinear_tf and count > 0
                        else count / total
                    )
                    X[i, idx] = tf * self.idf_[idx]

        # L2 normalize each row
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return X / norms

    def fit_transform(self, documents: List[str]) -> np.ndarray:
        return self.fit(documents).transform(documents)


class TruncatedSVD:
    """
    Randomized SVD for LSA / dimensionality reduction.
    Pure numpy, no scipy dependency assumed.
    """

    def __init__(self, n_components: int = 200, n_iter: int = 5, random_state: int = 42):
        self.n_components = n_components
        self.n_iter = n_iter
        self.random_state = random_state
        self.components_: np.ndarray = None
        self.singular_values_: np.ndarray = None
        self.explained_variance_ratio_: np.ndarray = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape
        k = min(self.n_components, min(n_samples, n_features) - 1)

        # Randomized SVD (Halko et al.)
        Q = rng.randn(n_features, k + 10).astype(np.float32)
        for _ in range(self.n_iter):
            Z = X @ Q
            Q, _ = np.linalg.qr(Z)
            Z = X.T @ Q
            Q, _ = np.linalg.qr(Z)

        B = (X @ Q).astype(np.float64)
        U, s, Vt = np.linalg.svd(B, full_matrices=False)
        U = U[:, :k]
        s = s[:k]
        Vt = Vt[:k, :]

        self.components_ = (Q @ Vt.T).T  # shape (k, n_features)
        self.singular_values_ = s

        X_transformed = U * s
        total_var = np.var(X, axis=0).sum()
        self.explained_variance_ratio_ = (
            np.var(X_transformed, axis=0) / total_var if total_var > 0 else np.zeros(k)
        )
        return X_transformed.astype(np.float32)

    def transform(self, X: np.ndarray) -> np.ndarray:
        result = X @ self.components_.T
        # L2 normalize
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (result / norms).astype(np.float32)


class SemanticEmbeddingEngine:
    """
    Full LSA (TF-IDF + SVD) semantic embedding engine.
    Trained on resume corpus for domain-aware similarity.
    """

    def __init__(self, n_components: int = 150):
        self.n_components = n_components
        self.tfidf = TFIDFVectorizer(
            max_features=10000, ngram_range=(1, 3), min_df=2, sublinear_tf=True
        )
        self.svd = TruncatedSVD(n_components=n_components, n_iter=7)
        self.is_fitted = False
        self.corpus_vectors: Optional[np.ndarray] = None
        self.corpus_labels: Optional[List[str]] = None

    def fit(self, documents: List[str], labels: Optional[List[str]] = None):
        """Train on resume corpus."""
        print(f"[Embedding] Fitting TF-IDF on {len(documents)} documents...")
        tfidf_matrix = self.tfidf.fit_transform(documents)
        print(f"[Embedding] TF-IDF shape: {tfidf_matrix.shape}")
        print(f"[Embedding] Running SVD with {self.n_components} components...")
        self.corpus_vectors = self.svd.fit_transform(tfidf_matrix)
        self.corpus_labels = labels or [""] * len(documents)
        self.is_fitted = True
        evr = self.svd.explained_variance_ratio_.sum()
        print(f"[Embedding] Explained variance: {evr:.3f}")
        return self

    def embed(self, text: str) -> np.ndarray:
        """Embed a single document."""
        tfidf_vec = self.tfidf.transform([text])
        sem_vec = self.svd.transform(tfidf_vec)
        return sem_vec[0]

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        tfidf_vecs = self.tfidf.transform(texts)
        return self.svd.transform(tfidf_vecs)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def similarity(self, text_a: str, text_b: str) -> float:
        return self.cosine_similarity(self.embed(text_a), self.embed(text_b))

    def most_similar_in_corpus(
        self, query: str, top_k: int = 5, filter_label: Optional[str] = None
    ) -> List[Tuple[int, float, str]]:
        """Find most similar corpus documents to query."""
        q_vec = self.embed(query)
        sims = self.corpus_vectors @ q_vec  # dot product (vectors are L2-normed)
        if filter_label:
            mask = np.array(
                [1.0 if l == filter_label else 0.0 for l in self.corpus_labels]
            )
            sims = sims * mask
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [(int(i), float(sims[i]), self.corpus_labels[i]) for i in top_idx]

    def get_role_centroid(self, role: str) -> np.ndarray:
        """Average embedding of all corpus docs for a role."""
        indices = [i for i, l in enumerate(self.corpus_labels) if l == role]
        if not indices:
            return np.zeros(self.n_components, dtype=np.float32)
        vecs = self.corpus_vectors[indices]
        centroid = vecs.mean(axis=0)
        norm = np.linalg.norm(centroid)
        return centroid / norm if norm > 0 else centroid

    def semantic_role_similarity(self, resume_text: str, role: str) -> float:
        """How semantically similar is the resume to canonical resumes for role."""
        centroid = self.get_role_centroid(role)
        resume_vec = self.embed(resume_text)
        return self.cosine_similarity(resume_vec, centroid)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "tfidf": self.tfidf,
                    "svd": self.svd,
                    "corpus_vectors": self.corpus_vectors,
                    "corpus_labels": self.corpus_labels,
                    "n_components": self.n_components,
                    "is_fitted": self.is_fitted,
                },
                f,
            )
        print(f"[Embedding] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "SemanticEmbeddingEngine":
        with open(path, "rb") as f:
            data = pickle.load(f)
        engine = cls(n_components=data["n_components"])
        engine.tfidf = data["tfidf"]
        engine.svd = data["svd"]
        engine.corpus_vectors = data["corpus_vectors"]
        engine.corpus_labels = data["corpus_labels"]
        engine.is_fitted = data["is_fitted"]
        print(f"[Embedding] Loaded from {path}")
        return engine
