"""
EXAI-ResumeIntel: Core research pipeline
=========================================

Contains the ontology engine, semantic embedding engine, resume parser,
and multi-dimensional scoring engine.

Author: Mithin Sagar S
GitHub: https://github.com/mithinsagar
Institution: Vellore Institute of Technology (VIT)
"""

from core.constants import (
    APP_NAME,
    APP_VERSION,
    AUTHOR,
    AUTHOR_GITHUB,
    INSTITUTION,
)
from core.embeddings import SemanticEmbeddingEngine, TFIDFVectorizer, TruncatedSVD
from core.ontology import OntologyEngine, get_engine
from core.parser import ResumeParser
from core.scorer import ScoringEngine, ScoringResult, ModelTrainer

__all__ = [
    "APP_NAME",
    "APP_VERSION",
    "AUTHOR",
    "AUTHOR_GITHUB",
    "INSTITUTION",
    "OntologyEngine",
    "get_engine",
    "SemanticEmbeddingEngine",
    "TFIDFVectorizer",
    "TruncatedSVD",
    "ResumeParser",
    "ScoringEngine",
    "ScoringResult",
    "ModelTrainer",
]

__version__ = APP_VERSION
