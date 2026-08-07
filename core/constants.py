"""
EXAI-ResumeIntel: Shared constants and metadata
================================================

Module: core.constants
Author: Mithin Sagar S
GitHub: https://github.com/mithinsagar
Institution: Vellore Institute of Technology (VIT)
"""

from pathlib import Path

# ─── Project metadata ──────────────────────────────────────────────
APP_NAME: str = "EXAI-ResumeIntel"
APP_VERSION: str = "1.0.0"
AUTHOR: str = "Mithin Sagar S"
AUTHOR_EMAIL: str = "mithinsagar@gmail.com"
AUTHOR_GITHUB: str = "https://github.com/mithinsagar"
INSTITUTION: str = "Vellore Institute of Technology (VIT)"
LICENSE: str = "MIT"

# ─── Paths ─────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
CONFIG_DIR: Path = PROJECT_ROOT / "config"
DATA_DIR: Path = PROJECT_ROOT / "data"
MODELS_DIR: Path = PROJECT_ROOT / "models"
LOGS_DIR: Path = PROJECT_ROOT / "logs"

RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

RESUME_CORPUS_PATH: Path = RAW_DATA_DIR / "clean_resume_data.csv"
JOB_POSTINGS_PATH: Path = RAW_DATA_DIR / "jobs_dataset_with_features.csv"
EMBEDDING_MODEL_PATH: Path = MODELS_DIR / "embedding_engine.pkl"
JOB_DF_PATH: Path = MODELS_DIR / "job_df.pkl"
JOB_EMBEDDINGS_PATH: Path = MODELS_DIR / "job_embeddings.memmap"

# ─── Scoring engine weights (paper eq. 7) ──────────────────────────
DEFAULT_SCORING_WEIGHTS: dict = {
    "ontology": 0.45,
    "semantic": 0.30,
    "depth": 0.15,
    "corpus": 0.10,
}

DEFAULT_DEPTH_WEIGHTS: dict = {
    "experience": 0.35,
    "education": 0.25,
    "quantified": 0.20,
    "tool_density": 0.20,
}

SCORE_CAP: float = 0.97

# ─── Embedding engine defaults ─────────────────────────────────────
DEFAULT_MAX_FEATURES: int = 10000
DEFAULT_NGRAM_RANGE: tuple = (1, 3)
DEFAULT_MIN_DF: int = 2
DEFAULT_N_COMPONENTS: int = 150
DEFAULT_N_ITER: int = 7
DEFAULT_RANDOM_STATE: int = 42

# ─── XAI defaults ──────────────────────────────────────────────────
DEFAULT_SHAPLEY_SAMPLES: int = 512
DEFAULT_SHAPLEY_EXACT_THRESHOLD: int = 12
DEFAULT_LIME_SAMPLES: int = 300
DEFAULT_LIME_KERNEL_WIDTH: float = 0.25
DEFAULT_LIME_RIDGE_ALPHA: float = 0.01

# ─── Ontology ──────────────────────────────────────────────────────
PARENT_DECAY: float = 0.8
NGRAM_MAX: int = 4
BASE_CONFIDENCE: float = 0.6

# ─── Impact labels ─────────────────────────────────────────────────
IMPACT_THRESHOLDS: dict = {
    "very_high": 0.15,
    "high": 0.08,
    "medium": 0.03,
    "low": 0.005,
    "hurting": -0.05,
}

IMPORTANCE_THRESHOLDS: dict = {
    "critical": 0.7,
    "high": 0.4,
    "medium": 0.2,
    "low": 0.05,
}

# ─── Education level mapping ───────────────────────────────────────
EDUCATION_KEYWORDS: dict = {
    "phd": 5, "doctorate": 5,
    "m.tech": 4, "mtech": 4, "m.s": 4, "ms ": 4, "mba": 4, "m.b.a": 4,
    "master": 4, "postgraduate": 4,
    "m.com": 3, "mca": 3, "b.tech": 3, "btech": 3, "b.e": 3, "be ": 3,
    "b.sc": 2, "bsc": 2, "b.com": 2, "bca": 2, "bachelor": 2, "graduate": 2,
    "diploma": 1, "certification": 2, "certified": 2,
}
