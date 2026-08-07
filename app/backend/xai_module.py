"""
EXAI-ResumeIntel: Skill perturbation and counterfactual XAI
===========================================================

Module: app.backend.xai_module
Author: Mithin Sagar S
GitHub: https://github.com/mithinsagar
Institution: Vellore Institute of Technology (VIT)
"""

from typing import List, Dict, Any
import re
from collections import Counter

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from backend.similarity import cosine_similarity


SKILL_VOCAB = [
    "python", "java", "c++", "sql", "nosql", "mongodb", "postgresql",
    "aws", "gcp", "azure", "docker", "kubernetes", "terraform", "linux",
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras",
    "nlp", "computer vision", "deep learning", "machine learning", "llm",
    "data analysis", "data visualization", "power bi", "tableau", "excel",
    "spark", "hadoop", "airflow", "mlops", "devops", "system design",
    "microservices", "rest api", "fastapi", "flask", "django", "streamlit",
    "react", "node.js", "javascript", "typescript", "git", "ci/cd",
    "pytest", "unit testing", "statistics", "a/b testing", "feature engineering",
    "model deployment", "distributed systems", "cybersecurity", "network security",
    "product management", "agile", "scrum", "data engineering", "etl",
]

ROLE_SKILL_HINTS = {
    "data scientist": ["python", "machine learning", "pandas", "numpy", "statistics", "sql", "feature engineering", "model deployment"],
    "data analyst": ["sql", "excel", "tableau", "power bi", "data analysis", "statistics", "python"],
    "data engineer": ["python", "sql", "spark", "hadoop", "airflow", "etl", "aws", "gcp"],
    "machine learning engineer": ["python", "machine learning", "deep learning", "pytorch", "tensorflow", "mlops", "docker", "kubernetes", "model deployment"],
    "ai researcher": ["python", "deep learning", "pytorch", "tensorflow", "nlp", "llm", "statistics"],
    "software engineer": ["java", "python", "system design", "rest api", "docker", "git", "unit testing"],
    "backend developer": ["python", "java", "rest api", "fastapi", "flask", "django", "postgresql", "system design"],
    "full stack developer": ["javascript", "typescript", "react", "node.js", "rest api", "sql", "git"],
    "devops engineer": ["aws", "gcp", "azure", "docker", "kubernetes", "terraform", "ci/cd", "linux"],
}

STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "have", "has", "had",
    "will", "would", "could", "should", "into", "onto", "your", "our", "their",
    "you", "they", "them", "his", "her", "was", "were", "are", "is", "am", "be",
    "as", "at", "by", "of", "to", "in", "on", "an", "a", "or", "if", "it", "its",
    "role", "resume", "experience", "work", "worked", "using", "used", "team",
}


def _normalize_role(role: str) -> str:
    return role.strip().lower()


def _extract_resume_skills(resume_text: str) -> List[str]:
    text = resume_text.lower()
    found = []
    for skill in SKILL_VOCAB:
        pattern = r"\b" + re.escape(skill.lower()) + r"\b"
        if re.search(pattern, text):
            found.append(skill)
    return found


def _role_expected_skills(target_role: str) -> List[str]:
    role_norm = _normalize_role(target_role)
    for role_key, skills in ROLE_SKILL_HINTS.items():
        if role_key in role_norm:
            return skills
    # fallback broad defaults for unknown roles
    return ["python", "sql", "docker", "git", "system design"]


def identify_missing_skills(resume_text: str, target_role: str, max_missing: int = 8) -> List[str]:
    present = set(_extract_resume_skills(resume_text))
    expected = _role_expected_skills(target_role)
    missing = [s for s in expected if s not in present]
    return missing[:max_missing]


def _fallback_candidate_tokens(resume_text: str, max_candidates: int = 20) -> List[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9\+\#\-/\.]{2,}", resume_text.lower())
    clean = [t for t in tokens if t not in STOPWORDS and not t.isdigit()]
    freq = Counter(clean)
    return [w for w, _ in freq.most_common(max_candidates)]


def compute_skill_contribution(
    resume_text: str,
    target_role: str,
    model: SentenceTransformer,
    job_df: pd.DataFrame = None,
    job_embeddings: np.memmap = None,
    top_k: int = 12,
) -> List[Dict[str, Any]]:

    if job_df is None or job_embeddings is None:
        return []

    # robust role matching
    norm_roles = job_df["Role"].astype(str).str.strip().str.lower()
    mask = norm_roles == _normalize_role(target_role)
    if not mask.any():
        return []

    role_indices = np.where(mask.values)[0]
    job_vec = job_embeddings[role_indices[0]]

    # Skill-aware candidates first to avoid nonsensical words in XAI output.
    candidate_skills = _extract_resume_skills(resume_text)
    if not candidate_skills:
        candidate_skills = _fallback_candidate_tokens(resume_text, max_candidates=18)
    candidate_skills = candidate_skills[:24]
    if not candidate_skills:
        return []

    resume_vec = model.encode(
        [resume_text], convert_to_numpy=True, show_progress_bar=False
    )[0]
    original_score = cosine_similarity(resume_vec, job_vec)

    # Batch perturbations in one encoder call for speed.
    modified_texts: List[str] = []
    labels: List[str] = []
    for skill in candidate_skills:
        pattern = r"\b" + re.escape(skill) + r"\b"
        modified_text = re.sub(pattern, " ", resume_text, flags=re.IGNORECASE)
        modified_text = " ".join(modified_text.split())
        if modified_text and modified_text != resume_text:
            modified_texts.append(modified_text)
            labels.append(skill)

    if not modified_texts:
        return []

    modified_vecs = model.encode(
        modified_texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=32,
    )

    contributions: List[Dict[str, Any]] = []
    for skill, modified_vec in zip(labels, modified_vecs):
        modified_score = cosine_similarity(modified_vec, job_vec)
        impact = original_score - modified_score
        if abs(impact) > 0.0005:
            contributions.append(
                {"skill": skill, "contribution": float(impact * 100)}
            )

    contributions = sorted(
        contributions,
        key=lambda x: abs(x["contribution"]),
        reverse=True
    )

    return contributions[:top_k]


def simulate_skill_addition(
    resume_text: str,
    target_role: str,
    model: SentenceTransformer,
    missing_skills: List[str],
    job_df: pd.DataFrame = None,
    job_embeddings: np.memmap = None,
    max_skills: int = 8,
) -> List[Dict[str, Any]]:

    if job_df is None or job_embeddings is None:
        return []

    norm_roles = job_df["Role"].astype(str).str.strip().str.lower()
    mask = norm_roles == _normalize_role(target_role)
    if not mask.any():
        return []

    role_indices = np.where(mask.values)[0]
    job_vec = job_embeddings[role_indices[0]]

    resume_vec = model.encode(
        [resume_text], convert_to_numpy=True, show_progress_bar=False
    )[0]
    original_score = cosine_similarity(resume_vec, job_vec)

    # If missing skills are not provided by upstream evaluation,
    # infer role-specific missing skills from hints.
    if not missing_skills:
        missing_skills = identify_missing_skills(resume_text, target_role, max_missing=max_skills)
    candidates = missing_skills[:max_skills]
    if not candidates:
        return []

    modified_texts = [f"{resume_text} {skill}" for skill in candidates]
    modified_vecs = model.encode(
        modified_texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=32,
    )

    improvements: List[Dict[str, Any]] = []
    for skill, modified_vec in zip(candidates, modified_vecs):
        new_score = cosine_similarity(modified_vec, job_vec)
        delta = new_score - original_score
        if delta > 0:
            improvements.append(
                {"skill": skill, "potential_increase": float(delta * 100)}
            )

    improvements = sorted(
        improvements,
        key=lambda x: x["potential_increase"],
        reverse=True
    )

    return improvements


def generate_detailed_explanation(
    resume_text: str,
    target_role: str,
    match_percentage: float,
    semantic_similarity: float,
    top_contributing_skills: list,
    missing_skills: list,
    skill_improvements: list,
) -> str:
    """
    Structured, human-readable explanation aligned with the provided template.
    """
    # Derive representative skills from inputs
    strength_skills = [s["skill"] for s in top_contributing_skills[:3]]
    more_strengths = [s["skill"] for s in top_contributing_skills[3:6]]
    missing_main = missing_skills[:3]
    suggested_skills = [s["skill"] for s in skill_improvements[:3]]

    def fmt_list(xs: list[str]) -> str:
        return ", ".join(xs) if xs else "relevant technical areas"

    # Stage description based on match
    if match_percentage >= 75:
        stage = "strong"
    elif match_percentage >= 50:
        stage = "moderate"
    else:
        stage = "early-stage"

    paragraphs: list[str] = []

    # Profile–Role Alignment
    paragraphs.append(
        f"After analyzing the resume against the selected role **{target_role}**, "
        f"the system identified an overall alignment score of **{match_percentage:.1f}%** "
        f"(semantic similarity ≈ {semantic_similarity:.3f}). "
        f"The candidate’s background shows relevant experience in {fmt_list(strength_skills)}, "
        f"which are commonly expected for this role. "
        f"However, some important capabilities related to {fmt_list(missing_main[:2])} appear limited or not clearly "
        f"highlighted in the resume, which affects the final compatibility score."
    )

    # Key Strengths
    paragraphs.append(
        f"**Key Strengths.** The analysis indicates that the candidate demonstrates strong capabilities in "
        f"{fmt_list(strength_skills or more_strengths)}. "
        f"These skills are highly relevant to the responsibilities associated with the **{target_role}** and "
        f"contribute positively to the overall evaluation. "
        f"The presence of hands‑on projects, prior work experience, or technical exposure around these areas "
        f"further supports the candidate’s suitability for this type of role."
    )

    # Areas That Could Be Improved
    paragraphs.append(
        f"**Areas That Could Be Improved.** While the profile shows a solid foundation, the system detected gaps in "
        f"areas such as {fmt_list(missing_main)}. "
        f"These skills are commonly expected in modern industry roles and their limited emphasis in the resume "
        f"slightly lowers the match score. Strengthening these areas—either by gaining experience or making existing "
        f"experience more explicit—would improve the candidate’s overall alignment with the role requirements."
    )

    # Potential Improvements
    paragraphs.append(
        f"**Potential Improvements.** If the resume included stronger evidence of {fmt_list(suggested_skills)}, "
        f"the compatibility with this role could improve noticeably. Demonstrating these skills through concrete "
        f"projects, production deployments, contributions to real systems, certifications, or measurable outcomes "
        f"would make the profile more competitive for positions related to **{target_role}**."
    )

    # Final Insight
    paragraphs.append(
        f"**Final Insight.** Overall, the resume shows **{stage}** alignment with the expectations of the "
        f"**{target_role}**. With additional focus on the highlighted improvement areas, the candidate’s profile could "
        f"become significantly stronger and more competitive for similar roles in the industry."
    )

    return "\n\n".join(paragraphs)