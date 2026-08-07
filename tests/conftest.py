"""
EXAI-ResumeIntel: Pytest configuration and shared fixtures
===========================================================

Module: tests.conftest
Author: Mithin Sagar S
GitHub: https://github.com/mithinsagar
"""

from __future__ import annotations

import pytest

from core.ontology import get_engine


@pytest.fixture(scope="session")
def ontology():
    """Return the singleton OntologyEngine instance."""
    return get_engine()


@pytest.fixture
def ml_resume() -> str:
    """A representative ML Engineer resume for testing."""
    return """
    Machine Learning Engineer with 5 years experience.

    Technical Skills:
    - Computer Vision: YOLOv8, COCO dataset, OpenCV, ResNet
    - Deep Learning: PyTorch, TensorFlow, ONNX, TensorRT
    - NLP: BERT, HuggingFace Transformers, LangChain
    - MLOps: MLflow, Docker, Kubernetes, AWS SageMaker
    - Python: pandas, numpy, scikit-learn, FastAPI
    - Databases: PostgreSQL, MongoDB, Redis

    Experience:
    Senior ML Engineer at TechCorp (2022-Present)
    Trained YOLOv8 on COCO achieving 78% mAP.
    Reduced inference latency 65% with TensorRT + INT8 quantization.

    Education: B.Tech Computer Science, VIT University
    """


@pytest.fixture
def finance_resume() -> str:
    """A representative Finance Analyst resume."""
    return """
    Senior Financial Analyst with 6 years experience in investment banking.

    Skills: DCF modeling, valuation, equity research, Bloomberg,
    Excel VBA, financial statements analysis, GAAP, IFRS, M&A due diligence.

    Experience: Goldman Sachs, HDFC Bank.
    Education: MBA Finance, IIM Ahmedabad. CA Inter.
    """


@pytest.fixture
def hr_resume() -> str:
    """A representative HR Manager resume."""
    return """
    HR Manager with 7 years experience.

    Skills: Talent acquisition, Workday HRIS, SuccessFactors, ADP payroll,
    employee engagement, DEI, labor law compliance, workforce planning,
    performance management.

    Experience: Infosys, TCS.
    Education: MBA HR, XLRI Jamshedpur.
    """
