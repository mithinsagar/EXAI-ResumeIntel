"""
EXAI-ResumeIntel: Ontology engine tests
========================================

Verifies:
    - Skill extraction and n-gram matching
    - Parent-node credit propagation
    - Role requirement lookup
    - Implicit skill inference (YOLO -> CV -> ML)

Module: tests.test_ontology
Author: Mithin Sagar S
GitHub: https://github.com/mithinsagar
"""

from __future__ import annotations

import pytest


def test_extract_common_skills(ontology, ml_resume):
    """The ML resume should register key ML/CV/NLP skills."""
    skills = ontology.extract_skills(ml_resume)
    assert "machine_learning" in skills or "computer_vision" in skills
    assert "python" in skills
    assert "deep_learning_frameworks" in skills
    assert all(0.0 < v <= 1.0 for v in skills.values())


def test_yolo_triggers_computer_vision(ontology):
    """The 'YOLOv8' alias should trigger the computer_vision canonical skill."""
    text = "Built YOLOv8 pipeline on COCO dataset with 78% mAP"
    skills = ontology.extract_skills(text)
    assert "computer_vision" in skills
    assert skills["computer_vision"] > 0.5


def test_parent_propagation(ontology):
    """Detecting computer_vision should credit machine_learning via parent."""
    text = "Implemented YOLOv8 object detection with OpenCV"
    skills = ontology.extract_skills(text)
    assert "computer_vision" in skills
    assert "machine_learning" in skills
    # Parent propagation uses decay 0.8, so ML confidence should be <= CV
    assert skills["machine_learning"] <= skills["computer_vision"]


def test_finance_domain(ontology, finance_resume):
    """Finance-domain terms should trigger finance and accounting skills."""
    skills = ontology.extract_skills(finance_resume)
    assert "finance" in skills


def test_hr_domain(ontology, hr_resume):
    """HR-domain terms should trigger the hr skill node."""
    skills = ontology.extract_skills(hr_resume)
    assert "hr" in skills


def test_role_requirements_ml_engineer(ontology):
    """MACHINE LEARNING ENGINEER should require ML-related skills."""
    reqs = ontology.get_role_requirements("MACHINE LEARNING ENGINEER")
    assert "machine_learning" in reqs
    assert "python" in reqs
    assert reqs["machine_learning"] >= 0.9


def test_role_requirements_fallback(ontology):
    """Unknown roles should return a sensible default requirement set."""
    reqs = ontology.get_role_requirements("SOME_MADE_UP_ROLE_XYZ")
    assert isinstance(reqs, dict)
    assert len(reqs) > 0


def test_all_roles_listed(ontology):
    """get_all_roles should return a non-empty sorted list."""
    roles = ontology.get_all_roles()
    assert isinstance(roles, list)
    assert len(roles) > 10
    assert roles == sorted(roles)


def test_empty_text_returns_empty_dict(ontology):
    """Empty resume text should return an empty skill dict."""
    skills = ontology.extract_skills("")
    assert skills == {} or len(skills) == 0


def test_display_name_conversion(ontology):
    """Display names should be title-cased with spaces."""
    assert ontology.get_skill_display_name("machine_learning") == "Machine Learning"
    assert ontology.get_skill_display_name("computer_vision") == "Computer Vision"
