"""
EXAI-ResumeIntel: API integration tests
========================================

Uses FastAPI's TestClient (starlette-based) to hit endpoints in-process.

Module: tests.test_api
Author: Mithin Sagar S
GitHub: https://github.com/mithinsagar
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.server import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_root_returns_metadata(client):
    """GET / should return application metadata."""
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "EXAI-ResumeIntel"
    assert "author" in data


def test_health_endpoint(client):
    """GET /health should return status=ok."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data
    assert "version" in data


def test_roles_endpoint(client):
    """GET /roles should return a non-empty sorted list."""
    resp = client.get("/roles")
    assert resp.status_code == 200
    data = resp.json()
    assert "roles" in data
    assert data["count"] > 0
    assert data["roles"] == sorted(data["roles"])


def test_analyze_endpoint_success(client, ml_resume):
    """POST /analyze with valid inputs should return a full XAI bundle."""
    payload = {"resume_text": ml_resume, "role": "MACHINE LEARNING ENGINEER"}
    resp = client.post("/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "overall_score" in data
    assert "shap_summary" in data
    assert "counterfactuals" in data
    assert "nl_overall" in data


def test_analyze_rejects_short_resume(client):
    """Resume text under 50 chars should be rejected (422)."""
    payload = {"resume_text": "too short", "role": "MACHINE LEARNING ENGINEER"}
    resp = client.post("/analyze", json=payload)
    assert resp.status_code == 422


def test_analyze_missing_role(client, ml_resume):
    """Missing role should be rejected (422)."""
    payload = {"resume_text": ml_resume}
    resp = client.post("/analyze", json=payload)
    assert resp.status_code == 422


def test_docs_endpoint(client):
    """Swagger UI should be reachable at /docs."""
    resp = client.get("/docs")
    assert resp.status_code == 200


def test_openapi_schema(client):
    """OpenAPI schema should be available at /openapi.json."""
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    schema = resp.json()
    assert schema["info"]["title"] == "EXAI-ResumeIntel"
