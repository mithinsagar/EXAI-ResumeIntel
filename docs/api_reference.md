# API Reference

Complete reference for the EXAI-ResumeIntel FastAPI backend.

**Author:** Mithin Sagar S ([@mithinsagar](https://github.com/mithinsagar))

## Base URL

Local development:
```
http://localhost:8765
```

Interactive documentation is auto-generated at `/docs` (Swagger UI) and `/redoc` (ReDoc).

## Endpoints

### GET /health

Returns the health status of the API and whether the embedding model is loaded.

**Response:** `HealthResponse`

```json
{
  "status": "ok",
  "model_loaded": true,
  "version": "1.0.0"
}
```

**Example:**

```bash
curl http://localhost:8765/health
```

---

### GET /roles

Returns the list of supported target job roles.

**Response:** `RolesResponse`

```json
{
  "roles": [
    "ACCOUNTANT",
    "ADVOCATE",
    "BANKING",
    "DATA SCIENTIST",
    "FINANCE",
    "HR",
    "MACHINE LEARNING ENGINEER",
    "..."
  ],
  "count": 24
}
```

**Example:**

```bash
curl http://localhost:8765/roles
```

---

### POST /analyze

Runs the full EXAI pipeline on a resume/role pair and returns the complete XAI bundle.

**Request:** `AnalyzeRequest`

```json
{
  "resume_text": "Machine Learning Engineer with 5 years experience...",
  "role": "MACHINE LEARNING ENGINEER"
}
```

Constraints:
- `resume_text`: string, minimum 50 characters
- `role`: string, non-empty, must be a supported role slug (see `/roles`)

**Response:** `AnalyzeResponse` (abbreviated)

```json
{
  "overall_score": 74.3,
  "overall_score_raw": 0.743,
  "score_components": {
    "ontology_score": 82.5,
    "semantic_score": 71.2,
    "depth_score": 68.0,
    "corpus_score": 65.5,
    "weights": {
      "ontology": 0.45,
      "semantic": 0.30,
      "depth": 0.15,
      "corpus": 0.10
    }
  },
  "parsed_resume": {
    "years_experience": 5.0,
    "education_level": 3,
    "word_count": 187,
    "has_quantified_achievements": true
  },
  "shap_summary": [
    {
      "skill": "python",
      "display_name": "Python",
      "shapley_value": 0.185,
      "contribution_pct": 18.5,
      "your_score": 90.0,
      "required_score": 100.0,
      "gap": 10.0,
      "direction": "positive",
      "impact_label": "Very High Impact"
    }
  ],
  "lime_results": [
    {
      "skill": "python",
      "display_name": "Python",
      "lime_weight": 0.152,
      "your_score": 90.0,
      "direction": "positive"
    }
  ],
  "counterfactuals": [
    {
      "skills_to_add": ["kubernetes"],
      "score_before": 74.3,
      "score_after": 79.1,
      "gain": 4.8,
      "description": "Learn Kubernetes",
      "priority": "Medium Priority"
    }
  ],
  "strong_skills": ["python", "machine_learning", "pytorch"],
  "missing_skills": ["kubernetes"],
  "partial_skills": ["databases"],
  "nl_overall": "You have a strong fit (74.3%) for the Machine Learning Engineer role...",
  "role": "MACHINE LEARNING ENGINEER",
  "n_skills_detected": 12,
  "n_skills_required": 10
}
```

**Example:**

```bash
curl -X POST http://localhost:8765/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "resume_text": "Machine Learning Engineer with 5 years experience in PyTorch, YOLOv8, COCO dataset, MLOps, Docker, Kubernetes...",
    "role": "MACHINE LEARNING ENGINEER"
  }'
```

**Error responses:**

| Status | Description |
|:---:|:---|
| 422 | Validation error (resume too short, missing fields) |
| 500 | Internal error during analysis |

---

## Response Schemas

### `ScoreComponents`

Four-component score breakdown following the paper's eq. (7).

| Field | Type | Description |
|:---|:---|:---|
| `ontology_score` | float | Ontology skill match, 0-100 |
| `semantic_score` | float | LSA semantic similarity to role centroid, 0-100 |
| `depth_score` | float | Experience + education + achievements + tool density, 0-100 |
| `corpus_score` | float | Similarity to top-5 similar resumes in role, 0-100 |
| `weights` | dict | Component weights (default: 0.45, 0.30, 0.15, 0.10) |

### `SHAPSummaryItem`

Per-feature Shapley value attribution.

| Field | Type | Description |
|:---|:---|:---|
| `skill` | str | Canonical skill name |
| `display_name` | str | Title-cased human-readable name |
| `shapley_value` | float | Raw Shapley value (fraction of score) |
| `contribution_pct` | float | Contribution as percentage of positive total |
| `your_score` | float | Detected skill confidence, 0-100 |
| `required_score` | float | Required weight for role, 0-100 |
| `gap` | float | Required minus detected, 0-100 |
| `direction` | str | "positive" / "neutral" / "negative" |
| `impact_label` | str | "Very High Impact" / "High" / "Medium" / "Low" / "Hurting" / "Negligible" |

### `LIMEResultItem`

Per-feature LIME local weight.

| Field | Type | Description |
|:---|:---|:---|
| `skill` | str | Canonical skill name |
| `display_name` | str | Title-cased name |
| `lime_weight` | float | Local ridge coefficient |
| `your_score` | float | Detected confidence, 0-100 |
| `direction` | str | "positive" / "negative" |

### `CounterfactualScenario`

Single what-if scenario.

| Field | Type | Description |
|:---|:---|:---|
| `type` | str | "add_single" or "add_pair" |
| `skills_to_add` | list[str] | Skill names to acquire |
| `score_before` | float | Current score, 0-100 |
| `score_after` | float | Projected score, 0-100 |
| `gain` | float | Projected gain, 0-100 |
| `description` | str | Human-readable action |
| `priority` | str | "Top Priority" / "High" / "Medium" / "Low" |

## Python Client Example

```python
import requests

resp = requests.post("http://localhost:8765/analyze", json={
    "resume_text": "Machine Learning Engineer with 5 years experience...",
    "role": "MACHINE LEARNING ENGINEER",
})
resp.raise_for_status()
result = resp.json()

print(f"Overall: {result['overall_score']}%")
print(f"Top skill: {result['shap_summary'][0]['display_name']}")
print(f"Suggestion: {result['counterfactuals'][0]['description']}")
```

## JavaScript Client Example

```javascript
const response = await fetch('http://localhost:8765/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        resume_text: 'Machine Learning Engineer with 5 years experience...',
        role: 'MACHINE LEARNING ENGINEER',
    }),
});
const result = await response.json();
console.log(`Overall: ${result.overall_score}%`);
```
