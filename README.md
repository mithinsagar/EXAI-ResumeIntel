<div align="center">

# EXAI-ResumeIntel

### An Explainable Artificial Intelligence Framework for Automated Resume Analysis

**Using Shapley Values, LIME, and Domain Skill Ontology**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Institution](https://img.shields.io/badge/Institution-VIT-red.svg)](https://vit.ac.in)

**Author:** [Mithin Sagar S](https://github.com/mithinsagar) &middot; **Institution:** Vellore Institute of Technology (VIT)

[Live Demo](#live-demo) &middot; [Documentation](docs/) &middot; [Paper](docs/paper/) &middot; [Citation](#citation)

</div>

---

## Abstract

Automated resume screening systems assign match scores without providing any rationale, preventing candidates from understanding which skills to develop and preventing recruiters from auditing shortlisting decisions. **EXAI-ResumeIntel** is a five-layer explainable AI framework that addresses this transparency gap.

First, a hierarchical domain ontology with 346 alias mappings across 22 canonical skill nodes enables implicit skill detection: a resume listing "YOLOv8," "COCO dataset," and "anchor boxes" is correctly identified as carrying Computer Vision and Machine Learning expertise even though neither canonical term appears in the text. Second, a custom TF-IDF vectoriser combined with Latent Semantic Analysis via Truncated SVD (k=150, 7 power iterations) produces 150-dimensional semantic embeddings capturing 27.8% of corpus variance. Third, a four-component scoring engine (ontology 45%, semantic 30%, depth 15%, corpus 10%) yields an interpretable match score. Fourth, exact Shapley values satisfying all four game-theoretic axioms are independently validated by LIME across 300 perturbation samples, achieving 100% SHAP-LIME directional agreement. Fifth, counterfactual what-if explanations quantify the exact score gain from acquiring each missing skill.

Evaluated on 2,484 real-world resumes across 24 job categories using supervised LinearSVC with 5-fold stratified cross-validation, the system achieves **70.73% overall accuracy**, **69.37% macro precision**, **66.45% recall**, and **66.05% macro F1-score**. ROC-AUC scores range from 0.931 (Advocate) to 0.993 (HR).

---

## Key Results

| Metric | Value |
|:---|:---:|
| Overall Accuracy | **70.73%** |
| Macro Precision | 69.37% |
| Macro Recall | 66.45% |
| Macro F1-Score | 66.05% |
| Weighted F1-Score | 69.42% |
| Best Category F1 (Designer) | 0.86 |
| Best ROC-AUC (HR) | 0.9925 |
| SHAP-LIME Directional Agreement | 100% |

Compared with a semantic-only baseline that scores only 38% for the Machine Learning Engineer role, EXAI raises this to **74%** &mdash; a **+36 percentage-point improvement** from ontological inference alone.

---

## Table of Contents

- [Architecture](#architecture)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Data and Model Downloads](#data-and-model-downloads)
- [Training](#training)
- [Evaluation](#evaluation)
- [Deployment](#deployment)
- [Testing](#testing)
- [Live Demo](#live-demo)
- [Screenshots](#screenshots)
- [Documentation](#documentation)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## Architecture

The EXAI framework consists of five sequential stages:

```
                Resume Text  +  Target Job Role
                          |
                          v
              +-------------------------+
              |     Resume Parser       |   Experience years, education, quantified achievements
              +-------------------------+
                          |
                          v
              +-------------------------+
              |     Ontology Engine     |   22 canonical skills, 346 aliases
              |  n-gram phrase matching |   Parent-node credit propagation
              +-------------------------+
                          |
                          v
              +-------------------------+
              |     TF-IDF + SVD (LSA)  |   10,000 features (1-3 grams)
              |     k = 150 components  |   27.8% explained variance
              +-------------------------+
                          |
                          v
              +-------------------------+
              |    Scoring Engine       |   Ontology 45% + Semantic 30%
              |   Four weighted comps.  |   Depth 15% + Corpus 10%
              +-------------------------+
                          |
                          v
              +-------------------------+
              |       XAI Layer         |   Shapley (exact for |F|<=12)
              |  SHAP | LIME | CF | HT  |   Kernel SHAP (512 permutations)
              +-------------------------+
                          |
                          v
                    Dashboard Output
```

See [`docs/architecture.md`](docs/architecture.md) for the complete architectural documentation.

---

## Features

### Explainability
- **Exact Shapley Values** &mdash; game-theoretic feature attribution satisfying efficiency, symmetry, dummy, and additivity axioms
- **LIME Local Explanations** &mdash; 300-perturbation weighted ridge regression validating Shapley attributions
- **Counterfactual Analysis** &mdash; quantifies exact score gain from acquiring each missing skill
- **Attention Heatmap** &mdash; token-level importance mapping over the resume text
- **Feature Interaction Detection** &mdash; Shapley interaction values for skill pair synergies
- **Natural Language Insights** &mdash; human-readable explanations generated from all attribution values

### Modelling
- **Hierarchical Skill Ontology** &mdash; 22 canonical skill nodes, 346 alias phrases across IT, Finance, Healthcare, Legal, HR, Engineering, and other domains
- **Implicit Skill Inference** &mdash; parent-node credit propagation with decay factor 0.8
- **Custom TF-IDF Vectoriser** &mdash; from-scratch implementation with sublinear TF and smooth IDF
- **Truncated SVD (LSA)** &mdash; randomised Halko-Martinsson-Tropp algorithm
- **Multi-Dimensional Scoring** &mdash; four interpretable components with fixed weights

### Interfaces
- **REST API** &mdash; FastAPI backend with auto-generated Swagger documentation at `/docs`
- **Static Web UI** &mdash; single-page dashboard rendering all XAI outputs
- **Streamlit App** &mdash; interactive analytics interface with SBERT semantic matching, PDF/DOCX upload, and analysis history

---

## Repository Structure

```
EXAI-ResumeIntel/
|
+-- config/                    Runtime configuration
|   +-- settings.yaml
|   +-- role_weights.json
|   +-- scoring_weights.yaml
|
+-- core/                      Core research pipeline
|   +-- ontology.py            Skill ontology and extraction engine
|   +-- embeddings.py          TF-IDF + Truncated SVD
|   +-- scorer.py              Multi-dimensional scoring engine
|   +-- parser.py              Resume parser
|   +-- constants.py           Shared constants
|
+-- xai/                       Explainable AI modules
|   +-- explainer.py           XAI coordinator
|   +-- shapley.py             Exact and Kernel SHAP
|   +-- lime_explainer.py      LIME local explanation
|   +-- counterfactual.py      Counterfactual scenarios
|   +-- attention.py           Token-level heatmap
|   +-- interactions.py        Feature interaction detection
|   +-- nl_generator.py        Natural language insight generator
|
+-- api/                       FastAPI REST API
|   +-- server.py              Application factory
|   +-- routes.py              Endpoint definitions
|   +-- schemas.py             Pydantic request/response models
|
+-- app/                       Streamlit interactive application
|   +-- main.py                Entry point
|   +-- backend/               SBERT + memmap + XAI backend
|   +-- assets/                Static assets
|
+-- ui/                        Static HTML dashboard
|   +-- index.html             Single-page analyzer
|   +-- static/                CSS, JS, images
|
+-- training/                  Model training and evaluation
|   +-- train.py               Trains embedding engine
|   +-- evaluate.py            LinearSVC 5-fold cross-validation
|   +-- benchmark.py           Baseline comparisons
|
+-- data/                      Dataset directory
|   +-- raw/                   Raw CSV inputs (gitignored)
|   +-- processed/             Processed artifacts (gitignored)
|
+-- models/                    Trained model artifacts
|
+-- notebooks/                 Jupyter notebooks
|   +-- 01_data_exploration.ipynb
|   +-- 02_ontology_demo.ipynb
|   +-- 03_xai_walkthrough.ipynb
|   +-- EXAI_Resume_Intelligence_FINAL.ipynb
|
+-- tests/                     Unit and integration tests
|
+-- scripts/                   Utility shell scripts
|
+-- docs/                      Documentation
|   +-- architecture.md
|   +-- api_reference.md
|   +-- methodology.md
|   +-- deployment.md
|   +-- data_hosting.md
|   +-- figures/               Paper figures
|   +-- paper/                 Paper PDF/DOCX/PPTX
|
+-- deployment/                Docker, Kubernetes, Nginx
|
+-- .github/                   CI workflows and templates
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip 23.0 or higher
- (Optional) CUDA-capable GPU for accelerated inference with the Streamlit app
- (Optional) Docker 24.0 or higher for containerised deployment

### Clone the repository

```bash
git clone https://github.com/mithinsagar/EXAI-ResumeIntel.git
cd EXAI-ResumeIntel
```

### Set up the environment

```bash
python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

For development (pytest, black, mypy):

```bash
pip install -r requirements-dev.txt
```

### Download data and model artifacts

The raw datasets and pretrained models are hosted externally due to their size. See [Data and Model Downloads](#data-and-model-downloads) below.

```bash
bash scripts/download_data.sh
```

---

## Quick Start

### Run the REST API

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8765 --reload
```

Then open the auto-generated Swagger UI at [http://localhost:8765/docs](http://localhost:8765/docs).

### Run the static web dashboard

```bash
bash scripts/run_ui.sh
```

Opens the analyzer at [http://localhost:8080](http://localhost:8080). The dashboard talks to the FastAPI backend running on port 8765.

### Run the Streamlit application

```bash
streamlit run app/main.py
```

The Streamlit interface loads at [http://localhost:8501](http://localhost:8501) with PDF/DOCX upload, role autocomplete, and analysis history.

---

## Usage

### Python API

```python
from core.ontology import get_engine
from core.embeddings import SemanticEmbeddingEngine
from core.scorer import ScoringEngine

# Load engines
ontology = get_engine()
embeddings = SemanticEmbeddingEngine.load("models/embedding_engine.pkl")
scorer = ScoringEngine(ontology, embeddings)

# Score a resume
resume_text = "Machine Learning Engineer with 5 years experience..."
result = scorer.score(resume_text, role="MACHINE LEARNING ENGINEER")

print(f"Overall score: {result.overall_score * 100:.1f}%")
print(f"Ontology score: {result.ontology_score * 100:.1f}%")
print(f"Semantic score: {result.semantic_score * 100:.1f}%")

# Access XAI bundle
xai = result.xai_bundle
for skill in xai["shap_summary"][:5]:
    print(f"{skill['display_name']}: {skill['shapley_value'] * 100:+.2f}")
```

### REST API

```bash
curl -X POST http://localhost:8765/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "resume_text": "Machine Learning Engineer with 5 years experience in PyTorch, YOLOv8, and MLOps...",
    "role": "MACHINE LEARNING ENGINEER"
  }'
```

Full endpoint documentation: [`docs/api_reference.md`](docs/api_reference.md)

---

## API Reference

| Method | Endpoint | Description |
|:---|:---|:---|
| `GET` | `/health` | Health check |
| `GET` | `/roles` | List all supported job roles |
| `POST` | `/analyze` | Analyze a resume against a role (returns full XAI bundle) |
| `GET` | `/docs` | Interactive Swagger UI |
| `GET` | `/redoc` | ReDoc documentation |

See [`docs/api_reference.md`](docs/api_reference.md) for complete request and response schemas.

---

## Data and Model Downloads

The raw datasets and pretrained model artifacts are hosted on the Hugging Face Hub. Two repositories:

- **Dataset:** `mithinsagar/exai-resumeintel-data` &mdash; contains `clean_resume_data.csv` (2,484 resumes) and `jobs_dataset_with_features.csv` (~600 MB, 1M+ job postings)
- **Model:** `mithinsagar/exai-resumeintel-models` &mdash; contains `embedding_engine.pkl`, `job_df.pkl`, `job_embeddings.memmap`

Automated download:

```bash
bash scripts/download_data.sh
```

Or manually via the Hugging Face CLI:

```bash
huggingface-cli download mithinsagar/exai-resumeintel-data \
  --repo-type dataset --local-dir data/raw

huggingface-cli download mithinsagar/exai-resumeintel-models \
  --repo-type model --local-dir models
```

See [`docs/data_hosting.md`](docs/data_hosting.md) for details on data sources, licensing, and how to host your own copies.

---

## Training

To train the semantic embedding engine on the resume corpus:

```bash
python -m training.train
```

Output: `models/embedding_engine.pkl` &mdash; contains the fitted TF-IDF vocabulary, IDF weights, SVD components, and corpus centroids.

To customise training parameters, edit `config/settings.yaml`:

```yaml
embeddings:
  n_components: 150
  n_iter: 7
  max_features: 10000
  ngram_range: [1, 3]
  min_df: 2
```

---

## Evaluation

Run the LinearSVC 5-fold stratified cross-validation reproducing the paper results:

```bash
python -m training.evaluate
```

Reproduce baseline comparisons:

```bash
python -m training.benchmark
```

Full integration test:

```bash
python -m tests.test_full
```

---

## Deployment

### Docker

```bash
docker build -t exai-resumeintel:latest -f deployment/Dockerfile .
docker run -p 8765:8765 exai-resumeintel:latest
```

### Docker Compose (API + UI + Streamlit)

```bash
docker-compose -f deployment/docker-compose.yml up -d
```

Services:
- FastAPI on port 8765
- Static UI on port 8080
- Streamlit on port 8501
- Nginx reverse proxy on port 80

### Kubernetes

```bash
kubectl apply -f deployment/kubernetes/deployment.yaml
```

See [`docs/deployment.md`](docs/deployment.md) for cloud deployment guides (Render, Railway, Hugging Face Spaces, AWS ECS).

---

## Testing

```bash
pytest tests/ -v
```

With coverage:

```bash
pytest tests/ --cov=core --cov=xai --cov=api --cov-report=html
```

Individual test modules:

```bash
pytest tests/test_ontology.py -v
pytest tests/test_scorer.py -v
pytest tests/test_explainer.py -v
```

---

## Live Demo

- **Web Dashboard:** _https://exai-resumeintel.example.com_ (deployment pending)
- **Streamlit App:** _https://huggingface.co/spaces/mithinsagar/exai-resumeintel_ (deployment pending)
- **API Docs:** _https://exai-resumeintel.example.com/docs_ (deployment pending)

Deployment guides in [`docs/deployment.md`](docs/deployment.md).

---

## Screenshots

Place UI screenshots in the `screenshots/` directory. Referenced from README:

| Screenshot | Description |
|:---|:---|
| `screenshots/dashboard.png` | Main dashboard view |
| `screenshots/shap_values.png` | SHAP feature attribution |
| `screenshots/lime.png` | LIME local explanation |
| `screenshots/heatmap.png` | Attention heatmap |
| `screenshots/counterfactuals.png` | Counterfactual scenarios |
| `screenshots/streamlit_app.png` | Streamlit analytics interface |

---

## Documentation

| Document | Contents |
|:---|:---|
| [`docs/architecture.md`](docs/architecture.md) | System architecture, component diagrams, data flow |
| [`docs/methodology.md`](docs/methodology.md) | Mathematical foundations, algorithms, formulas |
| [`docs/api_reference.md`](docs/api_reference.md) | REST API endpoints, schemas, examples |
| [`docs/deployment.md`](docs/deployment.md) | Docker, Kubernetes, cloud deployment |
| [`docs/data_hosting.md`](docs/data_hosting.md) | Hugging Face Hub setup, licensing |
| [`docs/paper/`](docs/paper/) | Full research paper (PDF, DOCX), presentation slides |

---

## Citation

If you use EXAI-ResumeIntel in your research, please cite:

```bibtex
@software{sagar2026exai,
  author       = {Mithin Sagar S},
  title        = {{EXAI-ResumeIntel: An Explainable Artificial Intelligence
                  Framework for Automated Resume Analysis Using Shapley
                  Values, LIME, and Domain Skill Ontology}},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  url          = {https://github.com/mithinsagar/EXAI-ResumeIntel}
}
```

Machine-readable citation metadata is available in [`CITATION.cff`](CITATION.cff).

---

## Contributing

Contributions are welcome. Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) before submitting pull requests.

---

## License

This project is licensed under the MIT License &mdash; see the [`LICENSE`](LICENSE) file for details.

---

## Contact

**Mithin Sagar S**
Vellore Institute of Technology (VIT), Vellore, Tamil Nadu, India
GitHub: [https://github.com/mithinsagar](https://github.com/mithinsagar)

---

<div align="center">

Built at VIT University &middot; 2025-26

</div>
