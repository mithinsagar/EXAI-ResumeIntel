# Architecture

This document describes the internal architecture of EXAI-ResumeIntel: how a raw resume flows through the five-layer pipeline to a fully-explained match score.

**Author:** Mithin Sagar S ([@mithinsagar](https://github.com/mithinsagar))

## System Overview

EXAI-ResumeIntel is organised as three coordinated systems that share the same core research pipeline:

- **Core research pipeline** (`core/`, `xai/`) &mdash; the reference implementation described in the paper, built on custom TF-IDF, Truncated SVD, exact Shapley values, and LIME with weighted ridge regression.
- **FastAPI service** (`api/`) &mdash; wraps the core pipeline as a REST API with auto-generated Swagger documentation.
- **Streamlit application** (`app/`) &mdash; interactive analytics interface using Sentence Transformers (`all-mpnet-base-v2`) and precomputed job embeddings for large-scale semantic matching.

The static web dashboard (`ui/`) is a client of the FastAPI service.

## Five-Layer Pipeline

```
Input:   raw resume text + target job role
                    |
                    v
Layer 1: Resume Parser
                    |    experience, education, quantified achievements
                    v
Layer 2: Ontology Engine
                    |    22 canonical skills, 346 aliases
                    |    n-gram phrase matching (n = 4, 3, 2, 1)
                    |    parent-node credit propagation (decay = 0.8)
                    v
Layer 3: Semantic Embedding (TF-IDF + LSA)
                    |    10,000 features, 1-3 grams, sublinear TF
                    |    Truncated SVD, k = 150, 7 power iterations
                    v
Layer 4: Scoring Engine
                    |    Ontology 45% + Semantic 30%
                    |    Depth 15% + Corpus 10%
                    v
Layer 5: XAI Layer
                    |    Shapley values (exact for |F| <= 12)
                    |    LIME local explanation (300 perturbations)
                    |    Counterfactual, attention, interactions, NL
                    v
Output:  overall_score + full XAI bundle
```

## Layer 1: Resume Parser

Module: `core/parser.py`

Extracts three structured signals:

1. **Years of experience** &mdash; regex `(\d+)\+?\s*(?:years?|yrs?)`, defaulting to 2 years.
2. **Education level** &mdash; 0-5 scale (PhD 5, MTech/MBA 4, BTech 3, BSc/BCom 2, Diploma 1).
3. **Quantified achievements** &mdash; percentage figures, multipliers (`\d+x`), and impact verbs.

## Layer 2: Ontology Engine

Module: `core/ontology.py`

The ontology graph `G = (V, E)` contains 22 canonical skill nodes and 346 alias strings covering IT, Finance, HR, Healthcare, Legal, and other domains.

### n-gram Phrase Matching

Given a normalised token sequence `T = [t1, ..., tm]`, the engine scans window sizes `n in {4, 3, 2, 1}` in decreasing order (longest-match priority). Match confidence for an n-gram hit is:

```
conf(n) = min(1.0, 0.6 + 0.1 * n)
```

yielding 1.0 for a 4-gram match and 0.7 for a unigram match.

### Parent-Node Propagation

When child skill `s_c` is detected with confidence `c_c`, each ancestor `s_p` at depth `d` is credited:

```
conf(s_p) <- max( conf(s_p), c_c * 0.8^d )
```

This is the mechanism by which "YOLOv8" implies ML expertise without the phrase "machine learning" appearing in the text.

## Layer 3: Semantic Embedding

Module: `core/embeddings.py`

### TF-IDF Vectoriser

Custom implementation with sublinear TF and smooth IDF:

```
TF_sub(t, d)     = 1 + log(count(t, d))
IDF_smooth(t, D) = log((1 + |D|) / (1 + df(t, D))) + 1
```

Vocabulary contains the top 10,000 terms (1-3 grams); all vectors are L2 normalised.

### Truncated SVD (LSA)

Randomised Halko-Martinsson-Tropp algorithm compresses the TF-IDF matrix to k = 150 components with 7 power iterations. Reaches 27.8% explained variance and cosine similarity 0.535 without over-dimensioning the embedding space.

### Role Centroids

For each category, a centroid `c_R` is the L2-normalised mean of all embeddings in that category. The semantic score component is:

```
S_sem = (cos(r, c_R) + 1) / 2
```

## Layer 4: Scoring Engine

Module: `core/scorer.py`

Combines four weighted components:

```
Score = 0.45 * S_ont + 0.30 * S_sem + 0.15 * S_dep + 0.10 * S_cor
```

capped at 0.97 to prevent false certainty.

### Ontology Sub-Score

Non-linear coverage adjustment rewards near-complete coverage disproportionately:

```
S_ont = sum_s [ w_s * (1 - (1 - cov(s))^1.5) ] / sum_s w_s
```

where `cov(s) = min(1, c_s / w_s)`.

### Depth Sub-Score

```
S_dep = 0.35 * S_exp + 0.25 * S_edu + 0.20 * S_quant + 0.20 * S_tool
```

### Corpus Sub-Score

Mean cosine similarity to the top-5 most similar resumes in the target category.

## Layer 5: XAI Layer

Module: `xai/`

Six independent components orchestrated by `XAICoordinator`:

### Shapley Values (`xai/shapley.py`)

Exact computation for feature sets `|F| <= 12`:

```
phi_i = sum over S of [|S|!(|F|-|S|-1)!/|F|!] * [v(S+i) - v(S)]
```

Kernel SHAP with 512 permutation samples for larger sets. Satisfies four game-theoretic axioms: Efficiency, Symmetry, Dummy, Additivity.

### LIME (`xai/lime_explainer.py`)

Generates `N = 300` perturbed skill vectors `x_j = x . m_j`, `m_j ~ Bernoulli(0.5)^n`, weighted by proximity kernel (sigma = 0.25), and fits weighted ridge regression (alpha = 0.01):

```
beta = (M^T W M + alpha I)^-1 M^T W y
```

### Counterfactual (`xai/counterfactual.py`)

For each missing skill `s`, computes exact score gain:

```
delta_s = v(c + {s: w_s}) - v(c)
```

Skills are labelled Top Priority (> 12%), High (> 8%), Medium (> 4%), or Low.

### Attention Heatmap (`xai/attention.py`)

Token-level importance map from ontology alias matches combined with TF-IDF fallback for unmatched tokens.

### Feature Interactions (`xai/interactions.py`)

Shapley interaction values:

```
phi_ij = v(both) - v(only_i) - v(only_j) + v(neither)
```

### Natural Language Generator (`xai/nl_generator.py`)

Templated narratives generated from computed attribution values.

## Streamlit Application Architecture

The `app/` directory implements a separate but complementary Streamlit interface built on SBERT (`all-mpnet-base-v2`) with precomputed job embeddings:

- **`app/backend/model_loader.py`** &mdash; loads SBERT model and memmapped job embeddings
- **`app/backend/preprocessing.py`** &mdash; PDF/DOCX text extraction and tokenization
- **`app/backend/similarity.py`** &mdash; chunked cosine similarity and fuzzy role autocomplete
- **`app/backend/evaluation.py`** &mdash; role-vs-resume semantic scoring
- **`app/backend/xai_module.py`** &mdash; skill perturbation and addition XAI
- **`app/backend/history_store.py`** &mdash; SQLite persistence of past analyses

## Data Flow

```
Static Web UI (ui/)     -->  FastAPI (api/)  -->  Core (core/) + XAI (xai/)
Streamlit App (app/)    -->  app/backend/    -->  SBERT + precomputed embeddings
Notebooks (notebooks/)  -->  Core (core/) + XAI (xai/)   (research and evaluation)
Training (training/)    -->  Core (core/)    -->  models/
```

## Extension Points

- **New skill domain:** add a canonical node to `SKILL_ONTOLOGY` with aliases, parents, and category. Update `ROLE_SKILL_WEIGHTS`.
- **New scoring weight profile:** add to `config/scoring_weights.yaml` and pass via `ScoringEngine(weights=...)`.
- **New XAI component:** implement class in `xai/`, register in `XAICoordinator.explain()`.
- **New API endpoint:** add route to `api/routes.py`, response schema to `api/schemas.py`.
