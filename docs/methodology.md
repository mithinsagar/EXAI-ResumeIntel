# Methodology

Mathematical foundations and algorithmic details of EXAI-ResumeIntel.

**Author:** Mithin Sagar S ([@mithinsagar](https://github.com/mithinsagar))

## Overview

EXAI addresses two compounding problems in automated resume screening:

1. From the candidate's perspective, opacity prevents targeted skill development.
2. From the recruiter's perspective, unauditable black-box scores cannot be justified under algorithmic transparency regulations (e.g. the EU AI Act).

A third technical challenge is that domain expertise is encoded in specialised terminology, not canonical labels. A computer vision engineer may write "YOLOv8, COCO dataset, anchor boxes, mAP, NMS" without ever mentioning "machine learning" or "computer vision".

## Domain Ontology

The hierarchical skill ontology `G = (V, E)` contains 22 canonical skill nodes and 346 alias strings.

### n-gram Phrase Matching

Given a normalised token sequence `T = [t1, ..., tm]`, the engine scans window sizes `n in {4, 3, 2, 1}` in decreasing order (longest-match priority). Match confidence for an n-gram hit is:

$$\text{conf}(n) = \min(1.0, 0.6 + 0.1 n)$$

### Parent-Node Propagation

When child skill `s_c` is detected with confidence `c_c`, each ancestor `s_p` at depth `d` is credited:

$$\text{conf}(s_p) \leftarrow \max\big(\text{conf}(s_p),\ c_c \cdot 0.8^d\big)$$

Detecting `computer_vision` (d = 1, child of `machine_learning`) with confidence 0.90 credits `machine_learning` with 0.90 * 0.8 = 0.72. This is the core mechanism by which "YOLOv8" implies ML expertise without the phrase "machine learning" appearing in the text.

## Semantic Embedding Engine

### TF-IDF Vectorisation

Following Salton and Buckley (1988), the vectoriser applies sublinear TF and smooth IDF:

$$\text{TF}_{\text{sub}}(t, d) = 1 + \log \text{count}(t, d)$$

$$\text{IDF}_{\text{smooth}}(t, D) = \log \frac{1 + |D|}{1 + \text{df}(t, D)} + 1$$

The vocabulary contains the top 10,000 terms (1-3 grams); all vectors are L2 normalised.

### Truncated SVD (LSA)

Following Deerwester et al. (1990), the TF-IDF matrix `X` is compressed using the randomised Halko-Martinsson-Tropp algorithm:

$$X \approx U_k \Sigma_k V_k^T,\quad k = 150$$

with 7 power iterations, achieving 27.8% explained variance and cosine similarity 0.535.

### Role Centroid and Cosine Similarity

For each category, a centroid `c_R` is the L2-normalised mean of all embeddings in that category. The semantic score component is:

$$S_{\text{sem}} = \frac{\cos(r, c_R) + 1}{2} \in [0, 1]$$

## Multi-Dimensional Scoring Engine

The overall match score combines four weighted components:

$$\text{Score} = 0.45\ S_{\text{ont}} + 0.30\ S_{\text{sem}} + 0.15\ S_{\text{dep}} + 0.10\ S_{\text{cor}}$$

capped at 0.97 to prevent false certainty.

### Ontology Sub-Score

Non-linear coverage adjustment rewards near-complete coverage disproportionately:

$$S_{\text{ont}} = \frac{\sum_{s \in R} w_s \left[1 - (1 - \text{cov}(s))^{1.5}\right]}{\sum_{s \in R} w_s}$$

where `cov(s) = min(1, c_s / w_s)`.

### Depth Sub-Score

$$S_{\text{dep}} = 0.35\ S_{\text{exp}} + 0.25\ S_{\text{edu}} + 0.20\ S_{\text{quant}} + 0.20\ S_{\text{tool}}$$

### Corpus Sub-Score

Mean cosine similarity to the top-5 most similar resumes in the target category.

## Shapley Value Attribution

Following Shapley (1953), let `F` be the extracted skill features and `v(S)` the ontology match score for coalition `S`. The Shapley value of skill `i` is:

$$\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|! (|F| - |S| - 1)!}{|F|!} \cdot \big[v(S \cup \{i\}) - v(S)\big]$$

Exact computation is used for `|F| <= 12`; Kernel SHAP (512 permutations) otherwise. This satisfies four axioms:

- **Efficiency:** the sum of Shapley values equals the total surplus.
- **Symmetry:** interchangeable features receive equal attribution.
- **Dummy:** features that never change the outcome receive zero attribution.
- **Additivity:** for compositions of two coalitional games, attributions add.

## LIME Local Explanation

LIME (Ribeiro et al., 2016) generates `N = 300` perturbed skill vectors:

$$\tilde{x}_j = x \odot m_j,\quad m_j \sim \text{Bernoulli}(0.5)^n$$

weighted by proximity kernel with bandwidth `sigma = 0.25`, and fits weighted ridge regression:

$$\beta = (M^T W M + \alpha I)^{-1} M^T W y,\quad \alpha = 0.01$$

The coefficient vector `beta` constitutes the LIME local explanation.

## Counterfactual Explanations

For each missing skill `s` (where `c_s < 0.40 * w_s`), the exact score gain from full acquisition is:

$$\Delta_s = v(c \oplus \{s : w_s\}) - v(c)$$

Skills are labelled Top Priority (`Delta > 12%`), High (`> 6%`), or Medium otherwise.

## Feature Interaction Values

Pairwise Shapley interaction values:

$$\phi_{ij} = v(\text{both}) - v(\text{only i}) - v(\text{only j}) + v(\text{neither})$$

Positive values indicate synergy; negative values indicate redundancy.

## Experimental Setup

All experiments use the full 2,484-resume corpus. Classification employs sklearn's `TfidfVectorizer` (40,000 features, 1-3 grams, sublinear TF, `min_df=1`) combined with `LinearSVC` (C=2.0, max_iter=3000) under 5-fold stratified cross-validation (`random_state=42`), running on CPU without GPU acceleration.

## Datasets

### Dataset 1: `clean_resume_data.csv`

Contains 2,484 real-world resumes across 24 job categories (ID, Category, Feature). One NaN entry in the Feature column (row 656, BUSINESS-DEVELOPMENT) is filled with an empty string before training. Average resume length is 587 words (median 549, range 77-3565).

### Dataset 2: `jobs_dataset_with_features.csv`

Contains over 1,000,000 job postings with Role and Features columns (approximately 600 MB total). A 50,000-row sample is used for exploratory analysis. This dataset provides ground-truth employer skill requirements for building weighted role profiles.

## Results Summary

### Classification Performance (LinearSVC, 5-Fold CV)

| Metric | Value |
|:---|:---:|
| Overall Accuracy | 70.73% |
| Macro Precision | 69.37% |
| Macro Recall | 66.45% |
| Macro F1-Score | 66.05% |
| Weighted Precision | 70.41% |
| Weighted Recall | 70.73% |
| Weighted F1-Score | 69.42% |

### Best Categories by F1

| Category | F1 |
|:---|:---:|
| Designer | 0.86 |
| Accountant | 0.82 |

### Lowest Category F1

| Category | F1 | Sample Size |
|:---|:---:|:---:|
| BPO | 0.15 | 22 |

### ROC-AUC by Category (One-vs-Rest, Calibrated SVM)

| Category | AUC |
|:---|:---:|
| HR | 0.9925 |
| Information Technology | 0.9868 |
| Finance | 0.9851 |
| Healthcare | 0.9623 |
| Advocate | 0.9313 |

### SHAP-LIME Agreement

100% directional agreement across all evaluated resumes and features.

### Comparative Study

For the Machine Learning Engineer role on the same test resume:

| Method | Score |
|:---|:---:|
| B1: Keyword Matching | ~35% |
| B2: TF-IDF Cosine (semantic-only) | 38% |
| B3: Ontology only | ~62% |
| EXAI Full System | 74% |

The +36 percentage-point gap between the semantic baseline and EXAI comes from ontological inference alone.

## References

- Deerwester, S., et al. (1990). Indexing by Latent Semantic Analysis. JASIS.
- Halko, N., Martinsson, P.-G., Tropp, J. A. (2011). Finding Structure with Randomness. SIAM Review.
- Lundberg, S. M., Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS.
- Ribeiro, M. T., Singh, S., Guestrin, C. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. KDD.
- Salton, G., Buckley, C. (1988). Term-weighting approaches in automatic text retrieval.
- Shapley, L. S. (1953). A value for n-person games. Contributions to the Theory of Games.
