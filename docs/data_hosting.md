# Data and Model Hosting Guide

Complete guide for hosting the large (approximately 5 GB) EXAI-ResumeIntel datasets and precomputed model artifacts on Hugging Face Hub.

**Author:** Mithin Sagar S ([@mithinsagar](https://github.com/mithinsagar))

## Why Hugging Face Hub

For public research and open-source ML projects, Hugging Face is the industry-standard hosting platform. The relevant advantages:

- **Free unlimited public storage.** No file size caps for individual files.
- **CDN-backed downloads.** Fast globally, no bandwidth caps for reasonable use.
- **Git-based versioning.** Datasets and models are just Git repos with LFS built in.
- **Programmatic access.** One-line loading via the `huggingface_hub` and `datasets` libraries.
- **Dataset and model cards.** Auto-rendered markdown pages that look professional.
- **Optional DOI.** Zenodo integration provides citable references.
- **Discoverability.** Search and community traffic through the HF marketplace.

Comparison with alternatives:

| Platform | Free Storage | Free Bandwidth | Public Discoverability | Best For |
|:---|:---:|:---:|:---:|:---|
| **Hugging Face Hub** | Unlimited | Unlimited (reasonable) | Excellent | ML datasets and models |
| Zenodo | 50 GB / record | Unlimited | Academic-only | Papers, DOI-required |
| Kaggle Datasets | 20 GB / dataset | Requires login | Excellent | Kaggle community |
| Google Drive | 15 GB total | Rate-limited | None | Personal file sharing |
| OSF | 5 GB / file | Unlimited | Academic | Research reproducibility |
| GitHub LFS | 1 GB (free tier) | 1 GB / month | Excellent | Small artifacts (< 100 MB) |
| GitHub Releases | 2 GB / file | Unlimited | Excellent | Distributable binaries |

For EXAI-ResumeIntel, the datasets (~615 MB total) and models (up to 4 GB depending on precomputed embeddings) exceed the free tiers of Git LFS and GitHub Releases. HF Hub is the only free option that scales.

## Prerequisites

Create a Hugging Face account:

1. Go to [huggingface.co/join](https://huggingface.co/join).
2. Create an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) with `write` scope.
3. Install the CLI:
   ```bash
   pip install huggingface_hub
   ```
4. Log in:
   ```bash
   huggingface-cli login
   # Paste your token when prompted
   ```

## Step 1: Create the Dataset Repository

The datasets belong in a Dataset repo (as opposed to a Model repo).

1. Create the repo at [huggingface.co/new-dataset](https://huggingface.co/new-dataset).
   - Owner: `mithinsagar`
   - Dataset name: `exai-resumeintel-data`
   - License: Choose (recommend `MIT`)
   - Visibility: Public

2. Upload the files:
   ```bash
   huggingface-cli upload mithinsagar/exai-resumeintel-data \
     data/raw/clean_resume_data.csv clean_resume_data.csv \
     --repo-type dataset

   huggingface-cli upload mithinsagar/exai-resumeintel-data \
     data/raw/jobs_dataset_with_features.csv jobs_dataset_with_features.csv \
     --repo-type dataset
   ```

3. Add a dataset card. Create `data/README.md` on the HF Hub (via web UI or via a git push) with:

```markdown
---
license: mit
language:
  - en
task_categories:
  - text-classification
tags:
  - resume
  - xai
  - shapley
size_categories:
  - 1M<n<10M
---

# EXAI-ResumeIntel Datasets

Datasets for the EXAI-ResumeIntel project.

- `clean_resume_data.csv`: 2,484 real resumes, 24 categories
- `jobs_dataset_with_features.csv`: ~1M job postings

Author: Mithin Sagar S | https://github.com/mithinsagar
Repository: https://github.com/mithinsagar/EXAI-ResumeIntel
```

## Step 2: Create the Model Repository

The precomputed embeddings and pickled engines belong in a Model repo.

1. Create the repo at [huggingface.co/new](https://huggingface.co/new).
   - Owner: `mithinsagar`
   - Model name: `exai-resumeintel-models`
   - License: `mit`
   - Visibility: Public

2. Upload the artifacts:
   ```bash
   huggingface-cli upload mithinsagar/exai-resumeintel-models \
     models/embedding_engine.pkl embedding_engine.pkl \
     --repo-type model

   huggingface-cli upload mithinsagar/exai-resumeintel-models \
     models/job_df.pkl job_df.pkl \
     --repo-type model

   huggingface-cli upload mithinsagar/exai-resumeintel-models \
     models/job_embeddings.memmap job_embeddings.memmap \
     --repo-type model
   ```

3. Add a model card explaining the artifacts.

## Step 3: Verify Downloads

Run the download script and confirm files land correctly:

```bash
bash scripts/download_data.sh
ls -lh data/raw/ models/
```

## Alternative: Zenodo (for citable DOI)

If you also want a DOI for the paper, mirror the artifacts to Zenodo:

1. Create an account at [zenodo.org](https://zenodo.org).
2. New Upload -> Dataset.
3. Upload the CSVs and pickles.
4. Fill in metadata (title, authors, description, license).
5. Publish. Zenodo assigns a permanent DOI.
6. Update `CITATION.cff` and README to reference the DOI.

Downloading from Zenodo:

```bash
wget https://zenodo.org/record/XXXXXXX/files/clean_resume_data.csv \
     -O data/raw/clean_resume_data.csv
```

## Alternative: Git LFS

Only viable if all your model artifacts stay under GitHub's free tier (1 GB storage, 1 GB monthly bandwidth). For EXAI-ResumeIntel, the 600 MB CSV alone will exhaust the bandwidth quota after two clones.

If you still want LFS, add `.gitattributes`:

```
*.csv     filter=lfs diff=lfs merge=lfs -text
*.pkl     filter=lfs diff=lfs merge=lfs -text
*.memmap  filter=lfs diff=lfs merge=lfs -text
```

Then:

```bash
git lfs install
git add .gitattributes data/raw/ models/
git commit -m "Track large files with LFS"
git push
```

## Recommended Strategy for EXAI-ResumeIntel

1. **Primary hosting:** Hugging Face Hub (datasets + models). Free, professional, scalable.
2. **Optional DOI:** Zenodo mirror for the paper citation.
3. **Repo strategy:** `.gitignore` excludes the large files. `data/README.md` and `models/README.md` link to HF. `scripts/download_data.sh` fetches them.

This keeps the GitHub repo lean (approximately 2 MB) while the actual data lives where it belongs.

## Files to Host

| File | Approx size | Where |
|:---|:---:|:---|
| `clean_resume_data.csv` | 15 MB | HF Dataset |
| `jobs_dataset_with_features.csv` | 600 MB | HF Dataset |
| `embedding_engine.pkl` | 5-50 MB | HF Model |
| `job_df.pkl` | ~50-200 MB | HF Model |
| `job_embeddings.memmap` | 3-4 GB | HF Model |

Total: approximately 4.7 GB, all served by HF Hub for free.
