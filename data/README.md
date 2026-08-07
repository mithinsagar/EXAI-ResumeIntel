# Data

This directory holds the raw and processed data used by EXAI-ResumeIntel.

**Author:** Mithin Sagar S ([@mithinsagar](https://github.com/mithinsagar))

## Contents

```
data/
├── raw/                              Raw input CSVs (not in Git)
│   ├── clean_resume_data.csv         2,484 resumes across 24 categories
│   └── jobs_dataset_with_features.csv  ~1M job postings with skill features
└── processed/                        Runtime artefacts (not in Git)
    └── analysis_history.db           SQLite history from the Streamlit app
```

## Getting the raw data

The raw CSVs are hosted on Hugging Face Hub because they exceed GitHub's file size limits.

**Automated download:**
```bash
bash scripts/download_data.sh
```

**Manual download:**
```bash
huggingface-cli download mithinsagar/exai-resumeintel-data \
  --repo-type dataset --local-dir data/raw
```

**Or via the web interface:**
[huggingface.co/datasets/mithinsagar/exai-resumeintel-data](https://huggingface.co/datasets/mithinsagar/exai-resumeintel-data)

## Dataset 1: `clean_resume_data.csv`

- **Rows:** 2,484
- **Columns:** ID, Category, Feature
- **Size:** approximately 15 MB
- **Description:** Real-world resumes across 24 job categories
  (INFORMATION-TECHNOLOGY, BUSINESS-DEVELOPMENT, FINANCE, ADVOCATE, ACCOUNTANT,
  ENGINEERING, CHEF, AVIATION, FITNESS, SALES, BANKING, HEALTHCARE, CONSULTANT,
  CONSTRUCTION, PUBLIC-RELATIONS, HR, DESIGNER, ARTS, TEACHER, APPAREL,
  DIGITAL-MEDIA, AGRICULTURE, AUTOMOBILE, BPO)
- **Average length:** 587 words (median 549, range 77-3,565)
- **Known issues:** One NaN entry in the Feature column (row 656) &mdash; filled with empty string before training
- **License:** MIT (same as parent project)

Used for training the semantic embedding engine (`training/train.py`) and reproducing the paper's LinearSVC classification results (`training/evaluate.py`).

## Dataset 2: `jobs_dataset_with_features.csv`

- **Rows:** approximately 1,000,000+
- **Columns:** Role, Features
- **Size:** approximately 600 MB
- **Description:** Job postings with extracted skill features
- **License:** MIT

Used by the Streamlit application (`app/`) to power the semantic role matching against real employer requirements. A 50,000-row sample is used for exploratory analysis in the paper (Fig. 3).

## `processed/`

The `data/processed/` directory is where the Streamlit application persists user history (`analysis_history.db`, an SQLite file). This directory is gitignored.

## Data usage in code

```python
from core.constants import RESUME_CORPUS_PATH, JOB_POSTINGS_PATH

# Load Dataset 1
import pandas as pd
df = pd.read_csv(RESUME_CORPUS_PATH)
print(f"{len(df)} resumes, {df['Category'].nunique()} categories")

# Load Dataset 2 (chunked for memory efficiency)
chunks = pd.read_csv(JOB_POSTINGS_PATH, chunksize=10000)
for chunk in chunks:
    process(chunk)
```
