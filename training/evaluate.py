"""
EXAI-ResumeIntel: LinearSVC 5-fold stratified cross-validation
===============================================================

Reproduces the paper's classification metrics on the 2,484-resume corpus
using sklearn's TfidfVectorizer + LinearSVC. This is the same procedure
that produced Table II (70.73% overall accuracy) and the ROC-AUC scores
in Table III.

Usage:
    python -m training.evaluate

Outputs:
    - Confusion matrix
    - Per-category precision, recall, F1
    - Overall accuracy and macro/weighted metrics

Module: training.evaluate
Author: Mithin Sagar S
GitHub: https://github.com/mithinsagar
Institution: Vellore Institute of Technology (VIT)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

from core.constants import RESUME_CORPUS_PATH


def load_corpus(csv_path: Path) -> tuple[list[str], list[str]]:
    """Load the resume corpus from CSV."""
    df = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip")
    df["Feature"] = df["Feature"].fillna("").astype(str)
    df["Category"] = df["Category"].astype(str).str.strip().str.upper()
    df = df[df["Feature"].str.len() > 0]
    return df["Feature"].tolist(), df["Category"].tolist()


def run_cross_validation(
    documents: list[str],
    labels: list[str],
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """Run LinearSVC 5-fold stratified cross-validation."""
    y = np.array(labels)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    all_y_true = []
    all_y_pred = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(documents, y), start=1):
        X_train_text = [documents[i] for i in train_idx]
        X_test_text = [documents[i] for i in test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        vectorizer = TfidfVectorizer(
            max_features=40000,
            ngram_range=(1, 3),
            sublinear_tf=True,
            min_df=1,
        )
        X_train = vectorizer.fit_transform(X_train_text)
        X_test = vectorizer.transform(X_test_text)

        clf = LinearSVC(C=2.0, max_iter=3000, random_state=random_state)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print(f"  Fold {fold_idx}: accuracy = {acc:.4f} (n_test = {len(y_test)})")

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

    return {
        "y_true": np.array(all_y_true),
        "y_pred": np.array(all_y_pred),
    }


def print_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Print overall and per-category metrics."""
    overall_acc = accuracy_score(y_true, y_pred)
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    print()
    print("=" * 60)
    print("  Overall Classification Performance")
    print("=" * 60)
    print(f"  Overall Accuracy   : {overall_acc * 100:.2f}%")
    print(f"  Macro Precision    : {macro_prec * 100:.2f}%")
    print(f"  Macro Recall       : {macro_rec * 100:.2f}%")
    print(f"  Macro F1-Score     : {macro_f1 * 100:.2f}%")
    print(f"  Weighted Precision : {weighted_prec * 100:.2f}%")
    print(f"  Weighted Recall    : {weighted_rec * 100:.2f}%")
    print(f"  Weighted F1-Score  : {weighted_f1 * 100:.2f}%")
    print()
    print("=" * 60)
    print("  Per-Category Report")
    print("=" * 60)
    print(classification_report(y_true, y_pred, zero_division=0))


def main() -> None:
    """Load corpus and run 5-fold CV."""
    print("=" * 60)
    print("  EXAI-ResumeIntel : LinearSVC 5-Fold Cross-Validation")
    print("  Author: Mithin Sagar S")
    print("=" * 60)

    if not RESUME_CORPUS_PATH.exists():
        print(f"[ERROR] Resume corpus not found: {RESUME_CORPUS_PATH}")
        print("Run: bash scripts/download_data.sh")
        sys.exit(1)

    print(f"\n[1] Loading corpus from {RESUME_CORPUS_PATH}")
    documents, labels = load_corpus(RESUME_CORPUS_PATH)
    print(f"    {len(documents)} documents, {len(set(labels))} categories")

    print(f"\n[2] Running 5-fold stratified cross-validation")
    results = run_cross_validation(documents, labels)

    print_metrics(results["y_true"], results["y_pred"])


if __name__ == "__main__":
    main()
