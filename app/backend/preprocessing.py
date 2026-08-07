"""
EXAI-ResumeIntel: Resume PDF/DOCX extraction and tokenization
=============================================================

Module: app.backend.preprocessing
Author: Mithin Sagar S
GitHub: https://github.com/mithinsagar
Institution: Vellore Institute of Technology (VIT)
"""

from __future__ import annotations

from io import BytesIO
from typing import Tuple

from PyPDF2 import PdfReader
from docx import Document
from transformers import AutoTokenizer


MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return _tokenizer


def _extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(file_bytes))
    texts = []
    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
            texts.append(page_text)
        except Exception:
            continue
    return "\n".join(texts)


def _extract_text_from_docx_bytes(file_bytes: bytes) -> str:
    bio = BytesIO(file_bytes)
    doc = Document(bio)
    paragraphs = [p.text for p in doc.paragraphs if p.text]
    return "\n".join(paragraphs)


def extract_text_from_document(file_source, filename: str | None = None) -> str:
    """
    Extract text from a resume (PDF or DOCX).

    `file_source` can be a Streamlit UploadedFile or raw bytes.
    """
    try:
        if hasattr(file_source, "read"):
            file_bytes = file_source.read()
            name = getattr(file_source, "name", "") or filename or ""
        else:
            file_bytes = file_source
            name = filename or ""

        lower_name = name.lower()

        if lower_name.endswith(".pdf"):
            return _extract_text_from_pdf_bytes(file_bytes)
        if lower_name.endswith(".docx"):
            return _extract_text_from_docx_bytes(file_bytes)

        # Fallback: try PDF first, then DOCX
        try:
            return _extract_text_from_pdf_bytes(file_bytes)
        except Exception:
            return _extract_text_from_docx_bytes(file_bytes)
    except Exception as e:
        raise RuntimeError(f"Document extraction failed: {e}") from e


def preprocess_resume_text(text: str) -> Tuple[str, int]:
    """
    Basic normalization + token counting using the transformer tokenizer.
    """
    normalized = " ".join(text.replace("\r", " ").replace("\n", " ").split())
    tokenizer = _get_tokenizer()
    tokens = tokenizer.tokenize(normalized)
    token_count = len(tokens)
    return normalized, token_count


