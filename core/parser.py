"""
EXAI-ResumeIntel: Resume parser for structured signal extraction
================================================================

Extracts structured information from raw resume text:
    - Years of experience (regex-based)
    - Education level (0-5 scale)
    - Quantified achievements (percentage figures, multipliers, impact verbs)
    - Word count

Corresponds to paper Section III.B (Resume Parser).

Module: core.parser
Author: Mithin Sagar S
GitHub: https://github.com/mithinsagar
Institution: Vellore Institute of Technology (VIT)
"""

from __future__ import annotations

import re
from typing import Dict

from core.constants import EDUCATION_KEYWORDS


class ResumeParser:
    """
    Extracts structured information from raw resume text.

    Attributes
    ----------
    EXPERIENCE_PATTERNS : list of str
        Regex patterns for extracting years of experience.
    EDUCATION_KEYWORDS : dict
        Mapping from education keyword to numeric level (0-5).
    """

    EXPERIENCE_PATTERNS = [
        r"(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|exp)",
        r"(?:experience|exp)[:\s]+(\d+)\+?\s*(?:years?|yrs?)",
    ]

    EDUCATION_KEYWORDS = EDUCATION_KEYWORDS

    def __init__(self) -> None:
        pass

    def parse(self, text: str) -> Dict:
        """
        Parse a raw resume text and return structured fields.

        Parameters
        ----------
        text : str
            Raw resume text (any format).

        Returns
        -------
        dict
            Keys: raw_text, clean_text, years_experience, education_level,
            word_count, has_quantified_achievements.
        """
        text_lower = text.lower()
        return {
            "raw_text": text,
            "clean_text": self._clean(text),
            "years_experience": self._extract_experience(text_lower),
            "education_level": self._extract_education(text_lower),
            "word_count": len(text.split()),
            "has_quantified_achievements": self._has_quantified(text_lower),
        }

    def _clean(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\x20-\x7E]", " ", text)
        return text.strip()

    def _extract_experience(self, text: str) -> float:
        years = []
        for pattern in self.EXPERIENCE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    years.append(int(match.group(1)))
                except (IndexError, ValueError):
                    pass
        return float(max(years)) if years else 2.0

    def _extract_education(self, text: str) -> int:
        best = 0
        for keyword, level in self.EDUCATION_KEYWORDS.items():
            if keyword in text:
                best = max(best, level)
        return best

    def _has_quantified(self, text: str) -> bool:
        return bool(
            re.search(
                r"\d+%|\d+x\b|increased|decreased|improved|reduced|grew|saved|generated",
                text,
                re.IGNORECASE,
            )
        )
