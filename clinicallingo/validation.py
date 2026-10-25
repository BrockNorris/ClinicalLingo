"""Stage: Validation.  RawInput -> ValidatedRequest, or InputValidationError.

Checks (in order):
    1. mode is one of word | sentence | page
    2. control characters are stripped (newlines/tabs kept only in page mode) and
       whitespace is collapsed
    3. text is non-empty after cleaning
    4. text length is within the per-mode cap (config.INPUT_LIMITS)
    5. word mode: a single term (at most 3 whitespace-separated tokens, e.g. "common bile duct")

Nothing that fails here ever reaches retrieval or the model.
"""
from __future__ import annotations

import re
import unicodedata

from .config import INPUT_LIMITS
from .contracts import RawInput, ValidatedRequest


class InputValidationError(ValueError):
    def __init__(self, reason: str, field: str = "text"):
        super().__init__(reason)
        self.reason = reason
        self.field = field


def _clean(text: str, keep_newlines: bool) -> str:
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if ch in "\n\t":
            out.append("\n" if keep_newlines and ch == "\n" else " ")
        elif cat.startswith("C"):  # control / format / unassigned
            continue
        else:
            out.append(ch)
    s = "".join(out)
    s = re.sub(r"[ \r\f\v]+", " ", s)
    if keep_newlines:
        s = re.sub(r"\n{3,}", "\n\n", s)
        s = "\n".join(line.strip() for line in s.split("\n"))
    return s.strip()


def validate(raw: RawInput) -> ValidatedRequest:
    mode = (raw.mode or "").strip().lower()
    if mode not in INPUT_LIMITS:
        raise InputValidationError(
            f"mode must be one of {sorted(INPUT_LIMITS)}, got {raw.mode!r}", field="mode"
        )

    if not isinstance(raw.text, str):
        raise InputValidationError(f"text must be a string, got {type(raw.text).__name__}")

    text = _clean(raw.text, keep_newlines=(mode == "page"))
    if not text:
        raise InputValidationError("text is empty after removing whitespace/control characters")

    limit = INPUT_LIMITS[mode]
    if len(text) > limit:
        raise InputValidationError(
            f"text is {len(text)} chars; limit for mode '{mode}' is {limit}"
        )

    if mode == "word" and len(text.split()) > 3:
        raise InputValidationError(
            "word mode expects a single term (up to 3 words); use --mode sentence for longer text"
        )

    return ValidatedRequest(text=text, mode=mode, char_count=len(text))
