from __future__ import annotations

import re
import unicodedata

# Arabic diacritics (harakat) Unicode ranges
_DIACRITICS_PATTERN = re.compile(
    r"[ً-ٰٟۖ-ۜ۟-۪ۤۧۨ-ۭ]"
)

# Tatweel (kashida) elongation character
_TATWEEL_PATTERN = re.compile(r"ـ")

# Alef variants → plain alef
_ALEF_VARIANTS = str.maketrans(
    {"آ": "ا", "أ": "ا", "إ": "ا", "ٱ": "ا"}
)

# Ya variants → dotless ya
_YA_VARIANTS = str.maketrans({"ى": "ي"})

# Ta marbuta → ha
_TA_MARBUTA = str.maketrans({"ة": "ه"})


def strip_diacritics(text: str) -> str:
    return _DIACRITICS_PATTERN.sub("", text)


def normalize_arabic(text: str) -> str:
    """Normalize Arabic text: strip diacritics, normalize letter variants, remove tatweel."""
    text = strip_diacritics(text)
    text = _TATWEEL_PATTERN.sub("", text)
    text = text.translate(_ALEF_VARIANTS)
    text = text.translate(_YA_VARIANTS)
    # Normalize Unicode (NFC)
    text = unicodedata.normalize("NFC", text)
    # Collapse whitespace
    text = " ".join(text.split())
    return text


def char_tokenize(text: str) -> list[str]:
    return list(text.replace(" ", "|"))


def compute_wer(hypothesis: str, reference: str) -> float:
    """Word Error Rate between two Arabic strings."""
    try:
        from jiwer import wer as _wer
        return float(_wer(reference, hypothesis))
    except ImportError:
        return _edit_distance_rate(reference.split(), hypothesis.split())


def compute_cer(hypothesis: str, reference: str) -> float:
    """Character Error Rate between two Arabic strings."""
    try:
        from jiwer import cer as _cer
        return float(_cer(reference, hypothesis))
    except ImportError:
        return _edit_distance_rate(list(reference), list(hypothesis))


def _edit_distance_rate(ref: list, hyp: list) -> float:
    if not ref:
        return float(len(hyp))
    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[m] / n
