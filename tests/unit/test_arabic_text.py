from __future__ import annotations

import pytest

from lsparabic.utils.arabic_text import (
    compute_cer,
    compute_wer,
    normalize_arabic,
    strip_diacritics,
)


def test_strip_diacritics():
    text = "مَرْحَبًا"
    result = strip_diacritics(text)
    assert result == "مرحبا"


def test_normalize_arabic_alef_variants():
    text = "أهلاً وآأإا"
    result = normalize_arabic(text)
    assert "أ" not in result
    assert "إ" not in result
    assert "آ" not in result


def test_normalize_arabic_removes_tatweel():
    text = "مرحـــبا"
    result = normalize_arabic(text)
    assert "ـ" not in result


def test_normalize_arabic_collapses_whitespace():
    text = "مرحبا   بكم"
    result = normalize_arabic(text)
    assert "   " not in result
    assert result == "مرحبا بكم"


def test_wer_identical():
    assert compute_wer("مرحبا", "مرحبا") == pytest.approx(0.0)


def test_wer_completely_wrong():
    wer = compute_wer("شيء آخر", "مرحبا بكم")
    assert wer > 0.0


def test_cer_identical():
    assert compute_cer("مرحبا", "مرحبا") == pytest.approx(0.0)
