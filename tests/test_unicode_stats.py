"""Tests for the audit-only Unicode helpers (hand-written words, no dataset labels)."""

from __future__ import annotations

import unicodedata

import pytest

from dfstr.data.audit import normalize_class
from dfstr.data.unicode_stats import (
    base_of,
    case_class,
    char_classes,
    decompose_char,
    tone_placement,
)


def test_decompose_char_reports_base_modifier_tone() -> None:
    assert decompose_char("\u1ebf") == ("e", "circumflex", "sac")  # ế
    assert decompose_char("\u0111") == ("d", "stroke", "none")  # đ
    assert decompose_char("a") == ("a", "none", "none")


def test_base_of_drops_marks_and_maps_stroke_d() -> None:
    assert base_of("\u0111i") == "di"  # đi
    assert base_of("\u0110I") == "DI"  # ĐI
    assert base_of(unicodedata.normalize("NFD", "ho\u00e0")) == "hoa"


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("h\u00f2a", "old"),  # hòa: tone on o
        ("ho\u00e0", "new"),  # hoà: tone on a
        ("kh\u1ecfe", "old"),  # khỏe
        ("kho\u1ebb", "new"),  # khoẻ
        ("th\u1ee7y", "old"),  # thủy
        ("thu\u1ef7", "new"),  # thuỷ
        ("qua\u0300", None),  # quà: u belongs to the onset
        ("hoa", None),  # no tone at all
    ],
)
def test_tone_placement(word: str, expected: str | None) -> None:
    assert tone_placement(unicodedata.normalize("NFC", word)) == expected


def test_tone_placement_independent_of_normalization() -> None:
    assert tone_placement(unicodedata.normalize("NFD", "h\u00f2a")) == "old"


@pytest.mark.parametrize(
    ("word", "expected"),
    [("hoa", "lower"), ("HOA", "upper"), ("Hoa", "title"), ("hOa", "mixed"), ("123", "uncased")],
)
def test_case_class(word: str, expected: str) -> None:
    assert case_class(word) == expected


def test_char_classes_ignores_combining_marks() -> None:
    counts = char_classes(unicodedata.normalize("NFD", "h\u00f2a 1"))
    assert counts["L"] == 3
    assert counts["N"] == 1
    assert counts["Z"] == 1


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("abc", "plain"),
        ("h\u00f2a", "nfc"),
        (unicodedata.normalize("NFD", "h\u00f2a"), "nfd"),
        ("h" + "\u00f2" + unicodedata.normalize("NFD", "a\u0300"), "mixed"),
    ],
)
def test_normalize_class(text: str, expected: str) -> None:
    assert normalize_class(text) == expected
