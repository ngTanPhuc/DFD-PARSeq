"""Tests for the tone-placement normalization shared by data prep and metrics."""

from __future__ import annotations

import unicodedata

import pytest

from dfstr.labels.orthography import tone_normalize


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("h\u00f2a", "ho\u00e0"),  # hòa → hoà
        ("kho\u0309e", "kho\u1ebb"),  # khỏe → khoẻ
        ("thu\u0309y", "thu\u1ef7"),  # thủy → thuỷ
        ("ho\u00e0", "ho\u00e0"),  # already new
        ("qu\u00e0", "qu\u00e0"),  # qu onset is not moved
        ("hoa", "hoa"),  # no tone
        ("", ""),
    ],
)
def test_tone_normalize_moves_tone_to_second_vowel(word: str, expected: str) -> None:
    assert tone_normalize(unicodedata.normalize("NFC", word)) == unicodedata.normalize(
        "NFC", expected
    )


def test_tone_normalize_keeps_case() -> None:
    assert tone_normalize("H\u00d2A") == "HO\u00c0"  # HÒA → HOÀ


def test_tone_normalize_is_idempotent_and_returns_nfc() -> None:
    once = tone_normalize("h\u00f2a")
    assert tone_normalize(once) == once
    assert once == unicodedata.normalize("NFC", once)


def test_tone_normalize_accepts_nfd_input() -> None:
    assert tone_normalize(unicodedata.normalize("NFD", "h\u00f2a")) == "ho\u00e0"


def test_tone_normalize_handles_several_pairs_in_one_label() -> None:
    assert tone_normalize("h\u00f2a thu\u0309y") == "ho\u00e0 thu\u1ef7"


def test_tone_normalize_leaves_other_marks_in_place() -> None:
    # ằ carries a breve and a tone; the pair is ă + no pair partner, so nothing moves
    assert tone_normalize("\u1eb1m") == "\u1eb1m"
