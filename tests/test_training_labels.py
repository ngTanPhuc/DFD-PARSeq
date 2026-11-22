"""Tests for the label string stored with each crop (hand-written words only)."""

from __future__ import annotations

import unicodedata

import pytest

from dfstr.data.training_labels import NORMALIZED_SPLITS, training_label


@pytest.mark.parametrize(
    ("shipped", "expected"),
    [
        ("hòa", "hoà"),  # old placement in, new placement out
        ("hoà", "hoà"),  # already new placement
        ("khỏe", "khoẻ"),
        ("thủy", "thuỷ"),
        ("quà", "quà"),  # "qu" is an onset, the tone stays on "a"
        ("hoa", "hoa"),  # no tone at all
    ],
)
def test_train_labels_get_the_new_tone_placement(shipped: str, expected: str) -> None:
    assert training_label(shipped, "train") == expected


@pytest.mark.parametrize("split", ["validation", "test"])
def test_validation_and_test_labels_are_left_as_shipped(split: str) -> None:
    assert training_label("hòa", split) == "hòa"


def test_only_train_is_normalized() -> None:
    assert NORMALIZED_SPLITS == frozenset({"train"})


def test_labels_are_nfc() -> None:
    decomposed = unicodedata.normalize("NFD", "hoà")
    assert training_label(decomposed, "validation") == "hoà"


def test_case_is_preserved() -> None:
    assert training_label("HÒA", "train") == "HOÀ"
