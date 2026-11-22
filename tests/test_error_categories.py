"""Tests for the exclusive error categories approved 22 Nov 2026 (hand-written words only)."""

from __future__ import annotations

import pytest

from dfstr.evaluation.errors import (
    CATEGORIES,
    COMBINED,
    D_STROKE,
    MODIFIER_WRONG,
    TONE_ADDED,
    TONE_MISSING,
    TONE_WRONG,
    categorize,
)
from dfstr.evaluation.metrics import evaluate, score_word


@pytest.mark.parametrize(
    ("prediction", "reference", "expected"),
    [
        ("meo", "m\u00e8o", TONE_MISSING),  # meo vs mèo
        ("m\u00e8o", "meo", TONE_ADDED),  # mèo vs meo
        ("m\u00e9o", "m\u00e8o", TONE_WRONG),  # méo vs mèo
        ("cay", "c\u00e2y", MODIFIER_WRONG),  # cay vs cây
        ("da", "\u0111a", D_STROKE),  # da vs đa
        ("cay", "c\u1eady", COMBINED),  # cay vs cậy: modifier and tone at one position
        ("duong", "\u0111\u01b0\u1eddng", COMBINED),  # duong vs đường: three positions
    ],
)
def test_categorize_names_the_single_error(prediction: str, reference: str, expected: str) -> None:
    assert categorize(prediction, reference) == expected


def test_every_category_is_reachable_and_declared() -> None:
    pairs = {
        TONE_MISSING: ("meo", "m\u00e8o"),
        TONE_ADDED: ("m\u00e8o", "meo"),
        TONE_WRONG: ("m\u00e9o", "m\u00e8o"),
        MODIFIER_WRONG: ("cay", "c\u00e2y"),
        D_STROKE: ("da", "\u0111a"),
        COMBINED: ("cay", "c\u1eady"),
    }
    assert set(pairs) == set(CATEGORIES)
    for expected, (prediction, reference) in pairs.items():
        assert categorize(prediction, reference) == expected


def test_categorize_rejects_base_wrong_words() -> None:
    with pytest.raises(ValueError, match="not base-correct"):
        categorize("meo", "m\u00e8o\u1ee3")  # extra base unit


def test_categorize_rejects_identical_words() -> None:
    with pytest.raises(ValueError, match="identical after normalization"):
        categorize("m\u00e8o", "m\u00e8o")


def test_tone_placement_is_never_a_category() -> None:
    # hòa vs hoà is base-correct, strict-inexact, and free of diacritic error after normalization
    scored = score_word("h\u00f2a", "ho\u00e0")
    assert scored.error_category is None
    assert not scored.diacritic_error


def test_categories_are_exclusive_and_sum_to_the_der_numerator() -> None:
    references = [
        "m\u00e8o",  # mèo: tone missing
        "c\u00e2y",  # cây: modifier wrong
        "\u0111a",  # đa: d stroke
        "c\u1eady",  # cậy: combined
        "b\u00e0n",  # bàn: exact, no category
        "xyz",  # base wrong, not in K
    ]
    predictions = ["meo", "cay", "da", "cay", "b\u00e0n", "abc"]
    metrics = evaluate(predictions, references)
    categories = metrics["error_categories"]
    assert metrics["diacritic_errors"] == 4
    assert sum(categories.values()) == metrics["diacritic_errors"]
    assert categories == {
        TONE_MISSING: 1,
        TONE_WRONG: 0,
        TONE_ADDED: 0,
        MODIFIER_WRONG: 1,
        D_STROKE: 1,
        COMBINED: 1,
    }
