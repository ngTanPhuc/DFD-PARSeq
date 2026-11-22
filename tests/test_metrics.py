"""Tests for the frozen metrics on hand-written, made-up words (never real VinText labels)."""

from __future__ import annotations

import unicodedata

import pytest

from dfstr.evaluation.components import base_letters, units
from dfstr.evaluation.metrics import (
    der_on_intersection,
    diacritic_error_share,
    evaluate,
    normalized_edit_distance,
    score_word,
)


def test_base_letters_strips_marks_and_folds_d_stroke() -> None:
    assert base_letters("b\u00e0n") == "ban"  # bàn
    assert base_letters("\u0111\u01b0\u1eddng") == "duong"  # đường
    assert base_letters("\u0110\u01af\u1edcNG") == "DUONG"  # ĐƯỜNG, case kept
    assert base_letters("abc123") == "abc123"


def test_base_letters_accepts_nfd_input() -> None:
    nfd = unicodedata.normalize("NFD", "b\u00e0n")
    assert base_letters(nfd) == "ban"


def test_units_name_modifier_and_tone() -> None:
    decomposed = units("thu\u1ef7")  # thuỷ
    assert [unit.base for unit in decomposed] == ["t", "h", "u", "y"]
    assert decomposed[-1].tone == "hoi"
    assert all(unit.modifier == "none" for unit in decomposed)
    assert units("\u01a1")[0].modifier == "horn"  # ơ
    assert units("\u00e2")[0].modifier == "circumflex"  # â


def test_units_keep_d_stroke_as_one_base() -> None:
    assert units("\u0111")[0].base == "\u0111"


def test_score_word_exact_word_has_no_error() -> None:
    scored = score_word("b\u00e0n", "b\u00e0n")
    assert scored.exact and scored.base_correct and not scored.diacritic_error
    assert scored.error_category is None
    assert scored.one_minus_ned == 1.0


def test_score_word_tone_placement_differs_on_strict_strings_only() -> None:
    scored = score_word("ho\u00e0", "h\u00f2a")  # hoà vs hòa
    assert scored.base_correct
    assert not scored.exact, "word accuracy stays strict"
    assert not scored.diacritic_error, "DER is computed on tone-normalized strings"
    assert scored.error_category is None


def test_score_word_accepts_nfd_predictions() -> None:
    # The release ships NFC labels, but a decoder can hand over NFD text; NFC is applied first
    nfd = unicodedata.normalize("NFD", "b\u00e0n")
    scored = score_word(nfd, "b\u00e0n")
    assert scored.exact and scored.one_minus_ned == 1.0


def test_score_word_rejects_an_empty_reference() -> None:
    with pytest.raises(ValueError, match="reference label is empty"):
        score_word("ban", "")


def test_score_words_rejects_misaligned_input() -> None:
    with pytest.raises(ValueError, match="2 predictions vs 1 references"):
        evaluate(["ban", "meo"], ["b\u00e0n"])


def test_evaluate_counts_denominators_and_component_rates() -> None:
    references = ["ho\u00e0", "m\u00e8o", "c\u00e0", "abc", "h\u00f2a"]
    predictions = ["ho\u00e0", "meo", "ca", "abd", "ho\u00e0"]
    metrics = evaluate(predictions, references)
    assert metrics["words"] == 5
    assert metrics["base_correct_words"] == 4
    assert metrics["exact_words"] == 1
    assert metrics["diacritic_errors"] == 2
    assert metrics["base_accuracy"] == pytest.approx(4 / 5)
    assert metrics["der"] == pytest.approx(2 / 4)
    assert metrics["uder"] == pytest.approx(2 / 5)
    assert metrics["word_accuracy"] == pytest.approx(1 / 5)
    assert metrics["modifier_accuracy"] == pytest.approx(4 / 4)
    assert metrics["tone_accuracy"] == pytest.approx(2 / 4)
    assert metrics["error_categories"]["tone_missing"] == 2
    assert sum(metrics["error_categories"].values()) == metrics["diacritic_errors"]
    assert metrics["one_minus_ned"] == pytest.approx((1.0 + 2 / 3 + 0.5 + 2 / 3 + 1 / 3) / 5)


def test_evaluate_rejects_a_category_missing_from_the_report_list(monkeypatch) -> None:
    monkeypatch.setattr(
        "dfstr.evaluation.metrics.categorize", lambda prediction, reference: "bogus"
    )
    with pytest.raises(ValueError, match="unknown error categories \\['bogus'\\]"):
        evaluate(["meo"], ["m\u00e8o"])


def test_word_accuracy_identity_holds_on_strict_strings() -> None:
    # No oa/oe/uy pair, so tone normalization changes nothing and the identity is exact
    metrics = evaluate(["b\u00e0n", "meo", "cay"], ["b\u00e0n", "m\u00e8o", "c\u00e2y"])
    base_accuracy = metrics["base_accuracy"]
    der = metrics["der"]
    assert metrics["word_accuracy"] == pytest.approx(base_accuracy * (1 - der))
    assert metrics["word_accuracy"] == pytest.approx(1 / 3)


def test_evaluate_empty_input_reports_none_not_zero() -> None:
    metrics = evaluate([], [])
    assert metrics["words"] == 0
    assert metrics["der"] is None
    assert metrics["uder"] is None
    assert metrics["word_accuracy"] is None
    assert metrics["one_minus_ned"] is None
    assert set(metrics["error_categories"].values()) == {0}


def test_der_on_intersection_scores_the_shared_subset() -> None:
    references = ["b\u00e0n", "m\u00e8o", "c\u00e2y"]
    predictions = {
        "model_a": ["ban", "meo", "cay"],
        "model_b": ["ban", "m\u00e8o", "xyz"],
    }
    result = der_on_intersection(predictions, references)
    assert result["words"] == 3
    assert result["intersection_words"] == 2
    assert result["excluded_words"] == 1
    assert result["models"]["model_a"]["der"] == pytest.approx(2 / 2)
    assert result["models"]["model_b"]["der"] == pytest.approx(1 / 2)


def test_der_on_intersection_needs_a_model() -> None:
    with pytest.raises(ValueError, match="at least one model"):
        der_on_intersection({}, ["b\u00e0n"])


def test_der_on_intersection_empty_intersection_has_no_rate() -> None:
    result = der_on_intersection({"m": ["xyz"]}, ["b\u00e0n"])
    assert result["intersection_words"] == 0
    assert result["models"]["m"]["der"] is None


@pytest.mark.parametrize(
    ("prediction", "reference", "expected"),
    [
        ("abc", "abc", 0.0),
        ("abc", "abd", 1 / 3),
        ("", "abc", 1.0),
        ("", "", 0.0),
    ],
)
def test_normalized_edit_distance(prediction: str, reference: str, expected: float) -> None:
    assert normalized_edit_distance(prediction, reference) == pytest.approx(expected)


def test_diacritic_error_share_is_k_over_wrong_words() -> None:
    assert diacritic_error_share(errors=3, words=10, exact=4) == pytest.approx(3 / 6)
    assert diacritic_error_share(errors=0, words=10, exact=10) is None
