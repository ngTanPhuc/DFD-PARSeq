"""The frozen metrics, computed from decoded Unicode strings only.

Definitions (``README.md`` "Evaluation protocol", ``docs/evaluation-protocol.md``):

- Ground truth and prediction are normalized to NFC, case is kept.
- A word is **base-correct** when :func:`dfstr.evaluation.components.base_letters` agrees.
- A word is **exact** when the NFC strings are equal with no tone-placement normalization.
- ``K`` is the number of base-correct words whose tone-normalized strings still differ.
- ``DER = K / A``, ``UDER = K / N``, word accuracy ``= E / N``.
- DER on the intersection set re-scores only the words that are base-correct in every model
  compared, which removes the conditional-denominator advantage of plain DER.
- 1-NED uses the strict NFC strings, with the Levenshtein distance divided by the longer length.
- Modifier and tone accuracy are word-level rates over the base-correct words, the same
  denominator DER uses, so a word with a base error never counts as a diacritic success.
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rapidfuzz.distance import Levenshtein

from dfstr.evaluation.components import base_letters, units
from dfstr.evaluation.errors import CATEGORIES, categorize
from dfstr.labels.orthography import tone_normalize


@dataclass(frozen=True)
class WordScore:
    """One prediction scored against its reference label."""

    prediction: str
    reference: str
    base_correct: bool
    exact: bool
    diacritic_error: bool
    error_category: str | None
    modifier_correct: bool | None
    tone_correct: bool | None
    one_minus_ned: float


def normalized_edit_distance(prediction: str, reference: str) -> float:
    # Input: two strings; Output: 0.0 for identical, 1.0 when nothing matches
    # How: Levenshtein distance over code points, divided by the longer length
    longest = max(len(prediction), len(reference))
    if longest == 0:
        return 0.0
    return Levenshtein.distance(prediction, reference) / longest


def score_word(prediction: str, reference: str) -> WordScore:
    # Input: a decoded prediction string and its reference label
    # Output: the word's flags, component results and category
    if not reference:
        raise ValueError("reference label is empty; unusable regions must be dropped, not scored")
    predicted_nfc = unicodedata.normalize("NFC", prediction)
    reference_nfc = unicodedata.normalize("NFC", reference)
    predicted_normalized = tone_normalize(predicted_nfc)
    reference_normalized = tone_normalize(reference_nfc)
    exact = predicted_nfc == reference_nfc
    base_correct = base_letters(predicted_nfc) == base_letters(reference_nfc)
    diacritic_error = base_correct and predicted_normalized != reference_normalized
    category = categorize(predicted_normalized, reference_normalized) if diacritic_error else None
    modifier_correct: bool | None = None
    tone_correct: bool | None = None
    if base_correct:
        predicted_units = units(predicted_normalized)
        reference_units = units(reference_normalized)
        modifier_correct = all(
            predicted.modifier == expected.modifier
            for predicted, expected in zip(predicted_units, reference_units, strict=True)
        )
        tone_correct = all(
            predicted.tone == expected.tone
            for predicted, expected in zip(predicted_units, reference_units, strict=True)
        )
    return WordScore(
        prediction=prediction,
        reference=reference,
        base_correct=base_correct,
        exact=exact,
        diacritic_error=diacritic_error,
        error_category=category,
        modifier_correct=modifier_correct,
        tone_correct=tone_correct,
        one_minus_ned=1.0 - normalized_edit_distance(predicted_nfc, reference_nfc),
    )


def score_words(predictions: Sequence[str], references: Sequence[str]) -> tuple[WordScore, ...]:
    # Input: aligned prediction and reference strings; Output: one score per word
    if len(predictions) != len(references):
        raise ValueError(f"{len(predictions)} predictions vs {len(references)} references")
    return tuple(
        score_word(prediction, reference)
        for prediction, reference in zip(predictions, references, strict=True)
    )


def _rate(numerator: int, denominator: int) -> float | None:
    # Input: a count and its denominator; Output: the rate, or None when the denominator is empty
    return None if denominator == 0 else numerator / denominator


def evaluate(predictions: Sequence[str], references: Sequence[str]) -> dict[str, object]:
    # Input: aligned prediction and reference strings for one run
    # Output: the metric block written to results/metrics, with every denominator
    scores = score_words(predictions, references)
    n = len(scores)
    base_correct = sum(1 for score in scores if score.base_correct)
    exact = sum(1 for score in scores if score.exact)
    diacritic_errors = sum(1 for score in scores if score.diacritic_error)
    modifier_correct = sum(1 for score in scores if score.modifier_correct)
    tone_correct = sum(1 for score in scores if score.tone_correct)
    categories = Counter(score.error_category for score in scores if score.error_category)
    unknown = set(categories) - set(CATEGORIES)
    if unknown:
        raise ValueError(f"unknown error categories {sorted(unknown)}; add them to CATEGORIES")
    if sum(categories.values()) != diacritic_errors:
        raise ValueError(
            f"error categories sum to {sum(categories.values())}, DER numerator is "
            f"{diacritic_errors}"
        )
    return {
        "words": n,
        "base_correct_words": base_correct,
        "exact_words": exact,
        "diacritic_errors": diacritic_errors,
        "base_accuracy": _rate(base_correct, n),
        "der": _rate(diacritic_errors, base_correct),
        "uder": _rate(diacritic_errors, n),
        "word_accuracy": _rate(exact, n),
        "one_minus_ned": (None if n == 0 else sum(score.one_minus_ned for score in scores) / n),
        "modifier_accuracy": _rate(modifier_correct, base_correct),
        "tone_accuracy": _rate(tone_correct, base_correct),
        "error_categories": {category: categories.get(category, 0) for category in CATEGORIES},
    }


def der_on_intersection(
    predictions_by_model: Mapping[str, Sequence[str]], references: Sequence[str]
) -> dict[str, object]:
    # Input: predictions of every compared model, aligned to one reference list
    # Output: DER per model over the words all models get base-correct, plus the counts
    # How: one base-correct mask across models; every model is then scored on exactly that subset
    if not predictions_by_model:
        raise ValueError("der_on_intersection needs at least one model")
    scores = {
        model: score_words(predictions, references)
        for model, predictions in predictions_by_model.items()
    }
    common = [
        index
        for index in range(len(references))
        if all(words[index].base_correct for words in scores.values())
    ]
    models = {}
    for model, words in scores.items():
        errors = sum(1 for index in common if words[index].diacritic_error)
        models[model] = {
            "base_correct_words": len(common),
            "diacritic_errors": errors,
            "der": _rate(errors, len(common)),
        }
    return {
        "words": len(references),
        "intersection_words": len(common),
        "excluded_words": len(references) - len(common),
        "models": models,
    }


def diacritic_error_share(errors: int, words: int, exact: int) -> float | None:
    # Input: K, N and E; Output: K / (N - E), the gate metric's share of wrong words
    # How: only counted when at least one word is wrong, so a perfect run has no share
    return _rate(errors, words - exact)
