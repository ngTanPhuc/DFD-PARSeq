"""Exclusive error categories for base-correct words whose diacritics are wrong.

Only words in the DER numerator (base letters right, diacritics wrong) get a category, so the
category counts always sum to ``K``. Categories are computed on tone-normalized strings, so tone
*placement* is never a category.

Approved rule (22 Nov 2026, see ``docs/evaluation-protocol.md``):

1. Exactly one position differs and exactly one component differs there: that component's category.
2. Anything else: ``combined``.

``d``/``đ`` confusion is its own category because ``đ`` is a base letter that the base-letter
definition folds away, so it would otherwise hide among the modifier errors.
"""

from __future__ import annotations

from dfstr.evaluation.components import base_letters, units
from dfstr.labels.orthography import NONE

TONE_MISSING = "tone_missing"
TONE_WRONG = "tone_wrong"
TONE_ADDED = "tone_added"
MODIFIER_WRONG = "modifier_wrong"
D_STROKE = "d_stroke"
COMBINED = "combined"

#: Every category a word can take, in report order.
CATEGORIES: tuple[str, ...] = (
    TONE_MISSING,
    TONE_WRONG,
    TONE_ADDED,
    MODIFIER_WRONG,
    D_STROKE,
    COMBINED,
)


def categorize(prediction: str, reference: str) -> str:
    # Input: a tone-normalized prediction and reference with equal base letters, not identical
    # Output: the single category of this diacritic error
    # How: walk the NFD units in step (base-correct means the same number of units) and collect
    #      one name per differing component; a lone difference names the category, else combined
    predicted_units = units(prediction)
    reference_units = units(reference)
    if len(predicted_units) != len(reference_units):
        raise ValueError(
            f"not base-correct: {prediction!r} has {len(predicted_units)} units, "
            f"{reference!r} has {len(reference_units)}"
        )
    differences: list[str] = []
    for index, (predicted, expected) in enumerate(
        zip(predicted_units, reference_units, strict=True)
    ):
        if predicted.base != expected.base:
            if base_letters(predicted.base) != base_letters(expected.base):
                raise ValueError(
                    f"not base-correct at unit {index}: {predicted.base!r} vs {expected.base!r}"
                )
            differences.append(D_STROKE)
        if predicted.modifier != expected.modifier:
            differences.append(MODIFIER_WRONG)
        if predicted.tone != expected.tone:
            if expected.tone == NONE:
                differences.append(TONE_ADDED)
            elif predicted.tone == NONE:
                differences.append(TONE_MISSING)
            else:
                differences.append(TONE_WRONG)
        if predicted.other != expected.other:
            differences.append(COMBINED)
    if not differences:
        raise ValueError(f"{prediction!r} and {reference!r} are identical after normalization")
    if len(differences) == 1:
        return differences[0]
    return COMBINED
