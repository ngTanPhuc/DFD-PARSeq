"""Unicode-level label statistics for the VinText audit.

This module is audit-only: it measures the labels, it never produces training targets. The
model-side decomposition lives in :mod:`dfstr.labels` (role C); a cross-check test should be
added once that module exists, so the audit and the training path cannot drift apart.

All helpers work on NFD internally, so they are independent of the input normalization.
"""

from __future__ import annotations

import unicodedata
from collections import Counter

from dfstr.labels.orthography import (
    MODIFIER_MARKS,
    NONE,
    STROKE_BASES,
    TONE_MARKS,
    TONE_PLACEMENT_PAIRS,
)


def decompose_char(char: str) -> tuple[str, str, str]:
    """
    Decomposes one Unicode character into its base letter, modifier, and tone (chữ gốc, dấu phụ, dấu thanh).
    Input: one Unicode character
    Output: (base letter, modifier name, tone name)
    How: NFD-split the character, then classify every combining mark; "none" when absent
    """

    if char in STROKE_BASES:  # đ and Đ
        return STROKE_BASES[char], "stroke", NONE
    
    decomposed = unicodedata.normalize("NFD", char)
    base = decomposed[0]
    modifier = NONE
    tone = NONE

    for mark in decomposed[1:]:
        if mark in MODIFIER_MARKS:
            modifier = MODIFIER_MARKS[mark]
        elif mark in TONE_MARKS:
            tone = TONE_MARKS[mark]

    return base, modifier, tone


def base_of(text: str) -> str:
    """
    Decomposes the input text into its base letters, dropping all combining marks and converting đ to d.
    Input: any label text
    Output: the frozen "base letters" form (NFD, marks dropped, đ→d)
    """

    decomposed = unicodedata.normalize("NFD", text)
    letters = [STROKE_BASES.get(char, char) for char in decomposed if not unicodedata.combining(char)]

    return "".join(letters)


def tone_placement(text: str) -> str | None:
    """
    Determines the tone placement for a given Vietnamese text (hòa (old) or hoà (new)).
    Input: one label
    Output: "old" for hòa/khỏe/thủy, "new" for hoà/khoẻ/thuỷ, else None
    How: find the first oa/oe/uy vowel pair and report which of the two carries the tone; "qu" is an onset, so a ``u`` straight after ``q`` is skipped
    """
    units: list[tuple[str, set[str]]] = []

    for char in unicodedata.normalize("NFD", text):
        if unicodedata.combining(char):
            if units:
                units[-1][1].add(char)
        else:
            units.append((char, set()))

    for index in range(len(units) - 1):
        first_base, first_marks = units[index]
        second_base, second_marks = units[index + 1]
        if first_base.lower() + second_base.lower() not in TONE_PLACEMENT_PAIRS:
            continue

        if first_base.lower() == "u" and index > 0 and units[index - 1][0].lower() == "q":
            continue

        first_toned = any(mark in TONE_MARKS for mark in first_marks)
        second_toned = any(mark in TONE_MARKS for mark in second_marks)

        if first_toned and not second_toned:
            return "old"
        if second_toned and not first_toned:
            return "new"
        
    return None


def case_class(text: str) -> str:
    """
    Returns the case class of a given text: lower, upper, title, mixed, or uncased.
    - hoa: lower
    - HOA: upper
    - Hoa: title
    - hOa: mixed
    - 123: uncased
    - 12H: upper
    Input: one label
    Output: lower | upper | title | mixed | uncased
    """
    cased = [char for char in text if char.isalpha()]
    if not cased: return "uncased"
    if all(char.islower() for char in cased): return "lower"
    if all(char.isupper() for char in cased): return "upper"
    if cased[0].isupper() and all(char.islower() for char in cased[1:]): return "title"

    return "mixed"


def char_classes(text: str) -> Counter[str]:
    """
    Categorizes the characters in a given text by their Unicode general category prefix (L, N, P, ...).
    - L: Letter
    - N: Number
    - P: Punctuation
    - S: Symbol
    - Z: Separator
    - M: Mark
    - C: Other
    Input: one label
    Output: counts per Unicode general category prefix (L, N, P, ...)
    """
    counts: Counter[str] = Counter()

    for char in text:
        category = unicodedata.category(char)
        if category.startswith("M"):
            continue  # combining marks are already counted through decompose_char

        counts[category[0]] += 1

    return counts
