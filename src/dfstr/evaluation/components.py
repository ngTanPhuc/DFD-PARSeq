"""The word components that the metrics compare: base letter, vowel modifier and tone.

Metrics score decoded Unicode strings, never head outputs, so this module reads a word with
``unicodedata`` and borrows only the Vietnamese character tables from
:mod:`dfstr.labels.orthography`. The metrics therefore do not depend on the model-side tables or
on the training tokenizer: one decomposition bug cannot change both the targets and the score.

A word is split into NFD units. Each unit is one base code point plus the combining marks written
after it, which is what makes position-by-position comparison possible.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from dfstr.labels.orthography import MODIFIER_MARKS, NONE, STROKE_BASES, TONE_MARKS


@dataclass(frozen=True)
class Unit:
    """One base code point and the marks that belong to it."""

    base: str
    modifier: str
    tone: str
    other: tuple[str, ...] = ()


def base_letters(word: str) -> str:
    # Input: any word, NFC or NFD; Output: its base letters, ``đ`` folded to ``d``
    # How: NFD separates a letter from its marks, so dropping the marks leaves the base;
    #      ``đ`` is a single code point and needs the explicit map
    decomposed = unicodedata.normalize("NFD", word)
    stripped = "".join(char for char in decomposed if not unicodedata.combining(char))
    return "".join(STROKE_BASES.get(char, char) for char in stripped)


def units(word: str) -> tuple[Unit, ...]:
    # Input: any word, NFC or NFD; Output: its NFD units in order
    # How: group each base with the combining marks after it, then name the marks from the
    #      Vietnamese tables; an unknown mark stays in ``other`` so it cannot pass as normal
    groups: list[tuple[str, list[str]]] = []
    for char in unicodedata.normalize("NFD", word):
        if unicodedata.combining(char):
            if not groups:
                raise ValueError(f"word {word!r} starts with combining mark {char!r}")
            groups[-1][1].append(char)
        else:
            groups.append((char, []))
    decomposed: list[Unit] = []
    for base, marks in groups:
        tones = [mark for mark in marks if mark in TONE_MARKS]
        modifiers = [mark for mark in marks if mark in MODIFIER_MARKS]
        if len(tones) > 1:
            raise ValueError(f"unit {base!r} in {word!r} carries {len(tones)} tone marks")
        if len(modifiers) > 1:
            raise ValueError(f"unit {base!r} in {word!r} carries {len(modifiers)} modifier marks")
        decomposed.append(
            Unit(
                base=base,
                modifier=MODIFIER_MARKS[modifiers[0]] if modifiers else NONE,
                tone=TONE_MARKS[tones[0]] if tones else NONE,
                other=tuple(
                    mark for mark in marks if mark not in TONE_MARKS and mark not in MODIFIER_MARKS
                ),
            )
        )
    return tuple(decomposed)
