"""Vietnamese orthography rules shared by the data pipeline and the metrics.

Two tone-placement conventions exist for the vowel pairs ``oa``, ``oe`` and ``uy``: the tone
sits on the first vowel (``hòa``, ``khỏe``, ``thủy``) or on the second (``hoà``, ``khoẻ``,
``thuỷ``). The audit found both in VinText, so one convention is chosen for training labels and
the metrics normalize the other before scoring. See ``docs/data-audit.md`` §5.

This module holds the character tables so the audit, the data pipeline and the metrics share one
definition instead of three copies that can drift apart.
"""

from __future__ import annotations

import unicodedata

#: Vietnamese tone marks as NFD combining characters.
TONE_MARKS = {
    "\u0301": "sac",
    "\u0300": "huyen",
    "\u0309": "hoi",
    "\u0303": "nga",
    "\u0323": "nang",
}

#: Vietnamese vowel modifiers as NFD combining characters.
MODIFIER_MARKS = {
    "\u0306": "breve",
    "\u0302": "circumflex",
    "\u031b": "horn",
}

#: ``đ`` is a precomposed letter, so it needs an explicit base mapping.
STROKE_BASES = {"\u0111": "d", "\u0110": "D"}

#: Vowel pairs whose tone position distinguishes the two conventions.
TONE_PLACEMENT_PAIRS = frozenset({"oa", "oe", "uy"})

#: Name of the "no modifier" and "no tone" class in every report and head output.
NONE = "none"


def tone_normalize(text: str) -> str:
    # Input: any word, NFC or NFD; Output: NFC word with the tone on the second vowel of oa/oe/uy
    # How: walk the NFD units, move a tone mark one vowel right when the pair is oa/oe/uy, then
    #      re-compose. Uppercase is preserved, "qu" is an onset so its u is never moved to.
    units: list[list] = []
    for char in unicodedata.normalize("NFD", text):
        if unicodedata.combining(char):
            if units:
                units[-1][1].append(char)
        else:
            units.append([char, []])
    for index in range(len(units) - 1):
        first, second = units[index], units[index + 1]
        if first[0].lower() + second[0].lower() not in TONE_PLACEMENT_PAIRS:
            continue
        if first[0].lower() == "u" and index > 0 and units[index - 1][0].lower() == "q":
            continue
        moved = [mark for mark in first[1] if mark in TONE_MARKS]
        if not moved or any(mark in TONE_MARKS for mark in second[1]):
            continue
        first[1] = [mark for mark in first[1] if mark not in TONE_MARKS]
        second[1] = second[1] + moved
    return unicodedata.normalize("NFC", "".join(base + "".join(marks) for base, marks in units))
