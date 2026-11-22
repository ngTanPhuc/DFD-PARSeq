"""The label charset: the set of characters a VinText transcript may contain.

The file format is one character per line, in NFC. It carries **no comments and no blank
lines**, because ``#`` is itself a label character (13 occurrences in the usable train labels)
and a space is a label character too (465 occurrences), so neither can be reserved as syntax.
A space is written as a line holding a single space; the reader never strips a line, so that
entry survives.

Only the set of characters is stored. Nothing here looks at word content, so a generated
charset file contains no ground-truth transcript.
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from collections.abc import Iterable
from pathlib import Path


def _check_char(char: str, *, source: str) -> str:
    # Input: one entry as written; Output: the character, or a ValueError with the reason
    # How: an entry must be one code point already in NFC, so the file cannot drift from labels
    if char == "":
        raise ValueError(f"{source}: empty entry")
    if unicodedata.normalize("NFC", char) != char:
        raise ValueError(f"{source}: entry {char!r} is not NFC")
    if len(char) != 1:
        raise ValueError(f"{source}: entry {char!r} is {len(char)} code points, expected 1")
    return char


def read_charset(path: Path) -> frozenset[str]:
    # Input: a charset file, one character per line; Output: its characters as a set
    chars: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        char = _check_char(line, source=f"{path}:{line_number}")
        if char in chars:
            raise ValueError(f"{path}:{line_number}: duplicate entry {char!r}")
        chars.add(char)
    return frozenset(chars)


def write_charset(path: Path, chars: Iterable[str]) -> None:
    # Input: an output path and the characters to store; Output: the file, sorted by code point
    # How: write one character per line in sorted order, then read it back to prove the round-trip
    ordered = sorted(_check_char(char, source="write_charset") for char in chars)
    if len(ordered) != len(set(ordered)):
        raise ValueError("write_charset: duplicate characters")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ordered) + "\n", encoding="utf-8")
    if read_charset(path) != set(ordered):
        raise ValueError(f"{path}: written file does not read back as the same charset")


def character_inventory(words: Iterable[str]) -> Counter[str]:
    # Input: label words; Output: how often each NFC character occurs
    counts: Counter[str] = Counter()
    for word in words:
        counts.update(unicodedata.normalize("NFC", word))
    return counts


def charset_coverage(charset: frozenset[str], counts: Counter[str]) -> dict[str, object]:
    # Input: a charset and a character inventory; Output: what the charset cannot encode
    # How: any character outside the set is reported with its occurrence count; entries never
    #      used are listed so an unreachable class in the output heads is visible
    out_of_charset = {char: count for char, count in counts.items() if char not in charset}
    return {
        "charset_size": len(charset),
        "out_of_charset_occurrences": sum(out_of_charset.values()),
        "out_of_charset_chars": dict(sorted(out_of_charset.items())),
        "unused_charset_entries": sorted(charset - set(counts)),
    }


def charset_from_inventory(counts: Counter[str], *, min_occurrences: int = 1) -> frozenset[str]:
    # Input: a character inventory and a minimum count; Output: the characters at or above it
    return frozenset(char for char, count in counts.items() if count >= min_occurrences)
