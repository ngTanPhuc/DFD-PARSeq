"""Tests for the label charset (synthetic fixtures and the committed charset file)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from dfstr.data.charset import (
    character_inventory,
    charset_coverage,
    charset_from_inventory,
    read_charset,
    write_charset,
)

CHARSET_FILE = Path(__file__).resolve().parents[1] / "configs" / "charset" / "baseline.txt"


def test_write_then_read_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "charset.txt"
    write_charset(path, {"b", "a", "d"})
    assert path.read_text(encoding="utf-8") == "a\nb\nd\n"
    assert read_charset(path) == frozenset({"a", "b", "d"})


def test_sorted_output_is_deterministic(tmp_path: Path) -> None:
    first, second = tmp_path / "a.txt", tmp_path / "b.txt"
    write_charset(first, {"d", "a", "b"})
    write_charset(second, {"b", "d", "a"})
    assert first.read_bytes() == second.read_bytes()


def test_space_survives_because_lines_are_not_stripped(tmp_path: Path) -> None:
    path = tmp_path / "charset.txt"
    write_charset(path, {" ", "a"})
    assert read_charset(path) == frozenset({" ", "a"})


def test_hash_is_a_character_not_a_comment(tmp_path: Path) -> None:
    path = tmp_path / "charset.txt"
    write_charset(path, {"#", "a"})
    assert read_charset(path) == frozenset({"#", "a"})


def test_decomposed_entry_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "charset.txt"
    path.write_text("e\u0301\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not NFC"):
        read_charset(path)


def test_multi_character_entry_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "charset.txt"
    path.write_text("ab\n", encoding="utf-8")
    with pytest.raises(ValueError, match="code points"):
        read_charset(path)


def test_blank_line_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "charset.txt"
    path.write_text("a\n\nb\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty entry"):
        read_charset(path)


def test_duplicate_entry_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "charset.txt"
    path.write_text("a\na\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        read_charset(path)


def test_inventory_counts_nfc_characters() -> None:
    # "hòa" is h, ò, a: the tone is a precomposed code point, not a combining mark
    counts = character_inventory(["h\u00f2a", "ho\u00e0"])
    assert counts["h"] == 2
    assert counts["a"] == 1
    assert counts["o"] == 1
    assert counts["\u00f2"] == 1
    assert counts["\u00e0"] == 1
    assert sum(counts.values()) == 6


def test_inventory_normalizes_decomposed_input() -> None:
    counts = character_inventory(["ho\u0300a"])
    assert dict(counts) == {"h": 1, "\u00f2": 1, "a": 1}


def test_coverage_reports_missing_and_unused() -> None:
    coverage = charset_coverage(frozenset({"a", "z"}), Counter({"a": 3, "b": 2}))
    assert coverage["charset_size"] == 2
    assert coverage["out_of_charset_occurrences"] == 2
    assert coverage["out_of_charset_chars"] == {"b": 2}
    assert coverage["unused_charset_entries"] == ["z"]


def test_coverage_of_a_sufficient_charset_is_empty() -> None:
    coverage = charset_coverage(frozenset({"a", "b"}), Counter({"a": 1, "b": 1}))
    assert coverage["out_of_charset_occurrences"] == 0
    assert coverage["out_of_charset_chars"] == {}
    assert coverage["unused_charset_entries"] == []


def test_min_occurrences_drops_rare_characters() -> None:
    counts = Counter({"a": 5, "b": 1})
    assert charset_from_inventory(counts) == frozenset({"a", "b"})
    assert charset_from_inventory(counts, min_occurrences=2) == frozenset({"a"})


@pytest.mark.skipif(not CHARSET_FILE.is_file(), reason="charset file not generated yet")
def test_committed_charset_is_the_expected_size_and_covers_space_and_hash() -> None:
    # Guards the committed file against an editor trimming the space line or a bad edit
    charset = read_charset(CHARSET_FILE)
    assert len(charset) == 228
    assert " " in charset
    assert "#" in charset
    assert all(len(char) == 1 for char in charset)
