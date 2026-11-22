"""Aggregate statistics over the VinText release, safe to print and to commit.

Every number this module produces is a count, a rate or a distribution. Label text itself is
never copied into the report, so the output can go into ``docs/`` and into the chat. Decisions
that depend on label statistics (charset, tone convention, rarity buckets) must be taken from
the folder passed as ``stats_folder`` — by default ``train_images`` — never from a test folder.

One exception, approved on 22 Nov 2026: when a charset is supplied, its *coverage counts* are
reported per folder, including the test folder. It is a count of code points with no word
content, it exists to prove that a frozen charset can encode every label it will be scored on,
and its only permitted use is to widen the charset — never to tune a model or a threshold.
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping

from dfstr.data.annotations import (
    AnnotationError,
    Region,
    image_id_from_label_name,
    is_usable_transcript,
    iter_regions,
)
from dfstr.data.charset import charset_coverage, character_inventory
from dfstr.data.unicode_stats import (
    NONE,
    base_of,
    case_class,
    char_classes,
    decompose_char,
    tone_placement,
)
from dfstr.data.vintext_source import FolderContents

DEFAULT_STATS_FOLDER = "train_images"


def is_readable_text(transcript: str) -> bool:
    # Kept as the audit-facing name; the rule itself lives in annotations
    return is_usable_transcript(transcript)


def normalize_class(text: str) -> str:
    # Input: one label; Output: plain | nfc | nfd | mixed, judged by re-normalizing the text
    nfc = unicodedata.normalize("NFC", text)
    nfd = unicodedata.normalize("NFD", text)
    if nfc == nfd:
        return "plain"
    if text == nfc:
        return "nfc"
    if text == nfd:
        return "nfd"
    return "mixed"


def decode_labels(
    contents: FolderContents,
) -> tuple[list[Region], dict[str, int], list[str]]:
    # Input: one folder of the release; Output: its regions, error counts, and error messages
    # How: decode every label file as UTF-8, then parse lines; failures are counted, not raised
    regions: list[Region] = []
    errors: Counter[str] = Counter()
    messages: list[str] = []
    for file_name, raw in contents.label_files:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            errors["decode"] += 1
            messages.append(f"{contents.folder}/{file_name}: not valid UTF-8")
            continue
        image_id = image_id_from_label_name(file_name)
        if image_id is None:
            errors["name"] += 1
            messages.append(f"{contents.folder}/{file_name}: unexpected label file name")
            continue
        try:
            regions.extend(iter_regions(text, image_id=image_id, folder=contents.folder))
        except AnnotationError as err:
            errors["annotation"] += 1
            messages.append(str(err))
    return regions, dict(errors), messages


def _region_histogram(regions: Iterable[Region]) -> dict[str, int | float]:
    per_image = Counter(region.image_id for region in regions)
    if not per_image:
        return {"min": 0, "max": 0, "mean": 0.0}
    counts = list(per_image.values())
    return {
        "min": min(counts),
        "max": max(counts),
        "mean": round(sum(counts) / len(counts), 2),
    }


def folder_report(contents: FolderContents, regions: list[Region]) -> dict[str, object]:
    images = set(contents.image_ids)
    with_regions = {region.image_id for region in regions}
    readable = [region.transcript for region in regions if is_readable_text(region.transcript)]
    normalization = Counter(normalize_class(text) for text in readable)
    changed = sum(1 for text in readable if text != unicodedata.normalize("NFC", text))
    distinct = len(character_inventory(unicodedata.normalize("NFC", text) for text in readable))
    hash_only: Counter[int] = Counter()
    empty = 0
    for region in regions:
        transcript = region.transcript
        if not transcript:
            empty += 1
        elif set(transcript) == {"#"}:
            hash_only[len(transcript)] += 1
    return {
        "images": len(images),
        "label_files": len(contents.label_files),
        "regions": len(regions),
        "regions_per_image": _region_histogram(regions),
        "unreadable_regions": sum(1 for region in regions if region.unreadable),
        "empty_transcripts": empty,
        "hash_only_by_length": {str(k): v for k, v in sorted(hash_only.items())},
        "images_without_regions": len(images - with_regions),
        "labels_without_image": len(with_regions - images),
        "normalization": dict(sorted(normalization.items())),
        "labels_changed_by_nfc": changed,
        "distinct_characters": distinct,
    }


def usable_words(regions: Iterable[Region]) -> list[str]:
    # Input: the regions of one folder; Output: the transcripts that survive the drop rule, in NFC
    return [
        unicodedata.normalize("NFC", region.transcript)
        for region in regions
        if is_readable_text(region.transcript)
    ]


def label_stats(
    regions: Iterable[Region], charset: frozenset[str] | None = None
) -> dict[str, object]:
    # Input: the regions of the statistics folder, and an optional baseline charset
    # Output: train-derived distributions of length, case, tone, modifier and charset coverage
    words = usable_words(regions)
    lengths: Counter[int] = Counter()
    cases: Counter[str] = Counter()
    placement: Counter[str] = Counter()
    tones: Counter[str] = Counter()
    modifiers: Counter[str] = Counter()
    combinations: Counter[str] = Counter()
    classes: Counter[str] = Counter()
    chars = 0
    for word in words:
        lengths[len(word)] += 1
        cases[case_class(word)] += 1
        placement[tone_placement(word) or "no_pattern"] += 1
        classes.update(char_classes(word))
        chars += len(word)
        for char in word:
            _, modifier, tone = decompose_char(char)
            tones[tone] += 1
            modifiers[modifier] += 1
            combinations[f"{modifier}|{tone}"] += 1
    stats: dict[str, object] = {
        "words": len(words),
        "characters": chars,
        "length_histogram": dict(sorted(lengths.items())),
        "case": dict(sorted(cases.items())),
        "tone_placement": dict(sorted(placement.items())),
        "tone": dict(sorted(tones.items())),
        "modifier": dict(sorted(modifiers.items())),
        "modifier_tone": dict(sorted(combinations.items())),
        "unicode_categories": dict(sorted(classes.items())),
        "distinct_base_words": len({base_of(word) for word in words}),
        "distinct_characters": len(character_inventory(words)),
        "no_tone_rate": round(tones.get(NONE, 0) / chars, 4) if chars else 0.0,
        "no_modifier_rate": round(modifiers.get(NONE, 0) / chars, 4) if chars else 0.0,
    }
    if charset is not None:
        stats["charset"] = charset_coverage(charset, character_inventory(words))
    return stats


def build_report(
    folders: Mapping[str, FolderContents],
    *,
    stats_folder: str = DEFAULT_STATS_FOLDER,
    charset: frozenset[str] | None = None,
) -> dict[str, object]:
    # Input: the image folders of a VinText release; Output: the full JSON-serializable report
    if stats_folder not in folders:
        raise KeyError(f"stats folder {stats_folder!r} not in {sorted(folders)}")
    per_folder: dict[str, object] = {}
    messages: list[str] = []
    errors: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    stats_regions: list[Region] = []
    for name, contents in sorted(folders.items()):
        regions, folder_errors, folder_messages = decode_labels(contents)
        errors.update(folder_errors)
        messages.extend(folder_messages)
        folder_stats = folder_report(contents, regions)
        if charset is not None:
            folder_stats["charset"] = charset_coverage(charset, character_inventory(usable_words(regions)))
        per_folder[name] = folder_stats
        totals["images"] += len(contents.image_ids)
        totals["label_files"] += len(contents.label_files)
        totals["regions"] += len(regions)
        totals["unreadable_regions"] += sum(1 for r in regions if r.unreadable)
        totals["empty_transcripts"] += sum(1 for r in regions if not r.transcript)
        if name == stats_folder:
            stats_regions = regions
    return {
        "dataset": dict(sorted(totals.items())),
        "folders": per_folder,
        "errors": dict(sorted(errors.items())),
        "error_messages": messages[:20],
        "label_stats": {"folder": stats_folder, **label_stats(stats_regions, charset=charset)},
    }
