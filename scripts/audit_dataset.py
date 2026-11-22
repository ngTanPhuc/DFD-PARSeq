#!/usr/bin/env python3
"""Audit the VinText release and print aggregate statistics only.

Reads either the release zip in place or an unpacked ``vietnamese/`` directory, never writes
images, and never prints label text: the report contains counts, rates and distributions.

Examples:
    uv run python scripts/audit_dataset.py --from_zip data/raw/vintext_original.zip
    uv run python scripts/audit_dataset.py --from_folder data/raw/vintext/vietnamese \\
        --out data/interim/audit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running this file directly, without installing the package (see pyproject: package = false)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dfstr.data.audit import DEFAULT_STATS_FOLDER, build_report  # noqa: E402
from dfstr.data.charset import read_charset  # noqa: E402
from dfstr.data.vintext_source import open_source  # noqa: E402


def print_summary(report: dict) -> None:
    dataset = report["dataset"]
    print(
        f"dataset: {dataset['images']} images, {dataset['label_files']} label files, "
        f"{dataset['regions']} regions, {dataset['unreadable_regions']} unreadable"
    )
    print()
    header = (
        f"{'folder':<20} {'images':>7} {'regions':>8} {'unreadable':>7} {'empty':>6} "
        f"{'nfc-fix':>8} {'nfd':>6} {'mixed':>6} {'per-image':>14}"
    )
    print(header)
    for name, folder in report["folders"].items():
        norm = folder["normalization"]
        per_image = folder["regions_per_image"]
        spread = f"{per_image['min']}-{per_image['max']} ({per_image['mean']})"
        print(
            f"{name:<20} {folder['images']:>7} {folder['regions']:>8} "
            f"{folder['unreadable_regions']:>7} {folder['empty_transcripts']:>6} "
            f"{folder['labels_changed_by_nfc']:>8} "
            f"{norm.get('nfd', 0):>6} {norm.get('mixed', 0):>6} {spread:>14}"
        )
    hash_only: dict[str, int] = {}
    for folder in report["folders"].values():
        for length, count in folder["hash_only_by_length"].items():
            hash_only[length] = hash_only.get(length, 0) + count
    if hash_only:
        print(f"hash-only transcripts by length: {dict(sorted(hash_only.items()))}")
    if report["errors"]:
        print(f"\nerrors: {report['errors']}")
    stats = report["label_stats"]
    print(f"\nlabel statistics from folder {stats['folder']!r} (readable words only):")
    print(
        f"  words {stats['words']}, characters {stats['characters']}, "
        f"distinct base words {stats['distinct_base_words']}"
    )
    print(f"  case: {stats['case']}")
    print(f"  length histogram: {stats['length_histogram']}")
    print(f"  tone: {stats['tone']}")
    print(f"  modifier: {stats['modifier']}")
    print(
        f"  no-tone positions: {stats['no_tone_rate']}, "
        f"no-modifier positions: {stats['no_modifier_rate']}"
    )
    print(f"  tone placement: {stats['tone_placement']}")
    if stats.get("charset"):
        coverage = stats["charset"]
        print(f"  charset size: {coverage['charset_size']}")
        print(f"  out-of-charset occurrences: {coverage['out_of_charset_occurrences']}")
        if coverage["out_of_charset_chars"]:
            print(f"  out-of-charset characters: {coverage['out_of_charset_chars']}")
        if coverage["unused_charset_entries"]:
            print(f"  charset entries never used here: {coverage['unused_charset_entries']}")
    for name, folder in report["folders"].items():
        if "charset" in folder:
            coverage = folder["charset"]
            print(f"  {name}: {coverage['out_of_charset_occurrences']} out-of-charset occurrences")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--from_zip", type=Path, help="VinText zip, read in place")
    source.add_argument("--from_folder", type=Path, help="unpacked vietnamese/ directory")
    parser.add_argument(
        "--stats-folder",
        default=DEFAULT_STATS_FOLDER,
        help=f"folder used for label statistics (default: {DEFAULT_STATS_FOLDER})",
    )
    parser.add_argument("--charset", type=Path, help="baseline charset, one character per line")
    parser.add_argument("--out", type=Path, help="directory for report.json")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args(argv)

    folders = open_source(root=args.from_folder, archive=args.from_zip).read_folders()
    charset = read_charset(args.charset) if args.charset else None
    report = build_report(folders, stats_folder=args.stats_folder, charset=charset)

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        print_summary(report)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        target = args.out / "report.json"
        target.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
