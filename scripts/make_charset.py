#!/usr/bin/env python3
"""Generate the label charset from the VinText release.

Writes one character per line, sorted by code point, with no comments and no blank lines
(``#`` and the space are both label characters, so neither can be reserved as syntax).

By default the charset is the **union of every folder**, because a charset must be able to
encode every label it will be scored on: ``\u1ebd`` occurs only in validation and ``\u1ef5``
only in test, so a train-only charset would make those words impossible to spell correctly.
The result is 228 characters, matching the baseline in the proposal. Character counts only
are printed; no transcript leaves this script.

Examples:
    uv run python scripts/make_charset.py --from_folder data/raw/vintext/vietnamese
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

# Allow running this file directly, without installing the package (see pyproject: package = false)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dfstr.data.audit import decode_labels, usable_words  # noqa: E402
from dfstr.data.charset import (  # noqa: E402
    character_inventory,
    charset_from_inventory,
    write_charset,
)
from dfstr.data.vintext_source import open_source  # noqa: E402

DEFAULT_OUT = Path("configs/charset/baseline.txt")

#: Release folder that plays each split role; local so this script needs no split module.
SPLIT_FOLDERS = {
    "train_images": "train",
    "test_image": "validation",
    "unseen_test_images": "test",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--from_zip", type=Path, help="VinText zip, read in place")
    source.add_argument("--from_folder", type=Path, help="unpacked vietnamese/ directory")
    parser.add_argument(
        "--min-occurrences",
        type=int,
        default=1,
        help="drop characters seen fewer than N times across all folders (default: 1)",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"default: {DEFAULT_OUT}")
    parser.add_argument("--dry-run", action="store_true", help="print counts, write nothing")
    args = parser.parse_args(argv)

    folders = open_source(root=args.from_folder, archive=args.from_zip).read_folders()
    missing = [folder for folder in SPLIT_FOLDERS if folder not in folders]
    if missing:
        raise KeyError(f"release is missing folders {missing}; found {sorted(folders)}")

    union: Counter[str] = Counter()
    print(f"{'split':<12} {'folder':<20} {'words':>8} {'distinct':>9}")
    for folder, split in SPLIT_FOLDERS.items():
        regions, _, _ = decode_labels(folders[folder])
        words = usable_words(regions)
        counts = character_inventory(words)
        union.update(counts)
        print(f"{split:<12} {folder:<20} {len(words):>8} {len(counts):>9}")

    charset = charset_from_inventory(union, min_occurrences=args.min_occurrences)
    dropped = sorted(char for char in union if char not in charset)
    print(f"\nunion distinct characters: {len(union)}")
    print(f"charset size (min occurrences {args.min_occurrences}): {len(charset)}")
    if dropped:
        print(f"dropped as too rare ({len(dropped)}): {[repr(char) for char in dropped]}")

    if args.dry_run:
        print("\ndry run: nothing written")
        return 0
    write_charset(args.out, charset)
    print(f"wrote {args.out} ({len(charset)} characters)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
