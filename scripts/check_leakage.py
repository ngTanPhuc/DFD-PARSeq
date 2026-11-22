#!/usr/bin/env python3
"""Check the tracked manifests for leakage between splits.

Reads ``data/splits/*.txt`` and the crop manifest only, so it needs no release and reads no
transcript. Prints aggregate counts; run it before every freeze and before the final test run.

Examples:
    uv run python scripts/check_leakage.py --crops data/interim/crops/manifest.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running this file directly, without installing the package (see pyproject: package = false)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dfstr.data.crop_manifest import read_manifest as read_crops  # noqa: E402
from dfstr.data.leakage import check_crop_manifest, check_split_disjointness  # noqa: E402
from dfstr.data.splits import SPLITS  # noqa: E402
from dfstr.data.splits import read_manifest as read_split  # noqa: E402

DEFAULT_SPLITS = Path("data/splits")
DEFAULT_CROPS = Path("data/interim/crops/manifest.csv")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--splits", type=Path, default=DEFAULT_SPLITS, help=f"default: {DEFAULT_SPLITS}"
    )
    parser.add_argument(
        "--crops", type=Path, default=DEFAULT_CROPS, help=f"default: {DEFAULT_CROPS}"
    )
    args = parser.parse_args(argv)

    image_ids_by_split = {split: read_split(args.splits / f"{split}.txt") for split in SPLITS}
    check_split_disjointness(image_ids_by_split)
    counts = check_crop_manifest(read_crops(args.crops), image_ids_by_split)

    print(f"{'split':<12} {'images':>7} {'crops':>7}")
    for split in SPLITS:
        print(f"{split:<12} {len(image_ids_by_split[split]):>7} {counts['by_split'][split]:>7}")
    total_images = sum(len(ids) for ids in image_ids_by_split.values())
    print(f"{'total':<12} {total_images:>7} {counts['crops']:>7}")
    print("no leakage between splits")
    return 0


if __name__ == "__main__":
    sys.exit(main())
