#!/usr/bin/env python3
"""Rectify VinText word regions into native-resolution crops and write the PARSeq LMDB stores.

Two passes, both CPU only and deterministic:

1. geometry: read each image's header for its size, rectify the quadrilaterals of its usable
   regions, and collect the crop manifest plus per-split drop counts.
2. write: decode each image once, warp its crops, and stream them as PNG into
   ``data/lmdb/<split>`` with the agreed PARSeq keys.

Nothing is written outside the given output paths; crops and labels stay under ignored
directories. Prints aggregate counts only, never transcript text.

Examples:
    uv run python scripts/make_crops.py --from_folder data/raw/vintext/vietnamese --dry-run
    uv run python scripts/make_crops.py --from_folder data/raw/vintext/vietnamese
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import cv2
from PIL import Image as PILImage
from tqdm import tqdm

# Allow running this file directly, without installing the package (see pyproject: package = false)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dfstr.data.annotations import (  # noqa: E402
    AnnotationError,
    Region,
    is_usable_transcript,
    iter_regions,
)
from dfstr.data.crop_manifest import CropRecord, write_manifest  # noqa: E402
from dfstr.data.crops import DROP_REASONS, Rectified, RectifyError, rectify, warp  # noqa: E402
from dfstr.data.lmdb_store import (  # noqa: E402
    DEFAULT_MAP_SIZE,
    LmdbSample,
    encode_png,
    lmdb_count,
    write_lmdb,
)
from dfstr.data.splits import SPLIT_FOLDERS, SPLITS, read_manifest  # noqa: E402
from dfstr.data.training_labels import training_label  # noqa: E402

DEFAULT_SPLITS = Path("data/splits")
DEFAULT_MANIFEST = Path("data/interim/crops/manifest.csv")
DEFAULT_LMDB = Path("data/lmdb")


@dataclass(frozen=True)
class PlannedCrop:
    """A crop that survived rectification: its manifest row, geometry and label."""

    split: str
    record: CropRecord
    rectified: Rectified
    transcript: str


@dataclass
class SplitStats:
    images: int = 0
    regions: int = 0
    kept: int = 0
    vertical: int = 0
    clamped: int = 0
    errors: Counter[str] = field(default_factory=Counter)


def read_label_regions(labels_dir: Path, image_id: str, folder: str) -> Sequence[Region]:
    # Input: the labels directory, an image id and its release folder; Output: its regions
    path = labels_dir / f"gt_{int(image_id[2:])}.txt"
    text = path.read_text(encoding="utf-8")
    return list(iter_regions(text, image_id=image_id, folder=folder))


def image_size(path: Path) -> tuple[int, int]:
    with PILImage.open(path) as image:
        return image.size


def plan_split(
    split: str,
    image_ids: Iterable[str],
    *,
    dataset_root: Path,
    limit_images: int | None = None,
) -> tuple[list[PlannedCrop], SplitStats]:
    # Input: one split's image ids and the unpacked dataset root
    # Output: the crops that survive rectification, and per-split counts including every drop
    folder = SPLIT_FOLDERS[split]
    labels_dir = dataset_root / "labels"
    planned: list[PlannedCrop] = []
    stats = SplitStats()
    for image_id in list(image_ids)[: limit_images or None]:
        stats.images += 1
        image_path = dataset_root / folder / f"{image_id}.jpg"
        if not image_path.exists():
            stats.errors["missing_image"] += 1
            continue
        width, height = image_size(image_path)
        try:
            regions = read_label_regions(labels_dir, image_id, folder)
        except (OSError, UnicodeDecodeError, AnnotationError, ValueError):
            stats.errors["unreadable_label_file"] += 1
            continue
        stats.regions += len(regions)
        for region in regions:
            if not is_usable_transcript(region.transcript):
                stats.errors["unreadable"] += 1
                continue
            try:
                rectified = rectify(region.quad, image_width=width, image_height=height)
            except RectifyError as err:
                stats.errors[err.reason] += 1
                continue
            except ValueError:
                stats.errors["bad_quad"] += 1
                continue
            stats.kept += 1
            stats.vertical += rectified.orientation == "vertical"
            stats.clamped += rectified.clamped
            planned.append(
                PlannedCrop(
                    split=split,
                    transcript=region.transcript,
                    rectified=rectified,
                    record=CropRecord(
                        crop_id=region.crop_id,
                        image_id=region.image_id,
                        split=split,
                        region_index=region.region_index,
                        width=rectified.width,
                        height=rectified.height,
                        orientation=rectified.orientation,
                        clamped=rectified.clamped,
                    ),
                )
            )
    return planned, stats


def _group_by_image(planned: list[PlannedCrop]) -> Iterator[tuple[str, list[PlannedCrop]]]:
    grouped: dict[str, list[PlannedCrop]] = defaultdict(list)
    for crop in planned:
        grouped[crop.record.image_id].append(crop)
    for image_id in sorted(grouped):
        yield image_id, grouped[image_id]


def iter_samples(
    planned: list[PlannedCrop], *, split: str, dataset_root: Path
) -> Iterator[LmdbSample]:
    # Input: one split's planned crops; Output: LMDB samples in manifest order
    # How: decode each source image once, warp its crops, encode them losslessly
    folder = SPLIT_FOLDERS[split]
    for image_id, crops in _group_by_image(planned):
        image = cv2.imread(str(dataset_root / folder / f"{image_id}.jpg"), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"cannot decode {folder}/{image_id}.jpg")
        for crop in crops:
            yield LmdbSample(
                image_png=encode_png(warp(image, crop.rectified)),
                label=training_label(crop.transcript, split),
            )


def print_summary(per_split: dict[str, SplitStats], planned: dict[str, list[PlannedCrop]]) -> None:
    print(f"{'split':<12}{'images':>8}{'regions':>9}{'kept':>8}{'vertical':>10}{'clamped':>9}")
    for split in per_split:
        stats = per_split[split]
        print(
            f"{split:<12}{stats.images:>8}{stats.regions:>9}{stats.kept:>8}"
            f"{stats.vertical:>10}{stats.clamped:>9}"
        )
    print()
    for split, stats in per_split.items():
        counted = {"unreadable", *DROP_REASONS}
        reasons = {reason: stats.errors.get(reason, 0) for reason in sorted(counted)}
        other = {key: value for key, value in stats.errors.items() if key not in counted}
        print(f"{split} drops: {reasons}{f' other: {other}' if other else ''}")
        crops = [crop.record for crop in planned[split]]
        if crops:
            heights = [record.height for record in crops]
            widths = [record.width for record in crops]
            print(
                f"{split} crop size: height min/median/max "
                f"{min(heights)}/{int(statistics.median(heights))}/{max(heights)}, "
                f"width {min(widths)}/{int(statistics.median(widths))}/{max(widths)}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--from_folder", type=Path, required=True, help="unpacked vietnamese/ directory"
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=DEFAULT_SPLITS,
        help=f"split manifests (default: {DEFAULT_SPLITS})",
    )
    parser.add_argument(
        "--manifest", type=Path, default=DEFAULT_MANIFEST, help=f"default: {DEFAULT_MANIFEST}"
    )
    parser.add_argument("--lmdb", type=Path, default=DEFAULT_LMDB, help=f"default: {DEFAULT_LMDB}")
    parser.add_argument(
        "--split", action="append", choices=SPLITS, help="only this split (repeatable)"
    )
    parser.add_argument("--limit-images", type=int, help="only the first N images of each split")
    parser.add_argument("--map-size", type=int, default=DEFAULT_MAP_SIZE, help="LMDB map size")
    parser.add_argument("--force", action="store_true", help="replace a complete existing store")
    parser.add_argument("--dry-run", action="store_true", help="count only; write nothing")
    args = parser.parse_args(argv)

    splits = args.split or list(SPLITS)
    planned: dict[str, list[PlannedCrop]] = {}
    per_split: dict[str, SplitStats] = {}
    for split in splits:
        image_ids = read_manifest(args.splits_dir / f"{split}.txt")
        crops, stats = plan_split(
            split,
            image_ids,
            dataset_root=args.from_folder,
            limit_images=args.limit_images,
        )
        planned[split] = crops
        per_split[split] = stats

    print_summary(per_split, planned)
    if args.dry_run:
        print("\ndry run: nothing written")
        return 0

    records = [crop.record for split in splits for crop in planned[split]]
    rows = write_manifest(records, args.manifest)
    print(f"\nwrote {rows} manifest rows to {args.manifest}")

    for split in splits:
        crops = planned[split]
        target = args.lmdb / split
        if not crops:
            print(f"{split}: no crops, skipped")
            continue
        existing = lmdb_count(target)
        if existing == len(crops) and not args.force:
            print(f"{split}: {target} already complete ({existing} samples), skipped")
            continue
        written = write_lmdb(
            target,
            tqdm(
                iter_samples(crops, split=split, dataset_root=args.from_folder),
                total=len(crops),
                desc=f"{split} crops",
                unit="crop",
            ),
            map_size=args.map_size,
            overwrite=args.force,
        )
        print(f"{split}: wrote {written} samples to {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
