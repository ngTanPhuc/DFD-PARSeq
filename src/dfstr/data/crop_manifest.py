"""The crop manifest: one row per written crop, keyed by ``crop_id``.

The manifest is the join table between the release (image id, region index) and the LMDB (image
key, label). It carries geometry only: crop id, owning image, split, source region index, native
crop size, orientation and whether the quadrilateral had to be clipped to the image. No transcript
text is stored here, so the file stays free of ground truth even though it lives under ``data/``.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

MANIFEST_COLUMNS: tuple[str, ...] = (
    "crop_id",
    "image_id",
    "split",
    "region_index",
    "width",
    "height",
    "orientation",
    "clamped",
)


@dataclass(frozen=True)
class CropRecord:
    crop_id: str
    image_id: str
    split: str
    region_index: int
    width: int
    height: int
    orientation: str
    clamped: bool = False

    def as_row(self) -> dict[str, object]:
        return {
            "crop_id": self.crop_id,
            "image_id": self.image_id,
            "split": self.split,
            "region_index": self.region_index,
            "width": self.width,
            "height": self.height,
            "orientation": self.orientation,
            "clamped": int(self.clamped),
        }


def sort_records(records: Iterable[CropRecord], split_order: Sequence[str]) -> list[CropRecord]:
    # Input: records in any order and the canonical split order
    # Output: records sorted by split, then image id, then region index (deterministic manifest)
    rank = {split: index for index, split in enumerate(split_order)}
    return sorted(records, key=lambda r: (rank.get(r.split, len(rank)), r.image_id, r.region_index))


def write_manifest(records: Iterable[CropRecord], path: Path) -> int:
    # Input: crop records and the manifest path; Output: the number of rows written
    rows = sort_records(records, sorted({record.split for record in records}))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for record in rows:
            writer.writerow(record.as_row())
    return len(rows)


def read_manifest(path: Path) -> list[CropRecord]:
    # Input: a manifest written by write_manifest; Output: its records in file order
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(MANIFEST_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: manifest is missing columns {sorted(missing)}")
        return [
            CropRecord(
                crop_id=row["crop_id"],
                image_id=row["image_id"],
                split=row["split"],
                region_index=int(row["region_index"]),
                width=int(row["width"]),
                height=int(row["height"]),
                orientation=row["orientation"],
                clamped=row["clamped"] == "1",
            )
            for row in reader
        ]
