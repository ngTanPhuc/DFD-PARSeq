"""Split manifests for VinText.

The official release already ships the split assignment as three image folders, so the manifests
are a copy of that assignment, not a resampling: no hashing, no shuffle, no crop-level split.
Folders are mapped to split names here, once, and the mapping is recorded in ``provenance.json``.

Only image ids and counts are written; no label text leaves this module.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from dfstr.data.annotations import is_usable_transcript
from dfstr.data.audit import decode_labels
from dfstr.data.vintext_source import FolderContents

#: Release folder that plays each split role. ``test_image`` is the 300-image validation split.
SPLIT_FOLDERS: dict[str, str] = {
    "train": "train_images",
    "validation": "test_image",
    "test": "unseen_test_images",
}

SPLITS = tuple(SPLIT_FOLDERS)

RULE = (
    "official folder assignment (train=train_images, validation=test_image, "
    "test=unseen_test_images); no reshuffle, no random sampling, all crops of an image stay "
    "in that image's split"
)

DROP_RULE = "drop regions whose transcript is empty or consists only of '#' (unreadable markers)"


@dataclass(frozen=True)
class SplitCounts:
    folder: str
    images: int
    regions_usable: int
    regions_dropped: int

    def as_dict(self) -> dict[str, object]:
        return {
            "folder": self.folder,
            "images": self.images,
            "regions_usable": self.regions_usable,
            "regions_dropped": self.regions_dropped,
        }


def split_counts(contents: FolderContents) -> SplitCounts:
    regions, _, _ = decode_labels(contents)
    usable = sum(1 for region in regions if is_usable_transcript(region.transcript))
    return SplitCounts(
        folder=contents.folder,
        images=len(contents.image_ids),
        regions_usable=usable,
        regions_dropped=len(regions) - usable,
    )


def build_manifest(
    folders: Mapping[str, FolderContents],
    *,
    source_archive: str,
    created: str | None = None,
    audit_report: Path | None = None,
) -> dict[str, object]:
    # Input: the image folders of the release plus the archive path used to read them
    # Output: the manifest payload (ids per split and the provenance block)
    missing = [folder for folder in SPLIT_FOLDERS.values() if folder not in folders]
    if missing:
        raise KeyError(f"release is missing folders {missing}; found {sorted(folders)}")
    splits: dict[str, dict[str, object]] = {}
    ids_by_split: dict[str, list[str]] = {}
    for split, folder in SPLIT_FOLDERS.items():
        contents = folders[folder]
        counts = split_counts(contents)
        ids_by_split[split] = list(contents.image_ids)
        splits[split] = counts.as_dict()
    all_ids = [image_id for ids in ids_by_split.values() for image_id in ids]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("image ids appear in more than one split")
    payload: dict[str, object] = {
        "created": created or date.today().isoformat(),
        "source_archive": source_archive,
        "rule": RULE,
        "drop_rule": DROP_RULE,
        "splits": splits,
        "images": ids_by_split,
    }
    if audit_report is not None:
        payload["audit_report"] = str(audit_report)
        payload["audit_report_sha256"] = hashlib.sha256(audit_report.read_bytes()).hexdigest()
    return payload


def write_manifests(payload: Mapping[str, object], out_dir: Path) -> list[Path]:
    # Input: a manifest payload and the output directory; Output: the files written
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    images = payload["images"]
    for split in SPLITS:
        path = out_dir / f"{split}.txt"
        path.write_text("\n".join(images[split]) + "\n", encoding="utf-8")
        written.append(path)
    provenance = {key: value for key, value in payload.items() if key != "images"}
    path = out_dir / "provenance.json"
    path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    written.append(path)
    return written


def read_manifest(path: Path) -> tuple[str, ...]:
    # Input: a split manifest file; Output: its image ids in file order
    return tuple(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
