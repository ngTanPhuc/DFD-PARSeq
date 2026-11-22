"""Leakage guards over the tracked manifests.

Three things must hold before any freeze or test run: no image id in two splits, every crop in the
split of its own image, and every crop id traceable back to that image. Both inputs are the
tracked manifests (``data/splits/*.txt`` and the crop manifest), so these checks never need the
release, and they read no transcript.

Train-only fitting (rarity buckets, the E6 dictionary) is checked where that code is built, not
here: it does not exist yet (see ``docs/evaluation-protocol.md`` §5).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence

from dfstr.data.crop_manifest import CropRecord


def check_split_disjointness(image_ids_by_split: Mapping[str, Sequence[str]]) -> None:
    # Input: image ids per split; Output: None, or a ValueError naming every offending image id
    # How: keep the first split seen for each id; a second sighting in another split is a leak,
    #      a second sighting in the same split is a manifest bug
    first_split: dict[str, str] = {}
    crossing: list[str] = []
    repeated: list[str] = []
    for split, image_ids in image_ids_by_split.items():
        local = Counter(image_ids)
        repeated.extend(f"{split}/{image_id}" for image_id, count in local.items() if count > 1)
        for image_id in local:
            previous = first_split.setdefault(image_id, split)
            if previous != split:
                crossing.append(f"{image_id} ({previous} and {split})")
    problems: list[str] = []
    if crossing:
        problems.append("image ids in more than one split: " + ", ".join(sorted(set(crossing))))
    if repeated:
        problems.append("image ids repeated inside one split: " + ", ".join(sorted(repeated)))
    if problems:
        raise ValueError("; ".join(problems))


def check_crop_manifest(
    crops: Iterable[CropRecord], image_ids_by_split: Mapping[str, Sequence[str]]
) -> dict[str, object]:
    # Input: crop records and the split manifests; Output: crop counts per split
    # How: every crop id is unique and starts with its image id, and the crop's recorded split is
    #      the split its image belongs to; any mismatch is a leak between splits
    known = {image_id: split for split, ids in image_ids_by_split.items() for image_id in ids}
    counts: Counter[str] = Counter()
    seen_crop_ids: set[str] = set()
    problems: list[str] = []
    total = 0
    for crop in crops:
        total += 1
        counts[crop.split] += 1
        if crop.crop_id in seen_crop_ids:
            problems.append(f"crop id {crop.crop_id} appears twice")
        seen_crop_ids.add(crop.crop_id)
        if not crop.crop_id.startswith(f"{crop.image_id}_r"):
            problems.append(f"crop id {crop.crop_id} does not start with its image {crop.image_id}")
        image_split = known.get(crop.image_id)
        if image_split is None:
            problems.append(
                f"crop {crop.crop_id} belongs to image {crop.image_id}, not in any split"
            )
        elif image_split != crop.split:
            problems.append(
                f"crop {crop.crop_id} is recorded in {crop.split} but image {crop.image_id} "
                f"is in {image_split}"
            )
    if problems:
        shown = "; ".join(problems[:10])
        more = "" if len(problems) <= 10 else f" (+{len(problems) - 10} more)"
        raise ValueError(shown + more)
    return {
        "crops": total,
        "by_split": {split: counts.get(split, 0) for split in image_ids_by_split},
    }
