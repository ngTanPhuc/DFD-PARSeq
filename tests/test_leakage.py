"""Tests for the leakage guards over the split and crop manifests (synthetic ids only)."""

from __future__ import annotations

import pytest

from dfstr.data.crop_manifest import CropRecord
from dfstr.data.leakage import check_crop_manifest, check_split_disjointness


def _crop(crop_id: str, image_id: str, split: str, region_index: int = 0) -> CropRecord:
    return CropRecord(
        crop_id=crop_id,
        image_id=image_id,
        split=split,
        region_index=region_index,
        width=40,
        height=20,
        orientation="horizontal",
    )


SYNTHETIC_SPLITS = {
    "train": ("im0001", "im0002"),
    "validation": ("im0003",),
    "test": ("im0004",),
}


def test_disjoint_splits_pass() -> None:
    check_split_disjointness(SYNTHETIC_SPLITS)


def test_image_in_two_splits_is_rejected() -> None:
    splits = {"train": ("im0001",), "validation": ("im0001",), "test": ()}
    with pytest.raises(ValueError, match="more than one split: im0001"):
        check_split_disjointness(splits)


def test_image_repeated_inside_one_split_is_rejected() -> None:
    with pytest.raises(ValueError, match="repeated inside one split: train/im0001"):
        check_split_disjointness({"train": ("im0001", "im0001"), "validation": (), "test": ()})


def test_crops_in_their_image_split_pass() -> None:
    counts = check_crop_manifest(
        [
            _crop("im0001_r000", "im0001", "train"),
            _crop("im0001_r001", "im0001", "train", region_index=1),
            _crop("im0003_r000", "im0003", "validation"),
        ],
        SYNTHETIC_SPLITS,
    )
    assert counts == {"crops": 3, "by_split": {"train": 2, "validation": 1, "test": 0}}


def test_crop_recorded_in_the_wrong_split_is_rejected() -> None:
    crop = _crop("im0001_r000", "im0001", "test")
    with pytest.raises(ValueError, match="recorded in test but image im0001 is in train"):
        check_crop_manifest([crop], SYNTHETIC_SPLITS)


def test_crop_of_unknown_image_is_rejected() -> None:
    crop = _crop("im9999_r000", "im9999", "train")
    with pytest.raises(ValueError, match="not in any split"):
        check_crop_manifest([crop], SYNTHETIC_SPLITS)


def test_duplicate_crop_id_is_rejected() -> None:
    crops = [
        _crop("im0001_r000", "im0001", "train"),
        _crop("im0001_r000", "im0001", "train"),
    ]
    with pytest.raises(ValueError, match="crop id im0001_r000 appears twice"):
        check_crop_manifest(crops, SYNTHETIC_SPLITS)


def test_crop_id_must_start_with_its_image_id() -> None:
    crop = _crop("im0002_r000", "im0001", "train")
    with pytest.raises(ValueError, match="does not start with its image im0001"):
        check_crop_manifest([crop], SYNTHETIC_SPLITS)
