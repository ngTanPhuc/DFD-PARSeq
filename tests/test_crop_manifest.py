"""Tests for the crop manifest round-trip (synthetic crop ids only)."""

from __future__ import annotations

from pathlib import Path

import pytest

from dfstr.data.crop_manifest import (
    MANIFEST_COLUMNS,
    CropRecord,
    read_manifest,
    sort_records,
    write_manifest,
)


def record(
    crop_id: str, split: str = "train", *, index: int = 0, clamped: bool = False
) -> CropRecord:
    image_id = crop_id.split("_r")[0]
    return CropRecord(
        crop_id=crop_id,
        image_id=image_id,
        split=split,
        region_index=index,
        width=42,
        height=11,
        orientation="horizontal",
        clamped=clamped,
    )


def test_manifest_round_trip(tmp_path: Path) -> None:
    records = [record("im0001_r000", index=0), record("im0002_r003", index=3, clamped=True)]
    path = tmp_path / "crops" / "manifest.csv"
    assert write_manifest(records, path) == 2
    assert read_manifest(path) == records


def test_manifest_has_the_agreed_columns(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    write_manifest([record("im0001_r000")], path)
    header = path.read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",") == list(MANIFEST_COLUMNS)


def test_manifest_is_sorted_by_split_then_image_then_region() -> None:
    records = [
        record("im0002_r000", "test"),
        record("im0001_r002", index=2),
        record("im0001_r001", index=1),
    ]
    ordered = sort_records(records, ["train", "validation", "test"])
    assert [r.crop_id for r in ordered] == ["im0001_r001", "im0001_r002", "im0002_r000"]


def test_manifest_holds_no_transcript_text(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    write_manifest([record("im0001_r000")], path)
    assert MANIFEST_COLUMNS == (
        "crop_id",
        "image_id",
        "split",
        "region_index",
        "width",
        "height",
        "orientation",
        "clamped",
    )


def test_missing_column_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    path.write_text("crop_id,image_id\nim0001_r000,im0001\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        read_manifest(path)
