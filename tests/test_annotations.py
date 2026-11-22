"""Tests for parsing the original VinText annotation format (synthetic fixtures only)."""

from __future__ import annotations

import pytest

from dfstr.data.annotations import (
    AnnotationError,
    format_image_id,
    image_id_from_image_name,
    image_id_from_label_name,
    iter_regions,
    parse_label_line,
)

QUAD = "0,0,10,0,10,5,0,5"


def test_parses_coordinates_and_transcript() -> None:
    region = parse_label_line(
        f"{QUAD},hoa", image_id="im0007", folder="train_images", region_index=2
    )
    assert region is not None
    assert region.quad == ((0.0, 0.0), (10.0, 0.0), (10.0, 5.0), (0.0, 5.0))
    assert region.transcript == "hoa"
    assert region.crop_id == "im0007_r002"


def test_transcript_keeps_commas() -> None:
    region = parse_label_line(
        f"{QUAD},a,b", image_id="im0007", folder="train_images", region_index=0
    )
    assert region is not None
    assert region.transcript == "a,b"


def test_blank_line_is_skipped() -> None:
    assert parse_label_line("  ", image_id="im0007", folder="f", region_index=0) is None


def test_unreadable_marker() -> None:
    region = parse_label_line(
        f"{QUAD},###", image_id="im0007", folder="train_images", region_index=0
    )
    assert region is not None
    assert region.unreadable


def test_too_few_fields_raises() -> None:
    with pytest.raises(AnnotationError, match="fields"):
        parse_label_line("0,0,10,0,10,5", image_id="im0007", folder="f", region_index=0)


def test_non_numeric_coordinate_raises() -> None:
    with pytest.raises(AnnotationError, match="non-numeric"):
        parse_label_line("0,0,ten,0,10,5,0,5,hoa", image_id="im0007", folder="f", region_index=0)


def test_iter_regions_counts_non_blank_lines() -> None:
    text = f"{QUAD},hoa\n\n{QUAD},###\n"
    regions = list(iter_regions(text, image_id="im0007", folder="train_images"))
    assert [region.region_index for region in regions] == [0, 1]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("im0461.jpg", "im0461"),
        ("im461.jpg", "im0461"),
        ("gt_461.txt", "im0461"),
        ("notes.txt", None),
    ],
)
def test_image_ids_are_padded_to_the_file_name(name: str, expected: str | None) -> None:
    from_image = image_id_from_image_name(name)
    from_label = image_id_from_label_name(name)
    assert (from_image or from_label) == expected


@pytest.mark.parametrize("number", [1, 7, 461, 1200, 2000])
def test_image_id_rebuilds_the_image_file_name(number: int) -> None:
    # The id must be the image file stem, so `<folder>/<id>.jpg` opens without a mapping
    image_id = format_image_id(number)
    assert image_id_from_image_name(f"{image_id}.jpg") == image_id
    assert image_id_from_label_name(f"gt_{number}.txt") == image_id
