"""Parsing of the official VinText annotation format.

Each label file holds one quadrilateral per line in the *original* annotation format:
``x1,y1,x2,y2,x3,y3,x4,y4,TRANSCRIPT``. The transcript may itself contain commas, so the
first eight comma-separated fields are coordinates and everything after belongs to the
transcript. Verified against the official release on 19 Nov 2026: 56,084 lines, all with
eight integer coordinates, 9 or 10 fields.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

COORD_FIELDS = 8

# Transcripts that mark a region as unreadable in the official release.
UNREADABLE_MARKERS = frozenset({"###"})

# Image names in the release are zero-padded to four digits (im0001.jpg … im2000.jpg) while
# label names are not (gt_1.txt), so the id is built from the image name and labels are padded
# to match. Measured on all 2,000 names of the official release: every one is `im` + 4 digits.
IMAGE_ID_DIGITS = 4

_LABEL_NAME_RE = re.compile(r"^gt_(\d+)\.txt$")  # gt_1.txt, gt_2.txt, etc.
_IMAGE_NAME_RE = re.compile(r"^im(\d+)\.(?:jpg|jpeg|png)$", re.IGNORECASE)  # im0001.jpg, im1.png, etc.


class AnnotationError(ValueError):
    """Error to raise when an annotation line does not follow the original VinText format."""


def is_usable_transcript(transcript: str) -> bool:
    # Input: one transcript; Output: False for unreadable markers and empty transcripts
    # How: the release marks unreadable regions with hashes and has lines ending in a bare comma
    return bool(transcript) and set(transcript) != {"#"}


@dataclass(frozen=True)
class Region:
    """One labeled word region: a quadrilateral plus its transcript."""

    image_id: str
    folder: str
    region_index: int
    quad: tuple[tuple[float, float], ...]
    transcript: str

    @property  # @property decorator makes this method accessible like an attribute
    def crop_id(self) -> str:
        return f"{self.image_id}_r{self.region_index:03d}"

    @property
    def unreadable(self) -> bool:
        return self.transcript in UNREADABLE_MARKERS


def format_image_id(number: int) -> str:
    """
    Input: the numeric part of a VinText image name
    Output: the canonical image id, matching the image file name (``im0001``)
    """
    return f"im{number:0{IMAGE_ID_DIGITS}d}"


def image_id_from_image_name(name: str) -> str | None:
    """
    Input: a bare image file name
    Output: canonical image id (its file stem, e.g. ``im0001``), or None if not a VinText image
    """
    match = _IMAGE_NAME_RE.match(name)
    return format_image_id(int(match.group(1))) if match else None


def image_id_from_label_name(name: str) -> str | None:
    """
    Input: a bare label file name (``gt_<n>.txt``)
    Output: canonical image id (``gt_1.txt`` → ``im0001``), or None
    """
    match = _LABEL_NAME_RE.match(name)
    return format_image_id(int(match.group(1))) if match else None


def parse_label_line(
    line: str, *, image_id: str, folder: str, region_index: int
) -> Region | None:
    """
    Parses one line of a VinText label file into a Region object.
    Input: one annotation line, the owning image id and folder, and the region index
    Output: the parsed Region, or None when the line is blank
    How: split on commas, the first eight fields are coordinates, the rest is the transcript (transcripts may contain commas like "hello, world").
    """
    stripped = line.strip()
    if not stripped:
        return None
    fields = stripped.split(",")
    if len(fields) < COORD_FIELDS:
        raise AnnotationError(
            f"{folder}/{image_id} region {region_index}: "
            f"{len(fields)} comma-separated fields, expected at least {COORD_FIELDS}"
        )
    try:
        coords = [float(value) for value in fields[:COORD_FIELDS]]
    except ValueError as err:
        raise AnnotationError(
            f"{folder}/{image_id} region {region_index}: non-numeric coordinate in "
            f"{fields[:COORD_FIELDS]!r}"
        ) from err
    quad = tuple((coords[i], coords[i + 1]) for i in range(0, COORD_FIELDS, 2))
    return Region(
        image_id=image_id,
        folder=folder,
        region_index=region_index,
        quad=quad,
        transcript=",".join(fields[COORD_FIELDS:]),
    )


def iter_regions(text: str, *, image_id: str, folder: str) -> Iterator[Region]:
    """
    Iterates over the lines in a label file, yielding the parsed Region objects.
    Input: the decoded text of one label file
    Output: its regions in file order
    """
    region_index = 0
    for line in text.splitlines():
        region = parse_label_line(
            line, image_id=image_id, folder=folder, region_index=region_index
        )
        if region is not None:
            yield region
            region_index += 1
