"""Perspective rectification of VinText word quadrilaterals into upright word crops.

A VinText label gives four corner points in image pixels. Rectification orders them to
top-left, top-right, bottom-right, bottom-left, decides the output size from the quadrilateral's
own edge lengths, and warps that quadrilateral onto an axis-aligned rectangle, so a slanted word
becomes a horizontal one.

Crops are produced at native resolution: the declared size is the quadrilateral's own size, never
a fixed 32x128, and resizing happens in the DataLoader. Vertical words (``height > width``) keep
their upright orientation; the crop is not rotated by 90 degrees.

Regions that cannot produce a sane crop are dropped with a reason instead of being guessed at.
The reasons are the strings in :data:`DROP_REASONS` and every drop is counted, so the count per
reason can go into the data notes while the region itself is discarded.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

import cv2
import numpy as np

Point: TypeAlias = tuple[float, float]
Quad: TypeAlias = tuple[Point, Point, Point, Point]

# A crop narrower or shorter than this (pixels) carries no usable glyph information.
MIN_SIDE = 2

# A quadrilateral smaller than this (square pixels) is a label artifact, not a word.
MIN_AREA = 1.0

DROP_REASONS: tuple[str, ...] = (
    "duplicate_points",
    "degenerate_area",
    "non_convex",
    "out_of_bounds",
    "too_small",
)


class RectifyError(ValueError):
    """Raised when a quadrilateral cannot be turned into a crop; ``reason`` names why."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class Rectified:
    """A quadrilateral that is safe to warp, with the output size it maps onto."""

    quad: Quad
    width: int
    height: int
    clamped: bool

    @property
    def orientation(self) -> str:
        return "vertical" if self.height > self.width else "horizontal"


def _signed_area(ring: Sequence[Point]) -> float:
    total = 0.0
    for index, (x1, y1) in enumerate(ring):
        x2, y2 = ring[(index + 1) % len(ring)]
        total += x1 * y2 - x2 * y1
    return total / 2


def _cross(origin: Point, first: Point, second: Point) -> float:
    return (first[0] - origin[0]) * (second[1] - origin[1]) - (first[1] - origin[1]) * (
        second[0] - origin[0]
    )


def order_quad(points: Sequence[Point]) -> Quad:
    # Input: four corner points in any order; Output: (top-left, top-right, bottom-right, bottom-left)
    # How: sort the corners by angle around the centroid, then rotate and flip the ring so it starts at the most top-left corner and runs clockwise on screen (positive signed area)
    if len(points) != 4:
        raise ValueError(f"a quadrilateral has 4 points, got {len(points)}")
    if len(set(points)) != 4:
        raise RectifyError("duplicate_points")
    centre_x = sum(point[0] for point in points) / 4
    centre_y = sum(point[1] for point in points) / 4
    ring = sorted(points, key=lambda p: math.atan2(p[1] - centre_y, p[0] - centre_x))
    start = min(
        range(len(ring)),
        key=lambda i: (ring[i][0] + ring[i][1], ring[i][1], ring[i][0]),
    )
    ring = ring[start:] + ring[:start]
    if _signed_area(ring) < 0:
        ring = [ring[0], *reversed(ring[1:])]
    return (ring[0], ring[1], ring[2], ring[3])


def _is_convex(ring: Quad) -> bool:
    # In image coordinates (y grows downwards) a clockwise ring has a positive cross product
    # at every corner; a bow-tie or a concave quadrilateral does not
    return all(
        _cross(ring[index], ring[(index + 1) % 4], ring[(index + 2) % 4]) > 0 for index in range(4)
    )


def _side_length(first: Point, second: Point) -> float:
    return math.hypot(second[0] - first[0], second[1] - first[1])


def _clip(point: Point, width: int, height: int) -> Point:
    return (
        min(max(point[0], 0.0), float(width - 1)),
        min(max(point[1], 0.0), float(height - 1)),
    )


def rectify(
    quad: Sequence[Point],
    *,
    image_width: int,
    image_height: int,
    clip: bool = True,
) -> Rectified:
    # Input: a label quadrilateral plus the source image size; Output: the rectified geometry
    # Raises RectifyError with one of DROP_REASONS when the region cannot be cropped
    if image_width < 1 or image_height < 1:
        raise ValueError(f"image size must be positive, got {image_width}x{image_height}")
    ring = order_quad(quad)
    if abs(_signed_area(ring)) < MIN_AREA:
        raise RectifyError("degenerate_area")
    if not _is_convex(ring):
        raise RectifyError("non_convex")
    clamped = False
    if clip:
        clipped = tuple(_clip(point, image_width, image_height) for point in ring)
        clamped = clipped != ring
        if clamped:
            # Clipping can collapse a quadrilateral that sat outside the image, so the same
            # two sanity checks run again and report the region as out of bounds
            try:
                ring = order_quad(clipped)
            except RectifyError as err:
                raise RectifyError("out_of_bounds") from err
            if abs(_signed_area(ring)) < MIN_AREA or not _is_convex(ring):
                raise RectifyError("out_of_bounds")
    top_left, top_right, bottom_right, bottom_left = ring
    width = round(max(_side_length(top_left, top_right), _side_length(bottom_left, bottom_right)))
    height = round(max(_side_length(top_left, bottom_left), _side_length(top_right, bottom_right)))
    if width < MIN_SIDE or height < MIN_SIDE:
        raise RectifyError("too_small")
    return Rectified(quad=ring, width=width, height=height, clamped=clamped)


def warp(image: np.ndarray, rectified: Rectified) -> np.ndarray:
    # Input: the source image (any channel count) and its rectified geometry
    # Output: the rectified crop, height x width, same channel count
    source = np.asarray(rectified.quad, dtype=np.float32)
    target = np.array(
        [
            [0.0, 0.0],
            [rectified.width - 1.0, 0.0],
            [rectified.width - 1.0, rectified.height - 1.0],
            [0.0, rectified.height - 1.0],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(source, target)
    return cv2.warpPerspective(
        image,
        matrix,
        (rectified.width, rectified.height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
