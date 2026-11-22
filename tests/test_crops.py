"""Tests for perspective rectification of word quadrilaterals (synthetic images only)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from dfstr.data.crops import (
    DROP_REASONS,
    RectifyError,
    order_quad,
    rectify,
    warp,
)

IMAGE_SIZE = {"image_width": 200, "image_height": 100}


def reason_of(quad, **kwargs) -> str:
    with pytest.raises(RectifyError) as caught:
        rectify(quad, **{**IMAGE_SIZE, **kwargs})
    return caught.value.reason


def test_order_quad_starts_top_left_and_runs_clockwise() -> None:
    stacked = [(10, 0), (10, 5), (0, 0), (0, 5)]  # bottom-right, bottom-left, top-left, top-right
    assert order_quad(stacked) == ((0.0, 0.0), (10.0, 0.0), (10.0, 5.0), (0.0, 5.0))


def test_order_quad_is_independent_of_input_order() -> None:
    quad = [(5, 5), (15, 5), (15, 9), (5, 9)]
    orders = {order_quad(permutation) for permutation in (quad, quad[::-1], quad[1:] + quad[:1])}
    assert len(orders) == 1


def test_order_quad_rejects_wrong_point_count() -> None:
    with pytest.raises(ValueError, match="4 points"):
        order_quad([(0, 0), (1, 1), (2, 2)])


def test_axis_aligned_quad_keeps_its_own_size() -> None:
    rectified = rectify([(10, 10), (40, 10), (40, 20), (10, 20)], **IMAGE_SIZE)
    assert (rectified.width, rectified.height) == (30, 10)
    assert rectified.orientation == "horizontal"
    assert not rectified.clamped


def test_vertical_word_is_not_rotated() -> None:
    rectified = rectify([(0, 0), (8, 0), (8, 40), (0, 40)], **IMAGE_SIZE)
    assert (rectified.width, rectified.height) == (8, 40)
    assert rectified.orientation == "vertical"


def test_slanted_quad_is_sized_from_its_edge_lengths() -> None:
    angle = math.radians(15)
    length, height = 40.0, 10.0
    corners = [
        (0.0, 0.0),
        (length * math.cos(angle), length * math.sin(angle)),
        (
            length * math.cos(angle) - height * math.sin(angle),
            length * math.sin(angle) + height * math.cos(angle),
        ),
        (-height * math.sin(angle), height * math.cos(angle)),
    ]
    rectified = rectify(corners, image_width=200, image_height=100)
    assert rectified.width == pytest.approx(length, abs=1.5)
    assert rectified.height == pytest.approx(height, abs=1.5)


def test_duplicate_points_are_dropped() -> None:
    assert reason_of([(0, 0), (0, 0), (10, 0), (0, 5)]) == "duplicate_points"


def test_collinear_quad_is_degenerate() -> None:
    assert reason_of([(0, 0), (5, 0), (10, 0), (15, 0)]) == "degenerate_area"


def test_concave_quad_is_dropped() -> None:
    # The fourth corner lies inside the triangle of the other three
    assert reason_of([(0, 0), (40, 0), (40, 40), (30, 5)]) == "non_convex"


def test_tiny_quad_is_too_small() -> None:
    assert reason_of([(0, 0), (1, 0), (1, 1), (0, 1)]) == "too_small"


def test_quad_entirely_outside_the_image_is_out_of_bounds() -> None:
    assert reason_of([(500, 500), (540, 500), (540, 540), (500, 540)]) == "out_of_bounds"


def test_quad_partly_outside_the_image_is_clipped_and_flagged() -> None:
    rectified = rectify([(-10, 10), (30, 10), (30, 20), (-10, 20)], **IMAGE_SIZE)
    assert rectified.clamped
    assert (rectified.width, rectified.height) == (30, 10)
    assert all(-0.0 <= x <= 199 for x, _ in rectified.quad)


def test_every_drop_reason_is_reachable() -> None:
    # Guards against a reason that no input can produce, which would hide a real drop case
    quads = {
        "duplicate_points": [(0, 0), (0, 0), (10, 0), (0, 5)],
        "degenerate_area": [(0, 0), (5, 0), (10, 0), (15, 0)],
        "non_convex": [(0, 0), (40, 0), (40, 40), (30, 5)],
        "out_of_bounds": [(500, 500), (540, 500), (540, 540), (500, 540)],
        "too_small": [(0, 0), (1, 0), (1, 1), (0, 1)],
    }
    assert set(quads) == set(DROP_REASONS)
    for expected, quad in quads.items():
        assert reason_of(quad) == expected


def test_warp_copies_the_quadrilateral_contents() -> None:
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    image[11:20, 5:45] = (0, 0, 255)  # a red bar, wider than the quad so its edges stay clear
    rectified = rectify([(7, 12), (43, 12), (43, 18), (7, 18)], image_width=50, image_height=50)
    crop = warp(image, rectified)
    assert crop.shape == (rectified.height, rectified.width, 3)
    assert (crop == np.array([0, 0, 255], dtype=np.uint8)).all()


def test_warp_output_shape_matches_rectified_size() -> None:
    image = np.full((60, 80, 3), 7, dtype=np.uint8)
    rectified = rectify([(10, 10), (60, 10), (60, 30), (10, 30)], image_width=80, image_height=60)
    crop = warp(image, rectified)
    assert crop.shape[:2] == (rectified.height, rectified.width)
