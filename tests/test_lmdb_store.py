"""Tests for the PARSeq-compatible LMDB store (synthetic crops only)."""

from __future__ import annotations

from pathlib import Path

import lmdb
import numpy as np
import pytest

from dfstr.data.lmdb_store import (
    COUNT_KEY,
    LmdbSample,
    decode_png,
    encode_png,
    image_key,
    label_key,
    lmdb_count,
    read_lmdb,
    write_lmdb,
)


def crop(value: int) -> np.ndarray:
    return np.full((6, 14, 3), value, dtype=np.uint8)


def samples() -> list[LmdbSample]:
    return [LmdbSample(encode_png(crop(value)), "hoa") for value in (10, 20, 30)]


def test_page_keys_are_one_based_and_zero_padded() -> None:
    assert image_key(1) == "image-000000001"
    assert label_key(42) == "label-000000042"
    with pytest.raises(ValueError, match="1-based"):
        image_key(0)


def test_round_trip_keeps_pixels_and_labels(tmp_path: Path) -> None:
    path = tmp_path / "train"
    assert write_lmdb(path, samples()) == 3
    count, stored = read_lmdb(path)
    assert count == 3
    assert [label for _, label in stored] == ["hoa", "hoa", "hoa"]
    for value, (image_png, _) in zip((10, 20, 30), stored, strict=True):
        assert (decode_png(image_png) == crop(value)).all()


def test_num_samples_key_is_written(tmp_path: Path) -> None:
    path = tmp_path / "train"
    write_lmdb(path, samples())
    environment = lmdb.open(str(path), readonly=True, lock=False, subdir=True)
    try:
        with environment.begin() as transaction:
            assert transaction.get(COUNT_KEY.encode()) == b"3"
    finally:
        environment.close()


def test_png_round_trip_is_lossless() -> None:
    original = np.arange(6 * 14 * 3, dtype=np.uint8).reshape(6, 14, 3)
    assert (decode_png(encode_png(original)) == original).all()


def test_existing_store_is_not_overwritten_silently(tmp_path: Path) -> None:
    path = tmp_path / "train"
    write_lmdb(path, samples())
    with pytest.raises(FileExistsError, match="already holds a store"):
        write_lmdb(path, samples())


def test_overwrite_replaces_the_whole_store(tmp_path: Path) -> None:
    path = tmp_path / "train"
    write_lmdb(path, samples())
    assert write_lmdb(path, samples()[:1], overwrite=True) == 1
    assert lmdb_count(path) == 1


def test_empty_label_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "train"
    with pytest.raises(ValueError, match="empty label"):
        write_lmdb(path, [LmdbSample(encode_png(crop(1)), "")])


def test_missing_store_has_no_count(tmp_path: Path) -> None:
    assert lmdb_count(tmp_path / "absent") is None


def test_repeated_writes_are_byte_identical(tmp_path: Path) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    write_lmdb(first, samples())
    write_lmdb(second, samples())
    assert (first / "data.mdb").read_bytes() == (second / "data.mdb").read_bytes()
