"""PARSeq-compatible LMDB writer: one store per split under ``data/lmdb/<split>``.

The layout agreed with the training side (Phase 2, step 16):

- 1-based ``num-samples``, ``image-%09d`` and ``label-%09d`` keys, the PARSeq convention, so
  ``LmdbDataset`` reads the store with no adapter.
- images are lossless PNG bytes at native crop resolution; resizing to ``32x128`` or a
  higher-resolution input is the ``DataLoader``'s job, not the store's.
- labels are plain UTF-8 strings (see :mod:`dfstr.data.training_labels`).

Writes are one LMDB transaction per split: either the whole split lands or nothing does, so a
half-written store cannot be mistaken for a complete one.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import cv2
import lmdb
import numpy as np

COUNT_KEY = "num-samples"
#: 2 GiB per split: native-resolution crops are far below this, and LMDB allocates lazily.
DEFAULT_MAP_SIZE = 2 * 1024**3


def image_key(index: int) -> str:
    if index < 1:
        raise ValueError(f"PARSeq image keys are 1-based, got {index}")
    return f"image-{index:09d}"


def label_key(index: int) -> str:
    if index < 1:
        raise ValueError(f"PARSeq label keys are 1-based, got {index}")
    return f"label-{index:09d}"


@dataclass(frozen=True)
class LmdbSample:
    """One crop as it enters the store: encoded image bytes plus its label."""

    image_png: bytes
    label: str


def encode_png(image: np.ndarray) -> bytes:
    # Input: a uint8 crop of any channel count; Output: lossless PNG bytes
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"PNG encoding failed for an image of shape {image.shape}")
    return buffer.tobytes()


def decode_png(data: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("PNG decoding failed")
    return image


def write_lmdb(
    path: Path,
    samples: Iterable[LmdbSample],
    *,
    map_size: int = DEFAULT_MAP_SIZE,
    overwrite: bool = False,
) -> int:
    # Input: a store directory and its samples in crop order; Output: the number of samples written
    # How: one transaction with num-samples, PNG images and UTF-8 labels under PARSeq keys
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{path} already holds a store; pass overwrite=True to replace it")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    written = 0
    environment = lmdb.open(str(path), map_size=map_size, subdir=True, lock=True, sync=True)
    try:
        with environment.begin(write=True) as transaction:
            for sample in samples:
                if not sample.image_png:
                    raise ValueError(f"sample {written + 1} of {path} has empty image bytes")
                if not sample.label:
                    raise ValueError(f"sample {written + 1} of {path} has an empty label")
                written += 1
                transaction.put(image_key(written).encode(), sample.image_png)
                transaction.put(label_key(written).encode(), sample.label.encode("utf-8"))
            transaction.put(COUNT_KEY.encode(), str(written).encode())
    finally:
        environment.close()
    return written


def read_lmdb(path: Path) -> tuple[int, list[tuple[bytes, str]]]:
    # Input: a store directory; Output: its num-samples count and its (PNG bytes, label) pairs
    # How: read the count key, then walk 1..count in key order
    environment = lmdb.open(str(path), readonly=True, lock=False, subdir=True)
    try:
        with environment.begin() as transaction:
            raw_count = transaction.get(COUNT_KEY.encode())
            if raw_count is None:
                raise ValueError(f"{path} has no {COUNT_KEY} key")
            count = int(raw_count.decode())
            samples: list[tuple[bytes, str]] = []
            for index in range(1, count + 1):
                image = transaction.get(image_key(index).encode())
                label = transaction.get(label_key(index).encode())
                if image is None or label is None:
                    raise ValueError(f"{path} is missing sample {index} of {count}")
                samples.append((image, label.decode("utf-8")))
    finally:
        environment.close()
    return count, samples


def lmdb_count(path: Path) -> int | None:
    # Input: a store directory; Output: its sample count, or None when there is no readable store
    if not (path / "data.mdb").exists():
        return None
    try:
        count, _ = read_lmdb(path)
    except (ValueError, lmdb.Error):
        return None
    return count
