"""Access to the official VinText dataset, from an unpacked tree or straight from the zip.

Both back ends expose the same shape: the image folders found in the dataset, each with the
image ids it contains and the raw bytes of the label files that belong to it. Nothing here
interprets labels; :mod:`dfstr.data.audit` does that.
"""

from __future__ import annotations

import zipfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from dfstr.data.annotations import image_id_from_image_name, image_id_from_label_name

# Name of the directory inside the dataset that holds the splits and ``labels/``.
DATASET_DIR = "vietnamese"


# dataclass is a decorator that automatically generates an __init__ method and other methods for the class
@dataclass(frozen=True)  # frozen=True makes it hashable, so we can use it as a dict key
class FolderContents:
    """Image ids and label files for one image folder of the dataset.
    Because the dataset has 3 image folders but only 1 labels folder, we need to keep track of each image folder contains which image ids and label files.
    """

    folder: str
    image_ids: tuple[str, ...]  # "..." in a tuple means "any number of elements of this type before it"
    label_files: tuple[tuple[str, bytes], ...]


class Source(Protocol):
    """Something that can list the image folders of a VinText dataset."""

    def read_folders(self) -> Mapping[str, FolderContents]: ...  # any class that implements this method can be used as a Source


@dataclass
class _Buckets:
    """Used to store the image ids and label files for one image folder while reading the dataset.
    There are 3 images folders: test_image, train_images and unseen_test_images. But only 1 labels folder. So we need to keep track of which image ids belong to which image folder while reading the dataset.
    """

    images: list[str] = field(default_factory=list)
    labels: list[tuple[str, bytes]] = field(default_factory=list)


def _finish(buckets: Mapping[str, _Buckets]) -> dict[str, FolderContents]:
    """Converts the buckets to FolderContents, sorting the image ids and label files."""
    
    return {
        folder: FolderContents(
            folder=folder,
            image_ids=tuple(sorted(bucket.images, key=_image_sort_key)),
            label_files=tuple(sorted(bucket.labels)),
        )
        for folder, bucket in buckets.items()
    }


def _image_sort_key(image_id: str) -> tuple[str, int]:
    """Sorts image ids by the prefix and then by the numeric suffix, if present"""

    prefix, _, number = image_id.rpartition("m")
    return (prefix, int(number) if number.isdigit() else -1)


@dataclass
class DirectorySource:
    """A class to read the VinText dataset from an unpacked directory tree."""

    root: Path

    def read_folders(self) -> dict[str, FolderContents]:
        buckets: dict[str, _Buckets] = {}
        for entry in sorted(self.root.iterdir()):
            if not entry.is_dir() or entry.name == "labels":
                continue

            bucket = buckets.setdefault(entry.name, _Buckets())
            for path in sorted(entry.iterdir()):
                image_id = image_id_from_image_name(path.name)
                if image_id is not None:
                    bucket.images.append(image_id)

        labels = self.root / "labels"
        if not labels.is_dir():
            raise FileNotFoundError(f"no labels/ directory under {self.root}")
        
        for path in sorted(labels.glob("gt_*.txt")):
            image_id = image_id_from_label_name(path.name)
            if image_id is None:
                continue

            folder = _folder_of_image(image_id, buckets)
            buckets.setdefault(folder, _Buckets()).labels.append((path.name, path.read_bytes()))

        return _finish(buckets)


@dataclass
class ArchiveSource:
    """A class to read the VinText dataset from a zip file."""

    path: Path
    prefix: str = f"{DATASET_DIR}/"

    def read_folders(self) -> dict[str, FolderContents]:
        buckets: dict[str, _Buckets] = {}
        label_names: list[tuple[str, str]] = []
        with zipfile.ZipFile(self.path) as archive:
            for name in archive.namelist():
                if not name.startswith(self.prefix):
                    continue

                folder, _, file_name = name[len(self.prefix) :].partition("/")
                if not file_name or "/" in file_name:
                    continue

                image_id = image_id_from_image_name(file_name)
                if image_id is not None:
                    buckets.setdefault(folder, _Buckets()).images.append(image_id)
                    continue

                if image_id_from_label_name(file_name) is not None:
                    label_names.append((folder, name))

            for _, name in label_names:
                file_name = name.rsplit("/", 1)[-1]
                image_id = image_id_from_label_name(file_name)
                folder = _folder_of_image(image_id, buckets)
                buckets[folder].labels.append((file_name, archive.read(name)))

        buckets.pop("labels", None)

        return _finish(buckets)


def _folder_of_image(image_id: str | None, buckets: Mapping[str, _Buckets]) -> str:
    # Labels live in a single labels/ directory, so an image id identifies its folder
    for folder, bucket in buckets.items():
        if image_id in bucket.images:
            return folder
    raise KeyError(f"label for image {image_id} has no image in any folder")


def open_source(root: Path | None = None, archive: Path | None = None) -> Source:
    # Input: either an unpacked dataset root or a dataset zip; Output: a matching Source
    if (root is None) == (archive is None):
        raise ValueError("pass exactly one of root or archive")
    if archive is not None:
        return ArchiveSource(path=archive)
    return DirectorySource(root=root)  # type: ignore[arg-type]
