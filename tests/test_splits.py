"""Tests for the VinText split manifests (synthetic release only)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from dfstr.data.splits import (
    SPLIT_FOLDERS,
    build_manifest,
    read_manifest,
    split_counts,
    write_manifests,
)
from dfstr.data.vintext_source import FolderContents

QUAD = "0,0,10,0,10,5,0,5"


def _folder(name: str, image_ids: list[str], labels: dict[str, list[str]]) -> FolderContents:
    return FolderContents(
        folder=name,
        image_ids=tuple(image_ids),
        label_files=tuple(
            (label_name, ("\n".join(lines) + "\n").encode()) for label_name, lines in labels.items()
        ),
    )


def _folders() -> dict[str, FolderContents]:
    return {
        "train_images": _folder(
            "train_images",
            ["im0001", "im0002"],
            {
                "gt_1.txt": [f"{QUAD},hoa", f"{QUAD},###"],
                "gt_2.txt": [f"{QUAD},"],
            },
        ),
        "test_image": _folder("test_image", ["im0003"], {"gt_3.txt": [f"{QUAD},d\u1ec7"]}),
        "unseen_test_images": _folder(
            "unseen_test_images", ["im0004"], {"gt_4.txt": [f"{QUAD},hoa", f"{QUAD},##"]}
        ),
    }


def test_split_roles_are_the_agreed_folders() -> None:
    assert SPLIT_FOLDERS == {
        "train": "train_images",
        "validation": "test_image",
        "test": "unseen_test_images",
    }


def test_counts_split_usable_and_dropped() -> None:
    counts = split_counts(_folders()["train_images"])
    assert counts.as_dict() == {
        "folder": "train_images",
        "images": 2,
        "regions_usable": 1,
        "regions_dropped": 2,
    }


def test_manifest_payload() -> None:
    payload = build_manifest(_folders(), source_archive="release.zip", created="2026-11-20")
    assert payload["created"] == "2026-11-20"
    assert payload["images"] == {
        "train": ["im0001", "im0002"],
        "validation": ["im0003"],
        "test": ["im0004"],
    }
    assert payload["splits"]["validation"]["folder"] == "test_image"
    assert payload["splits"]["test"]["regions_dropped"] == 1
    assert "no reshuffle" in payload["rule"]


def test_manifest_is_deterministic() -> None:
    first = build_manifest(_folders(), source_archive="release.zip", created="2026-11-20")
    second = build_manifest(_folders(), source_archive="release.zip", created="2026-11-20")
    assert first == second


def test_missing_folder_is_rejected() -> None:
    folders = _folders()
    del folders["test_image"]
    with pytest.raises(KeyError, match="missing"):
        build_manifest(folders, source_archive="release.zip")


def test_repeated_image_id_is_rejected() -> None:
    folders = _folders()
    folders["test_image"] = _folder("test_image", ["im0001"], {})
    with pytest.raises(ValueError, match="more than one split"):
        build_manifest(folders, source_archive="release.zip")


def test_write_then_read_manifests(tmp_path: Path) -> None:
    payload = build_manifest(_folders(), source_archive="release.zip", created="2026-11-20")
    written = write_manifests(payload, tmp_path)
    assert {path.name for path in written} == {
        "train.txt",
        "validation.txt",
        "test.txt",
        "provenance.json",
    }
    assert read_manifest(tmp_path / "train.txt") == ("im0001", "im0002")
    provenance = json.loads((tmp_path / "provenance.json").read_text(encoding="utf-8"))
    assert "images" not in provenance
    assert provenance["source_archive"] == "release.zip"


def test_audit_report_is_hashed_into_provenance(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text("{}\n", encoding="utf-8")
    payload = build_manifest(_folders(), source_archive="release.zip", audit_report=report)
    assert payload["audit_report_sha256"] == hashlib.sha256(report.read_bytes()).hexdigest()
