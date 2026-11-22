"""Tests for the VinText audit report, built from a synthetic release only."""

from __future__ import annotations

import json
import unicodedata
import zipfile
from pathlib import Path

import pytest

from dfstr.data.audit import build_report
from dfstr.data.vintext_source import ArchiveSource, DirectorySource, open_source

QUAD = "0,0,10,0,10,5,0,5"

HOA_OLD = "h\u00f2a"  # hòa
HOA_NEW = "ho\u00e0"  # hoà
HOA_NEW_NFD = unicodedata.normalize("NFD", HOA_NEW)
DI = "\u0111i"  # đi
KHOE = "kho\u1ebb"  # khoẻ
THUY = "thu\u1ef7"  # thuỷ

TRAIN_LABELS = {
    "gt_1.txt": [f"{QUAD},{HOA_OLD}", f"{QUAD},###", f"{QUAD},{DI}", f"{QUAD},"],
    "gt_2.txt": [f"{QUAD},{HOA_NEW_NFD}"],
}
HELDOUT_LABELS = {"gt_3.txt": [f"{QUAD},{KHOE}", f"{QUAD},{THUY}", f"{QUAD},##"]}


def _write_release(root: Path) -> Path:
    release = root / "vietnamese"
    for folder, labels in (("train_images", TRAIN_LABELS), ("heldout_images", HELDOUT_LABELS)):
        (release / folder).mkdir(parents=True)
        for name in labels:
            image_id = int(name.removeprefix("gt_").removesuffix(".txt"))
            (release / folder / f"im{image_id}.jpg").write_bytes(b"")
    (release / "labels").mkdir()
    for labels in (TRAIN_LABELS, HELDOUT_LABELS):
        for name, lines in labels.items():
            (release / "labels" / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return release


def _write_archive(root: Path, release: Path) -> Path:
    archive = root / "vintext_original.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for path in sorted(release.rglob("*")):
            if path.is_file():
                zf.write(path, f"vietnamese/{path.relative_to(release).as_posix()}")
    return archive


@pytest.fixture
def release(tmp_path: Path) -> Path:
    return _write_release(tmp_path)


def test_counts_per_folder(release: Path) -> None:
    report = build_report(DirectorySource(root=release).read_folders())
    assert report["dataset"] == {
        "images": 3,
        "label_files": 3,
        "regions": 8,
        "unreadable_regions": 1,
        "empty_transcripts": 1,
    }
    train = report["folders"]["train_images"]
    assert (train["images"], train["regions"], train["unreadable_regions"]) == (2, 5, 1)
    assert train["empty_transcripts"] == 1
    assert train["hash_only_by_length"] == {"3": 1}
    assert train["regions_per_image"] == {"min": 1, "max": 4, "mean": 2.5}
    assert train["labels_without_image"] == 0
    assert train["images_without_regions"] == 0
    heldout = report["folders"]["heldout_images"]
    assert heldout["hash_only_by_length"] == {"2": 1}


def test_normalization_is_measured_per_folder(release: Path) -> None:
    report = build_report(DirectorySource(root=release).read_folders())
    train = report["folders"]["train_images"]
    assert train["normalization"] == {"nfc": 1, "nfd": 1, "plain": 1}
    assert train["labels_changed_by_nfc"] == 1


def test_label_stats_use_the_requested_folder(release: Path) -> None:
    report = build_report(DirectorySource(root=release).read_folders())
    stats = report["label_stats"]
    assert stats["folder"] == "train_images"
    assert stats["words"] == 3
    assert stats["length_histogram"] == {2: 1, 3: 2}
    assert stats["tone"]["huyen"] == 2
    assert stats["tone"]["none"] == 6
    assert stats["no_tone_rate"] == 0.75
    assert stats["tone_placement"] == {"new": 1, "no_pattern": 1, "old": 1}


def test_charset_check_reports_out_of_charset_characters(release: Path) -> None:
    folders = DirectorySource(root=release).read_folders()
    report = build_report(folders, charset=frozenset("hoai"))
    coverage = report["label_stats"]["charset"]
    assert coverage["charset_size"] == 4
    assert "\u0111" in coverage["out_of_charset_chars"]


def test_charset_coverage_is_reported_per_folder(release: Path) -> None:
    folders = DirectorySource(root=release).read_folders()
    report = build_report(folders, charset=frozenset("hoai"))
    for folder in report["folders"].values():
        assert "charset" in folder
        assert "out_of_charset_occurrences" in folder["charset"]


def test_no_charset_block_without_a_charset(release: Path) -> None:
    report = build_report(DirectorySource(root=release).read_folders())
    assert "charset" not in report["label_stats"]
    assert all("charset" not in folder for folder in report["folders"].values())


def test_unknown_stats_folder_is_rejected(release: Path) -> None:
    folders = DirectorySource(root=release).read_folders()
    with pytest.raises(KeyError):
        build_report(folders, stats_folder="does_not_exist")


def test_report_contains_no_transcripts(release: Path) -> None:
    report = build_report(DirectorySource(root=release).read_folders())
    dumped = json.dumps(report, ensure_ascii=False)
    for labels in (TRAIN_LABELS, HELDOUT_LABELS):
        for lines in labels.values():
            for line in lines:
                transcript = line.split(",")[-1]
                if transcript and set(transcript) != {"#"}:
                    assert transcript not in dumped


def test_archive_and_directory_sources_agree(tmp_path: Path) -> None:
    release = _write_release(tmp_path)
    archive = _write_archive(tmp_path, release)
    from_archive = build_report(ArchiveSource(path=archive).read_folders())
    from_directory = build_report(DirectorySource(root=release).read_folders())
    assert from_archive == from_directory


def test_open_source_requires_exactly_one_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        open_source()
    with pytest.raises(ValueError, match="exactly one"):
        open_source(root=tmp_path, archive=tmp_path / "x.zip")
