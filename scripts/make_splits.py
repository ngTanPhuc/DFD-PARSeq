#!/usr/bin/env python3
"""Write the VinText split manifests under data/splits/.

Copies the split assignment that ships with the release; never resamples. Writes image ids
only, plus a provenance record with usable and dropped region counts.

Examples:
    uv run python scripts/make_splits.py --from_zip data/raw/vintext_original.zip \
        --audit-report data/interim/audit/report.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running this file directly, without installing the package (see pyproject: package = false)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dfstr.data.splits import SPLITS, build_manifest, write_manifests  # noqa: E402
from dfstr.data.vintext_source import open_source  # noqa: E402

DEFAULT_OUT = Path("data/splits")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--from_zip", type=Path, help="VinText zip, read in place")
    source.add_argument("--from_folder", type=Path, help="unpacked vietnamese/ directory")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"default: {DEFAULT_OUT}")
    parser.add_argument("--audit-report", type=Path, help="report.json to hash into provenance")
    parser.add_argument("--created", help="ISO date recorded in provenance (default: today)")
    args = parser.parse_args(argv)

    folders = open_source(root=args.from_folder, archive=args.from_zip).read_folders()
    payload = build_manifest(
        folders,
        source_archive=str(args.from_zip or args.from_folder),
        created=args.created,
        audit_report=args.audit_report,
    )
    written = write_manifests(payload, args.out)

    print(f"{'split':<12} {'folder':<20} {'images':>7} {'regions':>9} {'dropped':>8}")
    for split in SPLITS:
        counts = payload["splits"][split]
        print(
            f"{split:<12} {counts['folder']:<20} {counts['images']:>7} "
            f"{counts['regions_usable']:>9} {counts['regions_dropped']:>8}"
        )
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
