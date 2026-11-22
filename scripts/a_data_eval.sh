#!/bin/bash

# 1. Download and unzip the data
echo ">>> Downloading and unzipping the data..."
uv sync --group data
mkdir -p data/raw
uv run gdown 1UUQhNvzgpZy7zXBFQp0Qox-BBjunZ0ml -O data/raw/vintext_original.zip
unzip data/raw/vintext_original.zip -d data/raw/vintext

# 2. Audit and create charset
echo ">>> Auditing and creating charset..."
uv run python scripts/audit_dataset.py --from_folder data/raw/vintext/vietnamese --out data/interim/audit
uv run python scripts/make_charset.py --from_folder data/raw/vintext/vietnamese --out configs/charset/baseline.txt
uv run python scripts/audit_dataset.py --from_folder data/raw/vintext/vietnamese --charset configs/charset/baseline.txt --out data/interim/audit

# 3. Create train/val/test splits
echo ">>> Creating train/val/test splits..."
uv run python scripts/make_splits.py --from_zip data/raw/vintext_original.zip --audit-report data/interim/audit/report.json
uv run pytest tests/test_splits.py