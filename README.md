# Diacritic-Factorized Decoding for Vietnamese Scene Text Recognition

This repository is the codebase for the CO3133 Deep Learning and Applications (HCMUT) group project **Diacritic-Factorized Decoding for Vietnamese Scene Text Recognition**. The task is scene text *recognition* only: the input is a cropped image of one word and the output is a Vietnamese Unicode string with diacritics. Text detection is out of scope.

The base model is **PARSeq** (Bautista & Atienza, ECCV 2022; `baudm/parseq`, Apache-2.0). The project idea is to replace PARSeq's single per-position softmax with **three parallel heads** (base letter, modifier/variant, tone), optionally use factorized embeddings for previous-token context, and apply a soft syllable-validity mask at decode time. The dataset is **VinText** (CVPR 2021). The main metric is **DER** (diacritic error rate), supported by **UDER**, DER on the intersection set, word accuracy, and 1-NED. The project deadline is **Sunday 13 December 2026**.

**Status:** *Phase 0, not started.* The team updates this line at each phase.

## Team

| Role | Name | Responsibility |
|---|---|---|
| A | Phuc | Data and evaluation: VinText download/audit, crops, splits, leakage tests, metrics code, final test run (once). |
| B | Tri | Baseline and training infrastructure: PARSeq setup, charset/NFC changes, pilot E0, E0b, E1, E6, GPU/Slurm scheduling. |
| C | Qui | Proposed model: three-head output, loss, factorized context, syllable mask (E2, E3, E4, E5). |
| D | Hien | Generalization, analysis and report: E7 design, error analysis, related work, LNCS report, README/packaging. |

Compute: Kaggle/Colab.

## Repository structure

```
.
├── configs/                 # Hydra/YAML configuration for training and evaluation
│   ├── charset/             # Character-set definitions (baseline 228-char set, component vocabularies)
│   └── experiment/          # One YAML per experiment ID (E0, E0b, E1-E7)
├── data/                    # All datasets and derived data
│   ├── raw/                 # Original VinText download, untouched (contents ignored)
│   ├── interim/             # Word crops, intermediate files, audit outputs (contents ignored)
│   ├── processed/           # Training-ready data, e.g. LMDB (contents ignored)
│   └── splits/              # Split manifests, near-duplicate groups, rare-combination holdouts (tracked)
├── docs/                    # Proposal PDF, decision records, data-audit notes, PARSeq reuse notes
├── notebooks/               # Exploratory notebooks (data audit, error analysis)
├── outputs/                 # Hydra runs, checkpoints, TensorBoard logs, Slurm logs (contents ignored)
├── reports/                 # LaTeX/LNCS report sources and generated assets
│   ├── figures/             # Figures used in the report
│   └── paper/               # LNCS report source (.tex, .bib)
├── results/                 # Final numbers and predictions used in the report
│   ├── metrics/             # Per-run and aggregated metrics tables (CSV/JSON)
│   └── predictions/         # Per-word prediction files keyed by crop ID
├── scripts/                 # Command-line entry points
│   └── slurm/               # Slurm job scripts for the HPC/GPU cluster
├── src/                     # Python source root (src layout)
│   └── dfstr/               # Project package
│       ├── data/            # Crop generation, label normalization I/O, datasets/dataloaders, split utilities
│       ├── labels/          # Vietnamese decomposition, đ mapping, NFC handling, syllable-validity rules
│       ├── models/          # Three-head output, factorized embeddings, loss, decoding wrappers
│       └── evaluation/      # Metrics, error categorization, paired bootstrap
├── tests/                   # Unit tests for definitions the report depends on
└── third_party/             # External code kept separate from ours (PARSeq goes in third_party/parseq/)
```

## Setup

1. Install [`uv`](https://docs.astral.sh/uv/) and clone this repository:

   ```bash
   git clone git@github.com:ngTanPhuc/DFD-PARSeq.git
   cd DFD-PARSeq
   ```

2. Create/update the environment (core dependencies and the `dev` group):

   ```bash
   uv sync
   ```

3. **Add PARSeq** (a maintainer does this once; the exact commit is recorded in `docs/` and in this README under "Pinned versions"):

   ```bash
   git submodule add https://github.com/baudm/parseq third_party/parseq
   git -C third_party/parseq checkout 1902db043c029a7e03a3818c616c06600af574be
   ```

   PARSeq is Apache-2.0. Anyone cloning afterwards runs `git submodule update --init`.

4. **Install PARSeq's own requirements** following its README. At the time of writing it requires Python ≥ 3.9 and PyTorch ≥ 2.0 and installs with a platform choice (`cpu`, `cu118`, `cu121`, `rocm5.7`):

   ```bash
   source .venv/bin/activate
   cd third_party/parseq
   platform=<cpu|cu118|cu121|rocm5.7>      # TODO(Tri): choose for our GPU
   make torch-${platform}
   # install into the project's virtual environment
   pip install -r requirements/core.${platform}.txt -e .[train,test]
   ```

   If versions conflict with `pyproject.toml`, **do not silently change either side**: record the conflict and the resolution in `docs/` and tell the team.

5. Sanity check:

   ```bash
   uv run pytest
   uv run ruff check .
   ```

### Pinned versions

| Item | Value |
|---|---|
| PARSeq commit | 1902db043c029a7e03a3818c616c06600af574be |
| Python | TODO(user) |
| torch | TODO(user) |
| CUDA | TODO(user) |
| GPU driver | TODO(user) |

## Getting the dataset (VinText)

**Terms of use.** By downloading, users agree to use the dataset for **research or educational purposes only**, **not to distribute** any part of it in original or modified form, and to **cite the paper** when publishing results. The official repository is listed under AGPL-3.0, but the dataset terms above are stricter; **the team follows the stricter terms**. Consequently, never commit images, crops, LMDB files, or ground-truth transcripts. This is enforced by `.gitignore` and `AGENT.md`.

**Official source:** the VinAI `dict-guided` repository, `https://github.com/VinAIResearch/dict-guided` (Download section). Do **not** use third-party mirrors (for example Hugging Face copies); their license/statistics may differ from the official page.

**Which version to download:** the **Original** annotation format (`x1,y1,x2,y2,x3,y3,x4,y4,TRANSCRIPT`), because the project crops words by **perspective rectification from the quadrilaterals**. The "Converted dataset" is COCO format; it is not needed.

**Commands** (Google Drive file ID taken from the official README on 9 Nov 2026; re-check it on the official page if the download fails):

```bash
uv sync --group data
mkdir -p data/raw
uv run gdown 1UUQhNvzgpZy7zXBFQp0Qox-BBjunZ0ml -O data/raw/vintext_original.zip
unzip data/raw/vintext_original.zip -d data/raw/vintext
find data/raw/vintext -maxdepth 3 | head -50
```

If `gdown` hits Google Drive quota limits, download in a browser from `https://drive.google.com/file/d/1UUQhNvzgpZy7zXBFQp0Qox-BBjunZ0ml/view?usp=sharing` and unzip into `data/raw/vintext/`.

**Expected facts to verify after download** (a checklist, not assumptions):

- [ ] 2,000 images.
- [ ] About 56,084 text instances.
- [ ] Train and test splits exist.
- [ ] No official validation split. If none exists, member A creates a deterministic image-level validation split of 300 images and records it in `data/splits/`.
- [ ] How unreadable regions are marked (commonly `###`).
- [ ] The actual folder layout after unzip.

TODO(Phuc): paste the output of the `find` command here and update this section. The official page's folder layout description is inconsistent, so do not invent a layout.

**Citation.** *Dictionary-Guided Scene Text Recognition*, Nguyen et al., CVPR 2021. https://openaccess.thecvf.com/content/CVPR2021/papers/Nguyen_Dictionary-Guided_Scene_Text_Recognition_CVPR_2021_paper.pdf

## Workflow by phase

| Phase | Dates (2026) | Goal | Key outputs | Exit criterion |
|---|---|---|---|---|
| 0. Setup | 9 Nov → 10 Nov | Repo, environment, PARSeq added, data downloaded | Working `uv` env, PARSeq pinned, raw data in `data/raw/` | Everyone can run `uv run pytest` and see the raw data |
| 1. Data audit and pipeline | 9 Nov → 14 Nov | Trustworthy data and leakage-free splits | Crops in `data/interim/`, split manifests in `data/splits/`, audit notes in `docs/`, label statistics (base/modifier/tone, length), `đ` and hòa/hoà convention chosen, near-duplicate groups (perceptual hash), LMDB in `data/processed/` | Leakage tests pass; crop count recorded; unreadable-region rule recorded |
| 2. Pilot and decision gate | 9 Nov → 15 Nov | Is there headroom for the idea? | Pilot E0 (PARSeq, NFC, charset 228, 128×32): time per run, validation curve; error taxonomy of E0 on validation | **Gate on 15 Nov** (see below) |
| 3. Core experiments (P0) | 16 Nov → 22 Nov | Main comparison | E0, E0b (split-sequence baseline), E2 (three heads), 3 seeds each, metrics in `results/metrics/` | E0, E0b, E2 complete on validation with seeds; paired bootstrap prepared |
| 4. Resolution and generalization (P1) | 23 Nov → 29 Nov | Is the gain only from resolution? Does it generalize to rare combinations? | E1 (higher resolution), E7 (hold out rare combinations; both PARSeq and proposed model, 6 runs), E3 conditional on E1 | RQ2 and RQ5 answered on validation |
| 5. Ablations and dictionary (P2) | 30 Nov → 5 Dec | Which components help | E4 (factorized context), E5 (soft syllable mask), E6 (dictionary post-processing, dictionary built from train labels only) | Ablation table complete, even if some cells are single-seed |
| 6. Freeze and test | 6 Dec → 10 Dec | One clean test evaluation | Frozen configs, **test run once** by member A, error analysis, failure examples | Test numbers produced once; every report table rebuilt from `results/` |
| 7. Report and packaging | 23 Nov → 12 Dec (in parallel, final polish 11 Dec → 12 Dec) | English LNCS report + reproducible repo | `reports/paper/`, `reports/figures/`, README, checkpoint instructions, AI-usage declaration if AI helped with English | Report read by all four members; fresh-clone reproduction check |
| Submit | 13 Dec | | | |

**Decision gate.** Continue only if, on validation:

1. PARSeq reads accented Vietnamese and results are stable.
2. Diacritic-only errors (wrong words whose base letters are correct) make up **at least 20 %** of all wrong words (proposed threshold, may be revised before measuring).
3. Raising the resolution does not remove most diacritic errors.

If these conditions are not met, stop and switch to the fallback topic (`TODO(user)`).

**Experiment matrix.**

| ID | Question | Configuration | Compared against | Seeds |
|---|---|---|---|---|
| E0 | How well does PARSeq read Vietnamese? | PARSeq baseline: NFC labels, 228-character charset, 128×32 input | None (reference) | 42, 43, 44 |
| E0b | Does a split-sequence baseline remove the diacritic-specific gap? | Split-sequence baseline | E0 | 42, 43, 44 |
| E1 | Is any gain only from resolution? | Higher-resolution input | E0 | 42, 43, 44 |
| E2 | Do three parallel heads help? | Three heads: base letter, modifier/variant, tone | E0, E0b | 42, 43, 44 |
| E3 | Does the proposed model benefit from higher resolution? | Conditional follow-up to E1 | E1, E2 | 42, 43, 44 |
| E4 | Does factorized previous-token context help? | Optional factorized embeddings for the previous-token context | E2 without that component | 42, 43, 44 when compute allows; single-seed cells allowed and labeled |
| E5 | Does the soft syllable mask help? | Soft syllable-validity mask at decode time | E2 without that component | 42, 43, 44 when compute allows; single-seed cells allowed and labeled |
| E6 | Does dictionary post-processing help? | Dictionary built from train labels only | Same model without dictionary post-processing | 42, 43, 44 when compute allows; single-seed cells allowed and labeled |
| E7 | Does the method generalize to rare diacritic combinations? | Hold out rare combinations; both PARSeq and the proposed model, 6 runs | E0 vs E2 | 42, 43, 44 for each model |

**Priority rule.** P0 = E0, E0b, E2; P1 = E1, E7, E3 (conditional); P2 = E4, E5, E6. If compute or time runs short, reduce the number of configurations and say which were not run; never go below three seeds for a main configuration.

## Evaluation protocol (frozen)

- **Base letters** of a word: NFD, strip combining marks, map `đ` → `d`.
- Report **DER**, **UDER**, **DER on the intersection set**, word accuracy, and **1-NED**.
- Error categories are mutually exclusive: missing tone, wrong tone, extra tone, wrong modifier, `đ`/`d` confusion, combined.
- Apply a fixed priority order when assigning error categories.
- Use paired bootstrap resampling **by image**.
- Use seeds **42, 43, 44**.
- Select the checkpoint by **validation word accuracy**.
- Touch the test split **once**, after the configuration is frozen.

## Reproducibility and rules

- Every run logs its resolved config, seed, commit hash, library versions, hardware, and wall-clock time.
- `data/raw/`, `data/interim/`, `data/processed/`, and `outputs/` are ignored; `data/splits/`, `results/`, and report sources are tracked. See `.gitignore`.
- VinText data follows the stricter dataset terms: research/education only, no redistribution. Never commit images, crops, LMDB files, or ground-truth transcripts.
- PARSeq code is reused unmodified in `third_party/parseq/`; our code lives in `src/dfstr/`.
- Rebuild every report table from `results/metrics/` with the table-building command (`TODO`: add the command once the script exists).

## References

- PARSeq paper: https://arxiv.org/abs/2207.06966 and code: https://github.com/baudm/parseq
- Dictionary-guided Scene Text Recognition (VinText), CVPR 2021: https://openaccess.thecvf.com/content/CVPR2021/papers/Nguyen_Dictionary-Guided_Scene_Text_Recognition_CVPR_2021_paper.pdf
- VinText data page: https://github.com/VinAIResearch/dict-guided
