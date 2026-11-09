# INIT.md — Project initialization instructions for the AI agent

> **Read this file completely before doing anything.** Then execute Steps 1–6 in order and finish with the report in Step 7.
> This file is written for an AI coding agent. Humans on the team should read `README.md` instead.

---

## 0. Project context (read-only summary)

| Item | Value |
|---|---|
| Title | **Diacritic-Factorized Decoding for Vietnamese Scene Text Recognition** |
| Course | CO3133 Deep Learning and Applications (HCMUT), group project (Bài tập lớn), 40% of the course grade |
| Deadline | **Sunday 13 Dec 2026** |
| Team | 4 members (roles A, B, C, D — see Section 5.1 and `AGENTS.md`) |
| Task | Scene text *recognition* only: input is a cropped image of one word, output is a Vietnamese Unicode string with diacritics. Text *detection* is out of scope |
| Base model | **PARSeq** (Bautista & Atienza, ECCV 2022; repo `baudm/parseq`, Apache-2.0) |
| Idea | Replace PARSeq's single per-position softmax with **three parallel heads** (base letter, modifier/variant, tone), optional factorized embeddings for the previous-token context, and a soft syllable-validity mask at decode time |
| Dataset | **VinText** (CVPR 2021, Dictionary-guided Scene Text Recognition). 2,000 images, about 56k text instances, train/test only (no official validation split) |
| Main metric | **DER** (diacritic error rate) plus **UDER** and DER on the intersection set; word accuracy and 1-NED as guard/support metrics |
| Experiments | E0, E0b, E1, E2, E3, E4, E5, E6, E7 (see `AGENTS.md`) |
| Deliverables | English LNCS report + reproducible codebase |

The full proposal is the PDF `Đề-cương-Nhận-dạng-chữ-tiếng-Việt-trong-ảnh-với-đầu-ra-tách-dấu-v2.pdf` (kept by the team; copy it to `docs/` when available). Treat it as the source of truth for scope. If this file and the proposal disagree, **ask the user**.

---

## 1. Ground rules for this initialization session

1. **Create only what this file lists.** Folders, `.gitkeep` files, `pyproject.toml`, `.gitignore`, `README.md`, `AGENTS.md`. No Python source, no notebooks, no configs, no data.
2. **Do not run `git` commands.** The user commits to their own branches. Do not `git init`, `git add`, `git commit`, or add submodules.
3. **Do not install packages, download data, clone PARSeq, or start any job.** Those are listed as user actions in Step 7.
4. **Write all repo files in English** (code, comments, docs). Talk to the user in the language they use.
5. **Do not invent facts.** Where this file marks something `TODO(user)`, keep the placeholder in the output. Never fabricate names, GPU models, commit hashes, URLs, or results.
6. If anything here is ambiguous or conflicts with what you find on disk, **stop and ask** rather than guess.
7. Work in the repository root (the folder that contains this `INIT.md`). Do not create files outside it.

---

## 2. Step 1 — Create the folder structure

Create exactly this tree (directories only, plus the files noted). **Every directory below, including parent directories, gets a `.gitkeep`.**

```
.
├── AGENTS.md                      (Step 5)
├── INIT.md                       (this file, keep it)
├── README.md                     (Step 4)
├── pyproject.toml                (Step 2)
├── .gitignore                    (Step 3)
├── configs/
│   ├── charset/
│   └── experiment/
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── splits/
├── docs/
├── notebooks/
├── outputs/
├── reports/
│   ├── figures/
│   └── paper/
├── results/
│   ├── metrics/
│   └── predictions/
├── scripts/
│   └── slurm/
├── src/
│   └── dfstr/
│       ├── data/
│       ├── labels/
│       ├── models/
│       └── evaluation/
├── tests/
└── third_party/
```

Notes:
- **Do not create `third_party/parseq/`.** The user will add PARSeq there (a submodule or a clone fails if the path already exists). `third_party/.gitkeep` is enough.
- **Do not create `__init__.py` files.** `src/dfstr/` is a package that will be populated later; `pyproject.toml` is configured so this does not break tooling.

### 2.1 `.gitkeep` content

Each `.gitkeep` is a plain-text file that **describes the folder**, using this template (4 comment lines, then nothing else):

```
# <path>/
# What:    <what lives here>
# Purpose: <why it exists / which experiment or phase uses it>
# Git:     <tracked | contents ignored, see .gitignore>   Owner: <A|B|C|D|all>
```

Use the table below to fill the template. Keep wording close to the table; do not add files.

| Path | What | Purpose | Git | Owner |
|---|---|---|---|---|
| `configs/` | Hydra/YAML configuration for training and evaluation | Single place to define every run so any result can be reproduced from a config + seed | tracked | B, C |
| `configs/charset/` | Character-set definitions (baseline 228-char set, component vocabularies for base/modifier/tone) | Keep the charset an explicit, versioned input instead of hidden code | tracked | B, C |
| `configs/experiment/` | One YAML per experiment ID (E0, E0b, E1, E2, E3, E4, E5, E6, E7) | Make the experiment matrix of the proposal executable and diffable | tracked | B, C |
| `data/` | All datasets and derived data | Root for the data pipeline; nothing here except `splits/` is committed | partly ignored | A |
| `data/raw/` | Original VinText download, untouched (unzipped archive) | Immutable source of truth; VinText terms forbid redistribution | contents ignored | A |
| `data/interim/` | Word crops (perspective-rectified from quadrilaterals), intermediate files, audit outputs | Reproducible but regenerable data between raw and training format | contents ignored | A |
| `data/processed/` | Training-ready data (e.g. LMDB or label files in the format PARSeq expects) | Final inputs for training/evaluation, regenerated by scripts | contents ignored | A |
| `data/splits/` | Split manifests: image IDs per train/validation/test group, near-duplicate groups, rare-combination holdout lists for E7 | Leakage control; small, derived files that are safe to commit (IDs only, no images, no transcripts) | tracked | A |
| `docs/` | Proposal PDF, decision records, data-audit notes, PARSeq reuse notes | Keep the reasoning and frozen definitions next to the code | tracked | D |
| `notebooks/` | Exploratory notebooks (data audit, error analysis) | Exploration only; any logic that matters moves into `src/` or `scripts/` | tracked | A, D |
| `outputs/` | Hydra run directories, checkpoints, TensorBoard logs, Slurm logs | Everything a training run produces; large, regenerable | contents ignored | B |
| `reports/` | LaTeX/LNCS report sources and generated assets | Final English report deliverable | tracked | D |
| `reports/figures/` | Figures used in the report (plots, qualitative examples) | Generated from `results/` by scripts so figures are reproducible | tracked | D |
| `reports/paper/` | LNCS report source (`.tex`, `.bib`) | Report deliverable, written from week 3 in parallel with experiments | tracked | D |
| `results/` | Final numbers used in the report | Separate small, committed results from large run outputs | tracked | A, D |
| `results/metrics/` | Per-run and aggregated metrics tables (CSV/JSON): DER, UDER, DER on intersection, word accuracy, 1-NED, error categories, with seeds and CIs | Rebuild every table of the report without retraining | tracked | A |
| `results/predictions/` | Per-word prediction files keyed by crop ID | Allow error analysis and bootstrap; **must not contain ground-truth text or images** (see `AGENTS.md`) | tracked | A |
| `scripts/` | Command-line entry points (crop generation, data audit, training, evaluation, table building) | Thin wrappers around `src/dfstr/`; each documented in the README | tracked | B |
| `scripts/slurm/` | Slurm job scripts for the HPC/GPU cluster | Run the experiment matrix with fixed seeds and logged resources | tracked | B |
| `src/` | Python source root (src layout) | Keeps importable code separate from scripts and data | tracked | all |
| `src/dfstr/` | The project package (`dfstr` = diacritic-factorized STR) | Our own code; PARSeq stays untouched in `third_party/` | tracked | all |
| `src/dfstr/data/` | Crop generation, label normalization I/O, datasets/dataloaders, split and leakage utilities | Pipeline from VinText annotations to training samples | tracked | A |
| `src/dfstr/labels/` | Vietnamese character decomposition (base, modifier, tone), `đ` mapping, NFC handling, syllable-validity rules for the mask | Core linguistic logic; must be exactly invertible (compose(decompose(x)) == x) | tracked | C |
| `src/dfstr/models/` | Three-head output, factorized embeddings, loss, decoding wrappers around PARSeq | The model contribution of the project | tracked | C |
| `src/dfstr/evaluation/` | Metrics (DER, UDER, DER on intersection, word accuracy, 1-NED), error categorization, paired bootstrap | Frozen evaluation code shared by all models | tracked | A |
| `tests/` | Unit tests (decomposition round-trip, error categorization, leakage checks, metrics) | Protect the definitions the whole report depends on | tracked | A, C |
| `third_party/` | External code kept separate from ours (PARSeq goes in `third_party/parseq/`) | Clear boundary between reused code and our own, as the report requires | tracked | B |

---

## 3. Step 2 — Write `pyproject.toml`

Create `pyproject.toml` with **exactly** this content:

```toml
[project]
name = "dfstr"
version = "0.1.0"
description = "Diacritic-Factorized Decoding for Vietnamese Scene Text Recognition (CO3133 Deep Learning project)"
readme = "README.md"
requires-python = ">=3.10"
# NOTE: torch/torchvision are intentionally unpinned. The CUDA/CPU build is chosen per machine
# (see README, "Setup"). Reconcile versions with third_party/parseq/requirements/ after PARSeq is added.
dependencies = [
    "torch>=2.0",
    "torchvision",
    "pytorch-lightning>=2.0",
    "hydra-core>=1.3",
    "timm",
    "numpy",
    "pandas",
    "scipy",
    "pillow",
    "opencv-python-headless",
    "lmdb",
    "nltk",
    "imagehash",
    "rapidfuzz",
    "pyyaml",
    "tqdm",
    "tensorboard",
]

[dependency-groups]
dev = [
    "pytest",
    "ruff",
    "ipykernel",
]
data = [
    "gdown",
]
analysis = [
    "matplotlib",
    "seaborn",
    "jupyterlab",
]

[tool.uv]
# This repository is an application, not a distributable package: manage the environment only.
package = false

[tool.ruff]
line-length = 100
target-version = "py310"
extend-exclude = ["third_party", "data", "outputs"]

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

After writing it, validate that it parses (this does not install anything):

```bash
python3 -c "import tomllib; tomllib.load(open('pyproject.toml','rb')); print('pyproject.toml OK')"
```

---

## 4. Step 3 — Write `.gitignore`

Create `.gitignore` with **exactly** this content:

```gitignore
# --- Python ---
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.ruff_cache/
.mypy_cache/
.ipynb_checkpoints/
.venv/
venv/

# --- Secrets / local environment ---
.env
.env.*
!.env.example

# --- Data ---
# VinText terms: research/education only, no redistribution of any original or modified part.
# Raw data, crops, and LMDB/processed data must never be committed.
data/raw/*
!data/raw/.gitkeep
data/interim/*
!data/interim/.gitkeep
data/processed/*
!data/processed/.gitkeep
*.lmdb
*.mdb
# data/splits/ is tracked on purpose (IDs only).

# --- Training outputs and weights ---
outputs/*
!outputs/.gitkeep
lightning_logs/
tb_logs/
wandb/
*.ckpt
*.pt
*.pth
*.safetensors
slurm-*.out

# --- LaTeX build artifacts ---
*.aux
*.bbl
*.blg
*.fls
*.fdb_latexmk
*.log
*.out
*.synctex.gz
*.toc

# --- Editor / OS ---
.vscode/
.idea/
.DS_Store
Thumbs.db
```

---

## 5. Step 4 — Write `README.md`

Audience: the 4 team members. Language: **English**. Tone: practical, no marketing. Include **all** sections below, in this order. Where a value is marked `TODO(user)`, keep a visible `TODO` in the README.

### 5.1 Header and status
- Title, one-paragraph summary of the idea (use Section 0), course and deadline (13 Dec 2026).
- A short "Status" line that says: *Phase 0, not started*. The team updates it.
- Team table with columns `Role | Name | Responsibility`. Rows A, B, C, D with `TODO(user)` names and these responsibilities:
  - **A** Data and evaluation: VinText download/audit, crops, splits, leakage tests, metrics code, final test run (once).
  - **B** Baseline and training infrastructure: PARSeq setup, charset/NFC changes, pilot E0, E0b, E1, E6, GPU/Slurm scheduling.
  - **C** Proposed model: three-head output, loss, factorized context, syllable mask (E2, E3, E4, E5).
  - **D** Generalization, analysis and report: E7 design, error analysis, related work, LNCS report, README/packaging.
- Compute: `GPU: TODO(user)`, `Cluster/Slurm: TODO(user)`.

### 5.2 Repository structure
A tree of the folders from Section 2 with a one-line description each (reuse the "What" column).

### 5.3 Setup
Write these steps as copy-pasteable commands.

1. Install [`uv`](https://docs.astral.sh/uv/) and clone this repository.
2. `uv sync` (core dependencies and the `dev` group).
3. **Add PARSeq** (a maintainer does this once; the exact commit is recorded in `docs/` and in this README under "Pinned versions"):
   ```bash
   git submodule add https://github.com/baudm/parseq third_party/parseq
   git -C third_party/parseq checkout <PINNED_COMMIT>   # TODO(user): record the commit hash here
   ```
   PARSeq is Apache-2.0. Anyone cloning afterwards runs `git submodule update --init`.
4. **Install PARSeq's own requirements** following its README. At the time of writing it requires Python ≥ 3.9 and PyTorch ≥ 2.0 and installs with a platform choice (`cpu`, `cu118`, `cu121`, `rocm5.7`):
   ```bash
   cd third_party/parseq
   platform=<cpu|cu118|cu121|rocm5.7>      # TODO(user): choose for our GPU
   make torch-${platform}
   # install into the project's virtual environment (activate .venv first)
   pip install -r requirements/core.${platform}.txt -e .[train,test]
   ```
   If versions conflict with `pyproject.toml`, **do not silently change either side**: record the conflict and the resolution in `docs/` and tell the team.
5. Sanity check: `uv run pytest` (passes trivially until tests exist) and `uv run ruff check .`.

Add a short "Pinned versions" table with `TODO(user)` for PARSeq commit, Python, torch, CUDA, GPU driver.

### 5.4 Getting the dataset (VinText)
This section must be accurate and complete. Include:

- **Terms of use** (from the dataset's official page; quote briefly): by downloading, users agree to use the dataset for **research or educational purposes only**, **not to distribute** any part of it in original or modified form, and to **cite the paper** when publishing results. The official repository is listed under AGPL-3.0, but the dataset terms above are stricter; **the team follows the stricter terms**. Consequences for this repo: never commit images, crops, LMDB files, or ground-truth transcripts (this is enforced by `.gitignore` and `AGENTS.md`).
- **Official source:** the VinAI `dict-guided` repository, `https://github.com/VinAIResearch/dict-guided` (Download section). Do **not** use third-party mirrors (for example Hugging Face copies); their license/statistics may differ from the official page.
- **Which version to download:** the **Original** annotation format (`x1,y1,x2,y2,x3,y3,x4,y4,TRANSCRIPT`), because the project crops words by **perspective rectification from the quadrilaterals**. (The "Converted dataset" is COCO format; it is not needed.)
- **Commands** (Google Drive file ID taken from the official README on 9 Nov 2026; tell the reader to re-check it on the official page if the download fails):
  ```bash
  uv sync --group data
  mkdir -p data/raw
  uv run gdown 1UUQhNvzgpZy7zXBFQp0Qox-BBjunZ0ml -O data/raw/vintext_original.zip
  unzip data/raw/vintext_original.zip -d data/raw/vintext
  find data/raw/vintext -maxdepth 3 | head -50
  ```
  If `gdown` hits Google Drive quota limits, download in a browser from `https://drive.google.com/file/d/1UUQhNvzgpZy7zXBFQp0Qox-BBjunZ0ml/view?usp=sharing` and unzip into `data/raw/vintext/`.
- **Expected facts to verify after download** (a checklist, not assumptions): 2,000 images; about 56,084 text instances; train and test splits exist; **no official validation split** (if none, member A creates a deterministic image-level validation split of 300 images and records it in `data/splits/`); how unreadable regions are marked (commonly `###`); actual folder layout after unzip. The official page's folder layout description is inconsistent, so **leave a `TODO(user): paste the output of the find command and update this section`** and do not invent a layout.
- Citation block (BibTeX or plain text) for *Dictionary-guided Scene Text Recognition*, Nguyen et al., CVPR 2021.

### 5.5 Workflow by phase
A table or subsections for the phases below. For each: **goal, main tasks, outputs (with folder), owner, exit criterion**. Keep it consistent with the dates; they are the plan of record.

| Phase | Dates (2026) | Goal | Key outputs | Exit criterion |
|---|---|---|---|---|
| 0. Setup | 9 Nov → 10 Nov | Repo, environment, PARSeq added, data downloaded | working `uv` env, PARSeq pinned, raw data in `data/raw/` | everyone can run `uv run pytest` and see the raw data |
| 1. Data audit and pipeline | 9 Nov → 14 Nov | Trustworthy data and leakage-free splits | crops in `data/interim/`, split manifests in `data/splits/`, audit notes in `docs/`, label statistics (base/modifier/tone, length), `đ` and hòa/hoà convention chosen, near-duplicate groups (perceptual hash), LMDB in `data/processed/` | leakage tests pass; crop count recorded; unreadable-region rule recorded |
| 2. Pilot and decision gate | 9 Nov → 15 Nov | Is there headroom for the idea? | pilot E0 (PARSeq, NFC, charset 228, 128×32): time per run, validation curve; error taxonomy of E0 on validation | **Gate on 15 Nov** (see below) |
| 3. Core experiments (P0) | 16 Nov → 22 Nov | Main comparison | E0, E0b (split-sequence baseline), E2 (three heads), 3 seeds each, metrics in `results/metrics/` | E0, E0b, E2 complete on validation with seeds; paired bootstrap prepared |
| 4. Resolution and generalization (P1) | 23 Nov → 29 Nov | Is the gain only from resolution? Does it generalize to rare combinations? | E1 (higher resolution), E7 (hold out rare combinations; both PARSeq and proposed model, 6 runs), E3 conditional on E1 | RQ2 and RQ5 answered on validation |
| 5. Ablations and dictionary (P2) | 30 Nov → 5 Dec | Which components help | E4 (factorized context), E5 (soft syllable mask), E6 (dictionary post-processing, dictionary built from train labels only) | ablation table complete, even if some cells are single-seed |
| 6. Freeze and test | 6 Dec → 10 Dec | One clean test evaluation | frozen configs, **test run once** by member A, error analysis, failure examples | test numbers produced once; every report table rebuilt from `results/` |
| 7. Report and packaging | 23 Nov → 12 Dec (in parallel, final polish 11 Dec → 12 Dec) | English LNCS report + reproducible repo | `reports/paper/`, `reports/figures/`, README, checkpoint instructions, AI-usage declaration if AI helped with English | report read by all four members; fresh-clone reproduction check |
| Submit | 13 Dec | | | |

Then include the **decision gate** (verbatim logic from the proposal): continue only if, on validation, (1) PARSeq reads accented Vietnamese and results are stable; (2) diacritic-only errors (wrong words whose base letters are correct) make up **at least 20 %** of all wrong words (proposed threshold, may be revised before measuring); (3) raising the resolution does not remove most diacritic errors. If not met, stop and switch to the fallback topic (`TODO(user)`).

Then include the **experiment matrix** (ID, question, configuration, compared against, seeds) and the **priority rule**: P0 = E0, E0b, E2; P1 = E1, E7, E3 (conditional); P2 = E4, E5, E6. If compute or time runs short, reduce the number of configurations and say which were not run; never go below three seeds for a main configuration.

### 5.6 Evaluation protocol (frozen)
Summarize, in 8–12 lines: "base letters" of a word = NFD, strip combining marks, map `đ`→`d`; DER, UDER, DER on intersection, word accuracy, 1-NED; mutually exclusive error categories (missing tone, wrong tone, extra tone, wrong modifier, `đ`/`d` confusion, combined) with a fixed priority order; paired bootstrap resampling **by image**; seeds 42, 43, 44; checkpoint selected by validation word accuracy; test touched once.

### 5.7 Reproducibility and rules
- Every run logs config, seed, commit hash, library versions, hardware, wall-clock time.
- Which folders are tracked vs ignored; the data license rule.
- Where PARSeq code is reused (unmodified in `third_party/parseq/`) versus what is ours (`src/dfstr/`).
- How to rebuild every table from `results/metrics/` (placeholder command `TODO` until scripts exist).

### 5.8 References
List only these (all verified):
- PARSeq paper: https://arxiv.org/abs/2207.06966 and code: https://github.com/baudm/parseq
- Dictionary-guided Scene Text Recognition (VinText), CVPR 2021: https://openaccess.thecvf.com/content/CVPR2021/papers/Nguyen_Dictionary-Guided_Scene_Text_Recognition_CVPR_2021_paper.pdf
- VinText data page: https://github.com/VinAIResearch/dict-guided

---

## 6. Step 5 — Write `AGENTS.md`

`AGENTS.md` is the standing instruction file that **every future AI agent session reads first**. Language: English. Keep it under ~250 lines, imperative style, no fluff. Include these sections.

### 6.1 Project summary
Five to eight lines from Section 0, plus pointers: `README.md` (workflow and setup), `docs/` (proposal, decisions).

### 6.2 Repository map
A table `Path → purpose → owner`, reusing Section 2.1 (condensed).

### 6.3 Hard rules (never break; ask the user if a task seems to require it)
1. **No git commands.** The agent edits files; the user commits. (Code may read the commit hash for logging via a small utility.)
2. **The agent writes code; the user launches long runs.** Provide the exact command (and Slurm script when relevant); do not start training or evaluation jobs yourself unless the user explicitly asks.
3. **Data license.** Never commit, copy outside `data/`, upload, or paste into prompts/documents: VinText images, crops, LMDB files, or ground-truth transcripts. `results/predictions/` files are keyed by crop ID and contain predictions and derived error categories only, **no ground-truth text**.
4. **Test set discipline.** Do not read, evaluate on, or tune with the test split until the user declares the configuration frozen. The test run happens once, by member A.
5. **Frozen definitions.** The following change only with explicit user approval and a note in `docs/`: NFC normalization and the chosen tone-mark convention; the 228-character baseline charset; the definition of "base letters"; DER, UDER, DER-on-intersection; error-category priority order; split manifests; seeds 42/43/44; checkpoint-selection rule.
6. **Do not modify `third_party/parseq/`.** Extend by wrapping/subclassing in `src/dfstr/`. If a patch is truly unavoidable, document it in `docs/` and tell the user.
7. **No invented facts.** Never fabricate results, numbers, citations, URLs, or dataset statistics. Mark unverified claims as unverified. Measured values must come from files in `results/`.
8. **Ask before** adding a dependency, changing the split or charset, deviating from the PARSeq training protocol, or changing the experiment matrix.

### 6.4 Environment and commands
```bash
uv sync                          # create/update the environment
uv run pytest                    # tests
uv run ruff check . && uv run ruff format --check .
```
Mention the optional groups: `uv sync --group data`, `uv sync --group analysis`. State that torch is installed per machine (see README) and the GPU is `TODO(user)`.

### 6.5 Coding conventions
- Python ≥ 3.10, type hints on public functions, `ruff` clean (line length 100), docstrings on non-trivial functions.
- Our code in `src/dfstr/`, entry points in `scripts/`, configs in `configs/` (Hydra/YAML). No hard-coded paths; paths come from config or CLI arguments.
- Deterministic by default: set seeds, log them; no hidden randomness in data splitting.
- Tests are mandatory for: label decomposition round-trip (`compose(decompose(x)) == x` for every Vietnamese syllable and for `đ`, digits, ASCII symbols), error categorization (categories are exclusive and sum to the DER numerator), metrics on hand-made cases, and leakage checks (no `image_id` in two splits).
- Small commits' worth of change: one concern per task; list changed files at the end of a task.

### 6.6 Reuse versus new code
Reused from PARSeq: encoder/decoder, training loop, augmentation defaults, Hydra structure (unmodified). New in this project: label decomposition, three-head output, factorized embeddings, syllable mask, metrics, data pipeline for VinText. Keep `docs/` up to date with this split because the report must state it.

### 6.7 Experiment protocol
- Experiment IDs and meaning (E0, E0b, E1–E7), priorities P0/P1/P2, seeds, budget estimate (**estimate, not measured**: 0.5–2 GPU-hours per run, 35–45 runs).
- Each run saves: resolved config, seed, commit hash, library versions, hardware, wall-clock time, and a metrics file in `results/metrics/`.
- Main comparisons are fixed in advance (3–4 of them); all others are exploratory and labeled so in the report.

### 6.8 Working with the user
- Start a task by restating the goal in one sentence and the files you will touch; ask at most one clarifying question when blocked.
- Finish a task with: what changed (files), how it was checked (tests run), what the user must run, and open questions.
- Keep a short running list of open items in `docs/` when something is deferred.

### 6.9 Roles
A/B/C/D table (names `TODO(user)`) with the responsibilities from Section 5.1 and the rule that each member cross-reads another's code: A reads C, B reads A, C reads B, D checks report numbers against prediction files.

---

## 7. Step 6 — Verify, then Step 7 — report to the user

### Verify (run these, fix anything that fails)
```bash
# every directory has a .gitkeep, and no unexpected files exist
find . -type d -not -path './.git*' -not -name '.' | while read d; do [ -f "$d/.gitkeep" ] || echo "MISSING .gitkeep: $d"; done
find . -type f -not -path './.git/*' | sort
python3 -c "import tomllib; tomllib.load(open('pyproject.toml','rb')); print('pyproject.toml OK')"
```
Checklist:
- [ ] All 27 directories exist and each `.gitkeep` follows the template and its row in Section 2.1.
- [ ] Only these files exist outside `.gitkeep`: `INIT.md`, `README.md`, `AGENTS.md`, `pyproject.toml`, `.gitignore`.
- [ ] No `third_party/parseq/`, no `__init__.py`, no source code.
- [ ] `README.md` includes every subsection in Section 5 and keeps all `TODO(user)` markers.
- [ ] `AGENTS.md` includes every subsection in Section 6.
- [ ] Every URL and Drive file ID in the README is copied from this file, not edited.

### Report to the user
Reply with: (1) the final tree; (2) the list of files created; (3) any deviation from this file and why; (4) the **user actions** below.

### User actions after initialization (list these in your report; do not do them)
1. Fill the `TODO(user)` items: member names for roles A–D, GPU and cluster details, fallback topic.
2. Run `uv sync`, then add PARSeq as a submodule and pin its commit (README, "Setup"); choose the torch platform for the GPU and reconcile any version conflict with `pyproject.toml`.
3. Download VinText with the README commands, paste the `find` output back so the README can document the real folder layout.
4. Copy the proposal PDF into `docs/`.
5. Commit on a branch (the agent does not run git).

Open questions for the user that the agent should raise if relevant: GPU model and memory; whether the cluster uses Slurm for this course; the fallback topic; whether the repository will be private.
