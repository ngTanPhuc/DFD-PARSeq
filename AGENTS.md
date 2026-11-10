# AGENTS.md

Standing instructions for AI agents working in this repository. Read this file first. `README.md` documents setup and the team workflow; `docs/` holds the proposal and decision records; `INIT.md` is the initialization contract.

## 1. Project summary

- This is the CO3133 Deep Learning and Applications (HCMUT) group project **Diacritic-Factorized Decoding for Vietnamese Scene Text Recognition**, due **13 December 2026**.
- The task is scene text *recognition* only: a cropped image of one word goes in, a Vietnamese Unicode string with diacritics comes out. Text detection is out of scope.
- The base model is **PARSeq**, kept unmodified in `third_party/parseq/` (Apache-2.0).
- The contribution is three parallel heads (base letter, modifier/variant, tone), optional factorized embeddings for previous-token context, and a soft syllable-validity mask at decode time.
- The dataset is **VinText**; the main metric is **DER**, supported by UDER, DER on the intersection set, word accuracy, and 1-NED.
- Read `README.md` before setup or data work. Treat the proposal in `docs/` as the source of truth for scope.

## 2. Repository map

| Path | Purpose | Owner |
|---|---|---|
| `configs/` | Hydra/YAML configs; charset definitions and one file per experiment | B, C |
| `data/` | Dataset root; only `data/splits/` is tracked, the rest is ignored | A |
| `docs/` | Proposal, decision records, data-audit notes, PARSeq reuse notes | D |
| `notebooks/` | Exploration only; move reusable logic into `src/` or `scripts/` | A, D |
| `outputs/` | Hydra runs, checkpoints, logs; contents ignored | B |
| `reports/` | LNCS report sources and reproducible figures | D |
| `results/` | Committed metrics and prediction files used by the report | A, D |
| `scripts/` | Thin CLI entry points; `scripts/slurm/` holds cluster jobs | B |
| `src/dfstr/` | Our package: `data/`, `labels/`, `models/`, `evaluation/` | all |
| `tests/` | Unit tests for decomposition, metrics, error categories, leakage | A, C |
| `third_party/` | External code; PARSeq belongs in `third_party/parseq/` | B |

## 3. Hard rules

1. **No git commands.** The agent edits files; the user commits. Code may read the commit hash for run logging via a small utility.
2. **The agent writes code; the user launches long runs.** Provide the exact command and, when relevant, a Slurm script. Do not start training or evaluation jobs unless the user explicitly asks.
3. **Data license.** Never commit, copy outside `data/`, upload, or paste into prompts/documents: VinText images, crops, LMDB files, or ground-truth transcripts. `results/predictions/` files are keyed by crop ID and contain predictions and derived error categories only, **no ground-truth text**.
4. **Test set discipline.** Do not read, evaluate on, or tune with the test split until the user declares the configuration frozen. The test run happens once, by member A.
5. **Frozen definitions.** Change these only with explicit user approval and a note in `docs/`: NFC normalization and the chosen tone-mark convention; the 228-character baseline charset; the definition of "base letters"; DER, UDER, DER-on-intersection; error-category priority order; split manifests; seeds 42/43/44; checkpoint-selection rule.
6. **Do not modify `third_party/parseq/`.** Extend by wrapping or subclassing in `src/dfstr/`. If a patch is truly unavoidable, document it in `docs/` and tell the user.
7. **No invented facts.** Never fabricate results, numbers, citations, URLs, or dataset statistics. Mark unverified claims as unverified. Measured values must come from files in `results/`.
8. **Ask before** adding a dependency, changing the split or charset, deviating from the PARSeq training protocol, or changing the experiment matrix.

## 4. Environment and commands

```bash
uv sync                          # create/update the environment
uv sync --group data             # optional: gdown for VinText
uv sync --group analysis         # optional: matplotlib, seaborn, jupyterlab
uv run pytest                    # tests
uv run ruff check . && uv run ruff format --check .
```

- Python is ≥ 3.10. Torch is installed per machine; see `README.md` "Setup" and the "Pinned versions" table.
- Compute is **Kaggle/Colab**. Do not assume a CUDA version or a Slurm partition.
- `pyproject.toml` sets `pythonpath = ["src"]` for pytest and excludes `third_party`, `data`, and `outputs` from Ruff.

## 5. Coding conventions

- Python ≥ 3.10, type hints on public functions, `ruff` clean (line length 100), docstrings on non-trivial functions.
- Our code goes in `src/dfstr/`; entry points go in `scripts/`; configs go in `configs/` as Hydra/YAML. No hard-coded paths; take paths from config or CLI arguments.
- Be deterministic by default: set seeds, log them, and keep randomness out of data splitting.
- Tests are mandatory for:
  - label decomposition round-trip (`compose(decompose(x)) == x` for every Vietnamese syllable and for `đ`, digits, ASCII symbols);
  - error categorization (categories are exclusive and sum to the DER numerator);
  - metrics on hand-made cases;
  - leakage checks (no `image_id` in two splits).
- Make one concern per change. List changed files at the end of a task.

## 6. Reuse versus new code

- **Reused from PARSeq, unmodified:** encoder/decoder, training loop, augmentation defaults, Hydra structure.
- **New in this project:** label decomposition, three-head output, factorized embeddings, syllable mask, metrics, and the VinText data pipeline.
- Keep `docs/` up to date with this split because the report must state it.

## 7. Experiment protocol

- IDs: E0 (PARSeq baseline), E0b (split-sequence baseline), E1 (higher resolution), E2 (three heads), E3 (conditional follow-up to E1), E4 (factorized context), E5 (soft syllable mask), E6 (dictionary post-processing), E7 (rare-combination holdout).
- Priorities: P0 = E0, E0b, E2; P1 = E1, E7, E3 (conditional); P2 = E4, E5, E6.
- Seeds are 42, 43, 44. Never go below three seeds for a main configuration; single-seed ablation cells are allowed only if labeled as such.
- Budget estimate (**estimate, not measured**): 0.5–2 GPU-hours per run, 35–45 runs.
- Each run saves the resolved config, seed, commit hash, library versions, hardware, wall-clock time, and a metrics file under `results/metrics/`.
- The main comparisons are fixed in advance (3–4); all others are exploratory and must be labeled so in the report.

## 8. Working with the user

- Start a task by restating the goal in one sentence and listing the files you will touch. Ask at most one clarifying question when blocked.
- Finish a task with: what changed (files), how it was checked (tests run), what the user must run, and open questions.
- Keep a short running list of open items in `docs/` whenever something is deferred.

## 9. Roles

| Role | Name | Responsibility |
|---|---|---|
| A | Phuc | Data and evaluation: VinText download/audit, crops, splits, leakage tests, metrics code, final test run (once). |
| B | Tri | Baseline and training infrastructure: PARSeq setup, charset/NFC changes, pilot E0, E0b, E1, E6, GPU/Slurm scheduling. |
| C | Qui | Proposed model: three-head output, loss, factorized context, syllable mask (E2, E3, E4, E5). |
| D | Hien | Generalization, analysis and report: E7 design, error analysis, related work, LNCS report, README/packaging. |

Cross-reading rule: A reads C, B reads A, C reads B, and D checks report numbers against prediction files.

Compute: Kaggle/Colab. Do not assume a CUDA version or a Slurm partition.
