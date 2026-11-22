# AGENTS.md

Standing instructions for every AI agent session on this repository. Read this file first. Then read `README.md` (setup, phases) and the files in `docs/` (proposal, frozen decisions). If this file and the proposal PDF disagree, stop and ask the user; do not guess.

## 1. Project summary

- Course: CO3133 Deep Learning and Applications (HCMUT), 40% of the grade. Deadline: Sunday 13 Dec 2026.
- Task: scene text **recognition** only. Input: one-word crop. Output: Vietnamese Unicode string with diacritics. Detection is out of scope.
- Base model: PARSeq (`baudm/parseq`, Apache-2.0). Reused **unmodified** in `third_party/parseq/`.
- Idea: replace PARSeq's single per-position softmax with **three parallel heads** (base letter, modifier/variant, tone); optional factorized previous-token embeddings; optional soft syllable-validity mask at decode.
- Dataset: VinText (CVPR 2021), 2,000 images, ~56k text instances, no official validation split.
- Metrics: DER (main), UDER, DER on intersection, word accuracy, 1-NED. Frozen definitions: `README.md` § "Evaluation protocol (frozen)".
- Deliverables: English LNCS report + reproducible codebase.

## 2. Repository map

| Path | Purpose | Owner |
|---|---|---|
| `configs/charset/` | Charset definitions (228-char baseline; component vocabs) | B, C |
| `configs/experiment/` | One YAML per experiment ID (E0, E0b, E1–E7) | B, C |
| `data/raw/` | Original VinText download (ignored) | A |
| `data/interim/` | Word crops and intermediate files (ignored) | A |
| `data/processed/` | LMDB / training-ready data (ignored) | A |
| `data/splits/` | Split manifests, near-dup groups, rare-combo holdouts (tracked) | A |
| `docs/` | Proposal PDF, decision records, audit notes | D |
| `notebooks/` | Exploration only; logic moves to `src/` | A, D |
| `outputs/` | Hydra runs, checkpoints, logs (ignored) | B |
| `reports/figures/`, `reports/paper/` | Report assets and LNCS source | D |
| `results/metrics/` | Per-run and aggregated metrics (tracked) | A |
| `results/predictions/` | Per-crop predictions (tracked, no GT) | A |
| `scripts/`, `scripts/slurm/` | CLI entry points and Slurm jobs | B |
| `src/dfstr/data/` | Crop gen, label I/O, datasets, split utilities | A |
| `src/dfstr/labels/` | VN decomposition, `đ` mapping, syllable rules | C |
| `src/dfstr/models/` | Three heads, factorized embeddings, loss, decoding | C |
| `src/dfstr/evaluation/` | Metrics, error categories, paired bootstrap | A |
| `tests/` | Unit tests for frozen definitions | A, C |
| `third_party/` | External code (PARSeq goes in `third_party/parseq/`) | B |

## 3. Hard rules (never break; ask the user if a task seems to require it)

1. **No git commands.** The agent edits files; the user commits. Code may read the commit hash for logging, nothing else.
2. **The agent writes code; the user launches runs.** Provide the exact command (and Slurm script when relevant). Do not start training or evaluation jobs unless the user explicitly asks.
3. **Data license.** Never commit, copy outside `data/`, upload, or paste into prompts/documents: VinText images, crops, LMDB files, or ground-truth transcripts. `results/predictions/` files are keyed by crop ID and contain predictions and derived error categories only — no GT text.
4. **Test set discipline.** Do not read, evaluate on, or tune with the test split until the user declares the configuration frozen. The test run happens once, by member A.
5. **Frozen definitions.** Change only with explicit user approval and a note in `docs/`: NFC normalization and the chosen tone-mark convention (`hòa` vs `hoà`); the 228-character baseline charset; the definition of "base letters"; DER, UDER, DER-on-intersection; error-category priority order; split manifests; seeds 42/43/44; checkpoint-selection rule.
6. **Do not modify `third_party/parseq/`.** Extend by wrapping/subclassing in `src/dfstr/`. If a patch is truly unavoidable, document it in `docs/` and tell the user.
7. **No invented facts.** Never fabricate results, numbers, citations, URLs, or dataset statistics. Mark unverified claims as unverified. Measured values must come from files in `results/`.
8. **Ask before** adding a dependency, changing the split or charset, deviating from the PARSeq training protocol, or changing the experiment matrix.

## 4. Environment and commands

```bash
uv sync                          # core + dev
uv sync --group data             # gdown for VinText
uv sync --group analysis         # matplotlib/seaborn/jupyterlab
uv run pytest                    # tests
uv run ruff check . && uv run ruff format --check .
```