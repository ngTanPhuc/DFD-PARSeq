# Protocol freeze record

Status: **DRAFT — not frozen, not signed.** Phase 5, step 38 (`AGENTS.local.md`).

This file is the record the guarded test script (`scripts/eval_test.py`, Phase 5 step 40, not
written yet) must find before it will score the test split. It is filled in, checked against the
repository, and signed by Person A **before** any test evaluation. Until every row below is
`DONE` and the signature block is filled, the configuration is not frozen, and the test split
must not be touched.

Rules for this file:

- Every value is copied from a tracked file or a committed result file. Nothing is guessed; a
  value that cannot be checked yet is `PENDING`, never a plausible-looking placeholder.
- A change to any `DONE` row invalidates downstream validation results and needs a dated note at
  the bottom of the file, plus Person A's re-signature.
- Numbers never live here. Results belong in `results/metrics/`; this file only fixes *what* will
  be measured and *how*.

## 1. Label normalization and charset

| Item | Frozen value | Status | Source |
|---|---|---|---|
| Unicode normalization | NFC on ground truth and prediction before scoring | DONE (20 Nov 2026) | `README.md` "Evaluation protocol (frozen)", `docs/evaluation-protocol.md` §2 |
| Tone-mark placement | New placement (`hoà`, `khoẻ`, `thuỷ`) is canonical; `tone_normalize()` applied to both strings for DER, UDER, intersection DER, per-component accuracy and error categories only; word accuracy and 1-NED stay strict | DONE (20 Nov 2026) | `src/dfstr/labels/orthography.py`, `docs/data-audit.md` §5 |
| Case | Case-sensitive; no case folding anywhere | DONE (20 Nov 2026) | `docs/data-audit.md` §7 |
| Charset | 228 characters, union of the three splits, `N + 1` output classes with `[EOS]` | DONE (22 Nov 2026) | `configs/charset/baseline.txt` |
| Charset SHA-256 | `ecee416e48700696b76aafb8b74979c02e39b6374e40f228b73cc90a6a3881f8` | DONE (checked 22 Nov 2026) | `sha256sum configs/charset/baseline.txt` |
| Train label tone form | Train labels tone-normalized when the training data is built; validation and test labels stay as shipped | DONE (22 Nov 2026) | `src/dfstr/data/training_labels.py` |

## 2. Base-letter definition and metric definitions

| Item | Frozen value | Status | Source |
|---|---|---|---|
| Base letters | NFD, strip combining marks, `đ` → `d`, `Đ` → `D` | DONE (22 Nov 2026) | `src/dfstr/evaluation/components.py` |
| Main metric | DER = `K / A`, `A` = base-correct words, `K` = base-correct but not exact after tone normalization | DONE (22 Nov 2026) | `docs/evaluation-protocol.md` §2 |
| Supporting metrics | UDER = `K / N`; DER on the intersection set across the compared models; word accuracy = `E / N` (strict); 1-NED = `1 − Levenshtein / max(len)` (strict); modifier and tone accuracy over base-correct words | DONE (22 Nov 2026) | `docs/evaluation-protocol.md` §2, `src/dfstr/evaluation/metrics.py` |
| Error categories | Exclusive: missing tone, wrong tone, extra tone, wrong modifier, `d`/`đ` confusion, combined. One differing position and one differing component names the category; anything else is `combined`. Computed on tone-normalized strings. Counts sum to `K`. | DONE (22 Nov 2026) | `src/dfstr/evaluation/errors.py`, `docs/evaluation-protocol.md` §1 |
| Denominators | Every rate is reported with its denominator (`A`, `K`, `N`, `E`) next to it | DONE (22 Nov 2026) | `docs/evaluation-protocol.md` §3 |
| Gate share | `K / (N − E)`, the share of wrong words that is diacritic-only | DONE | `diacritic_error_share` in `src/dfstr/evaluation/metrics.py` |
| Metric code commit | Metrics module unchanged since the definition approval; hash recorded at freeze time | PENDING | filled by Person A at sign-off |

## 3. Splits and crop data

| Item | Frozen value | Status | Source |
|---|---|---|---|
| Split rule | Release folder assignment: `train_images` → train (1,200 images), `test_image` → validation (300), `unseen_test_images` → test (500); no reshuffle, no crop-level split | DONE (22 Nov 2026) | `data/splits/provenance.json` |
| `data/splits/train.txt` | `b394d19315cbdc9dee6b7a1c4d03ef410dc73e996f8bfe007415ab88283a84f3` | DONE (verified 22 Nov 2026) | `sha256sum` |
| `data/splits/validation.txt` | `46349e7062e58076c52d3b156638aad987bf5d85fb046818ca6b5b2584281999` | DONE (verified 22 Nov 2026) | `sha256sum` |
| `data/splits/test.txt` | `0066350f558ffbc4fed08de9f1f3dbf453a49c54c6d7e478e9c536afbedaacc7` | DONE (verified 22 Nov 2026) | `sha256sum` |
| `data/splits/provenance.json` | `33ebe0e369efe725653066250cb179dc82d681aa711356739e47b0b4ee1b2ab2` | DONE (22 Nov 2026) | `docs/data-audit.md` §9 |
| Crop data | 42,981 crops (train 25,723 / validation 7,193 / test 10,065), native-resolution PNG in `data/lmdb/{train,validation,test}`, 1-based PARSeq keys | DONE (22 Nov 2026) | `docs/crop-pipeline.md` §6 |
| Crop manifest | `data/interim/crops/manifest.csv`, geometry only, no transcript | DONE (22 Nov 2026) | `docs/crop-pipeline.md` |
| Leakage suite | `uv run pytest tests/test_leakage.py` and `scripts/check_leakage.py` green | PENDING — run immediately before sign-off | Phase 3 step 23-24 |
| Known unmeasured risk | Cross-split near-duplicate images not hashed (skipped 22 Nov 2026). Reported as a limitation, not fixed. | DONE as a recorded limitation | `docs/data-audit.md` §9-§10 |

## 4. Seeds, checkpoints and selection rule

| Item | Frozen value | Status | Source |
|---|---|---|---|
| Seeds | 42, 43, 44 for every main configuration | DONE | `README.md` "Evaluation protocol (frozen)" |
| Checkpoint-selection rule | Highest **validation word accuracy**; one checkpoint per (experiment, seed) | DONE | `README.md` "Evaluation protocol (frozen)" |
| Checkpoint list | Path or hash, seed and config hash for each (experiment, seed) pair that enters the test run | PENDING — no trained checkpoints exist yet (`outputs/` empty, `results/metrics/` empty) | Phase 3-5 |
| Config hashes | Resolved Hydra config hash per run | PENDING — `configs/experiment/` is empty | Phase 3-5 |

The checkpoint list is the part most likely to be wrong at sign-off. Fill it from the run
directories, not from memory, and confirm each entry selects on validation only.

## 5. Main comparisons and thresholds

| Item | Frozen value | Status |
|---|---|---|
| Main comparisons (3-4, fixed in writing) | Not chosen yet. Candidates from `README.md` "Experiment matrix": E2 vs E0, E2 vs E0b, E1 vs E0, E7 E0 vs E2. Person A + team must pick 3-4 and write them here before the test run. | PENDING |
| Minimum DER improvement | Not chosen yet. Must be an absolute DER difference (percentage points), with the direction and the pair it applies to. | PENDING |
| Word-accuracy tolerance | Not chosen yet. Must state the tolerance below which a word-accuracy drop counts against the method. | PENDING |
| Number of validation seeds used for the claim | 3 per main configuration | DONE by the seed rule above |
| Everything else | Labeled exploratory in the report; no test-based selection | PENDING — confirm at sign-off |

Thresholds are chosen before the test run and never revised after seeing test numbers.

## 6. Evaluation groups and cutoffs

All group definitions are derived from **train only** and fixed before final results are read
(hard rule 10). None exist yet.

| Group | Definition | Status |
|---|---|---|
| Crop height (small / medium / large) | Tertiles or fixed pixel cutoffs from the train height distribution; exact cutoffs to be written here | PENDING |
| Word type (Vietnamese with diacritics, Vietnamese without diacritics, English, digits/symbols) | Classification rule undecided (`docs/data-audit.md` §10, `AGENTS.local.md` §13 item 4) | PENDING |
| Rarity of the `(base, modifier, tone)` combination (common / medium / rare) | Train-frequency rule undecided; shared with E7 and Person D | PENDING |
| Low-sample flag | Groups below the agreed sample count are reported as low-sample, not interpreted | PENDING |

## 7. Significance testing

| Item | Frozen value | Status |
|---|---|---|
| Resampling unit | Image (cluster bootstrap), because crops from one image are correlated | DONE | `README.md` "Evaluation protocol (frozen)" |
| Resamples and interval | Not chosen. Default proposal: 1,000 resamples, 95% interval. | PENDING |
| Reported separately | Seed variability (training randomness) and bootstrap interval (test sampling) reported as separate quantities | DONE as a reporting rule |
| Bootstrap code | Not implemented yet (`src/dfstr/evaluation/` has no bootstrap module) | PENDING |

## 8. Code state and test-run guard

| Item | Frozen value | Status |
|---|---|---|
| Code commit hash | The frozen commit that the test run must use | PENDING — filled at sign-off, and the tree must be clean |
| Library versions | From `uv.lock` at the frozen commit | PENDING — recorded at sign-off |
| `scripts/eval_test.py` | Must refuse to run without this file and an explicit confirmation flag, score all main configurations and seeds in one batch through the validation code path, and write a lock file under `results/` | PENDING — not written yet |
| Pre-run smoke test | Identical code path exercised on validation and on a tiny synthetic fixture | PENDING |
| Test-split access | No test label or test metric has been read. `results/metrics/` and `results/predictions/` are empty at the time this draft was written. | DONE (22 Nov 2026) |

## 9. Explicit frozen declaration (Person A)

This section is filled by Person A, not by the agent. It is what step 39 and the guard script
check.

- Statement: _I declare the configuration frozen for the test run._
- Date: `PENDING`
- Commit hash: `PENDING`
- Split manifest hashes re-verified on that date: `PENDING`
- Checkpoint list complete: `PENDING`
- Main comparisons and thresholds in §5 complete: `PENDING`
- Signed: `PENDING`

## 10. Change log

| Date | Change | Reason | Affects |
|---|---|---|---|
| 22 Nov 2026 | File created as a draft by the agent (Phase 5 step 38). No item marked frozen that is not already frozen elsewhere in the repository. | Phase 5 preparation | nothing — no test run happened |
