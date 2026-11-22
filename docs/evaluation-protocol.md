# Evaluation protocol: definitions, approval record and code

Status: definitions below are frozen; the code implementing them lives in
`src/dfstr/evaluation/`. Measured numbers belong in `results/`, never in this file. The metric
list and normalization rules were already fixed in `README.md` § "Evaluation protocol (frozen)"
and `AGENTS.local.md` §9.6; this note records the decisions that were still open when the code
was written.

## 1. Approval record

| Item | Decision | Date |
|---|---|---|
| Error-category rule (priority, category set, exclusivity) | Approved as drafted in `AGENTS.local.md` §9.6: one differing position and one differing component → that component's category, everything else → `combined`; `d`/`đ` stays its own category though the report may merge it with modifier errors | 22 Nov 2026 |
| Categories computed on tone-normalized strings, so tone placement is never a category | Approved with the rule above | 22 Nov 2026 |
| Leakage scope for this phase: split disjointness and crop-to-image mapping only | Train-only fitting checks are written next to the code they guard (rarity buckets, E6 dictionary) when that code exists; nothing to guard yet | 22 Nov 2026 |

Still open (not in this phase): bootstrap resamples and interval, word-accuracy tolerance,
minimum DER improvement, evaluation-group cutoffs, checkpoint-selection list. Those are Phase
4-5 items and are recorded in `docs/freeze.md` when the team locks them. The cross-check of this
decomposition against the model's own tables (Phase 3, step 27) waits for C's tables to exist.

## 2. Definitions as implemented

| Quantity | Definition in code |
|---|---|
| NFC | Ground truth and prediction are NFC-normalized first (`unicodedata.normalize`). Case is kept. |
| Base letters | NFD, drop combining marks, `đ`→`d`, `Đ`→`D` (`components.base_letters`). This folds the stroke away, which is why `d`/`đ` needs its own error category. |
| Base-correct word | `base_letters(pred) == base_letters(gt)`. Implies equal base length. |
| Exact word | Strict NFC equality, no tone normalization. |
| `K` | Base-correct words whose **tone-normalized** strings still differ. |
| DER | `K / A`, `A` = base-correct words. `None` when `A == 0`, never a fake zero. |
| UDER | `K / N`. |
| DER on the intersection set | One base-correct mask over every model compared; each model is then scored on exactly that subset. `der_on_intersection` returns the per-model numerators, denominators and the excluded count. |
| Word accuracy | `E / N` on strict strings. |
| 1-NED | `1 - Levenshtein(pred, gt) / max(len(pred), len(gt))` on strict strings. The longer length bounds it in `[0, 1]`. |
| Modifier / tone accuracy | Word-level rates over the base-correct words (the DER denominator): a word counts when every position's modifier (resp. tone) matches after tone normalization. A base error therefore never counts as a diacritic success. |
| Gate share | `K / (N - E)`, the share of wrong words that are diacritic-only (`diacritic_error_share`). |

Consequence to keep in mind when reading tables: because DER runs on tone-normalized strings,
`WordAcc = BaseAcc x (1 - DER)` is an identity only on strings where normalization changes
nothing. The test suite checks it that way.

## 3. Code

| Module | Contents |
|---|---|
| `src/dfstr/evaluation/components.py` | `base_letters`, NFD `units` (base, modifier, tone, unknown marks) |
| `src/dfstr/evaluation/errors.py` | `categorize`, `CATEGORIES` |
| `src/dfstr/evaluation/metrics.py` | `score_word`, `score_words`, `evaluate`, `der_on_intersection`, `diacritic_error_share`, `normalized_edit_distance` |
| `src/dfstr/data/leakage.py` | `check_split_disjointness`, `check_crop_manifest` |
| `tests/test_metrics.py`, `tests/test_error_categories.py`, `tests/test_leakage.py` | Hand-written fixtures only; no real VinText label is used |

`evaluate(predictions, references)` returns one dict with `words`, `base_correct_words`,
`exact_words`, `diacritic_errors`, `base_accuracy`, `der`, `uder`, `word_accuracy`,
`one_minus_ned`, `modifier_accuracy`, `tone_accuracy`, `error_categories`. Every rate comes with
its denominator in the same block, and `error_categories` always sums to `diacritic_errors`
(the function raises rather than write an inconsistent file).

## 4. Prediction interface for the model side (Role C)

- One decoded Unicode string per crop, keyed by `crop_id` (`<image_id>_r<region:03d>`).
- No head outputs, token ids or logits cross this boundary: the metrics read strings only.
- `tone_normalize` is applied inside the metrics; do not normalize tone placement in the
  decoder.
- A string with a combining mark before any base letter raises `ValueError` instead of being
  scored: it cannot come from the frozen 228-character charset, so it is a decoder bug worth
  seeing, not a diacritic error to bucket.
- Per-word prediction files (`results/predictions/`) have no ground-truth text, so scoring
  happens where both strings are available, and only the derived flags are written. The writer
  is a Phase 4 item.

## 5. Leakage checks

```bash
uv run python scripts/check_leakage.py --crops data/interim/crops/manifest.csv
uv run pytest tests/test_leakage.py
```

The script reads the tracked manifests only (image ids and crop geometry, no transcript) and
prints aggregate counts. Run it before every freeze and before the final test run. Near-duplicate
image hashing remains unmeasured by decision of 22 Nov 2026 (`docs/data-audit.md` §9), and the
test-versus-train label-overlap check stays blocked until the procedure in `AGENTS.local.md`
§13 item 1 is approved.
