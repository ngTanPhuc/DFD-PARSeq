# VinText data audit

Status: **measured** on 19 Nov 2026 against the official release zip
(`data/raw/vintext_original.zip`, 1,061,921,369 bytes, 4,007 entries). Everything below is a
count, a rate or a distribution produced by `scripts/audit_dataset.py`; no transcript text is
reproduced in this file or in the report it reads.

Reproduce with:

```bash
uv run python scripts/audit_dataset.py --from_zip data/raw/vintext_original.zip --out data/interim/audit

```
or
```bash
uv run python scripts/audit_dataset.py --from_folder data/raw/vintext/vietnamese --out data/interim/audit
```

The script reads the zip in place, so the full audit runs before the 1 GB of images is
extracted. Runtime ≈ 0.5 s, CPU only. Machine-readable output: `data/interim/audit/report.json`
(ignored by git).

## 0. Decisions taken on this audit (20 Nov 2026)

| # | Decision | Recorded in |
|---|---|---|
| 1 | Validation split = the release folder `test_image` (300 images, ids 1201–1500); test = `unseen_test_images` (500, 1501–2000); train = `train_images` (1,200, 1–1200) | `data/splits/provenance.json` |
| 2 | Regions with an all-`#` or empty transcript are **dropped everywhere** (crops, training, metrics): 13,049 of 56,084, leaving 43,035 usable | `data/splits/provenance.json`, `is_usable_transcript()` |
| 3 | Train labels use the **new** tone placement (`hoà`, `khoẻ`, `thuỷ`); validation and test labels stay as shipped; DER, UDER, intersection DER, per-component accuracy and error categories are computed on `tone_normalize()`d strings, while word accuracy and 1-NED stay strict | §5 below, `src/dfstr/labels/orthography.py` |
| 4 | Metrics are **case-sensitive**; label case is left untouched | §7 below |
| 5 | The charset is the **union of all three splits**, 228 characters, matching the proposal. Train alone has 226: `ẽ` occurs only in validation and `ỵ` only in test, so a train-only charset would make those words impossible to spell | §8 below, `configs/charset/baseline.txt` |

The NFC rule (§4) is confirmed as a no-op for this release.

## 1. Release layout (`vietnamese/`)

| Path | Content | Count |
|---|---|---|
| `train_images/` | `im0001.jpg` … `im1200.jpg` | 1,200 |
| `test_image/` | `im1201.jpg` … `im1500.jpg` | 300 |
| `unseen_test_images/` | `im1501.jpg` … `im2000.jpg` | 500 |
| `labels/` | `gt_<n>.txt`, one per image | 2,000 |
| `general_dict.txt`, `vn_dictionary.txt` | word lists shipped with the release | 2 |

Verified: image ids are `1…1200` (train), `1201…1500` (`test_image`), `1501…2000`
(`unseen_test_images`); the three folders are disjoint; every image has exactly one label file
and every label file has an image (0 orphans in either direction). Image names are zero-padded
to four digits (`im0461.jpg`, all 2,000 names are `im` + 4 digits) while label names are not
(`gt_461.txt`). The canonical id is the **image file stem**, `im0461`, so `<folder>/<id>.jpg`
opens an image with no mapping and a label is `labels/gt_<int(id[2:])>.txt`. `format_image_id()`
in `src/dfstr/data/annotations.py` is the only place the padding is written.

## 2. Region counts

| Folder | Images | Regions | Unreadable (`###`) | Empty transcript | Regions per image |
|---|---:|---:|---:|---:|---|
| `test_image` | 300 | 8,737 | 1,517 | 19 | 1–170 (mean 29.12) |
| `train_images` | 1,200 | 35,094 | 9,300 | 18 | 1–419 (mean 29.25) |
| `unseen_test_images` | 500 | 12,253 | 2,167 | 18 | 1–147 (mean 24.51) |
| **Total** | **2,000** | **56,084** | **12,984** | **55** | 1–419 |

The total of **56,084 text regions** matches the number reported in the VinText paper exactly
(previously an unverified claim; now measured). 0 images have no regions, 0 label files are
invalid UTF-8, 0 lines fail to parse.

## 3. Annotation format (verified)

One region per line: `x1,y1,x2,y2,x3,y3,x4,y4,TRANSCRIPT`, UTF-8, no BOM.

- 55,328 lines have 9 comma-separated fields, 756 have 10 — the extra field is a comma inside
  the transcript, not a new column. Parsing therefore treats the first 8 fields as coordinates
  and joins everything after them as the transcript (`src/dfstr/data/annotations.py`).
- All coordinates parse as numbers; no line has fewer than 8 fields.
- `###` marks an unreadable region: 12,984 lines.

**Unusable regions (decided: dropped).** Other all-`#` transcripts and empty transcripts exist
and must not be treated as text. All four classes are dropped from crops, training and metrics:

| Transcript | Count |
|---|---:|
| `###` | 12,984 |
| `######` | 2 |
| `##` | 4 |
| `#` | 4 |
| empty (line ends with a comma) | 55 |
| **Total dropped** | **13,049** |

`is_usable_transcript()` in `src/dfstr/data/annotations.py` is the single definition of the rule;
the audit, the split manifests, crop generation and the metrics all call it.

| Split | Regions | Usable | Dropped |
|---|---:|---:|---:|
| train | 35,094 | 25,766 | 9,328 |
| validation | 8,737 | 7,201 | 1,536 |
| test | 12,253 | 10,068 | 2,185 |
| **Total** | **56,084** | **43,035** | **13,049** |

## 4. Unicode normalization (measured)

All 43,035 usable transcripts across the three folders are already **NFC**: 0 labels are NFD,
0 are mixed, and NFC normalization changes **0** labels. No NFD-mixing problem exists in this
release, so the frozen NFC rule is confirmed as a no-op here.

Per-folder split of usable transcripts, from `report.json`:

| Split | With diacritics | Without any mark |
|---|---:|---:|
| train | 15,054 (58.4 %) | 10,712 (41.6 %) |
| validation | 4,225 (58.7 %) | 2,976 (41.3 %) |
| test | 5,935 (59.0 %) | 4,133 (41.0 %) |

"Without any mark" means the label contains no combining mark at all; it is not yet a word-type
group (see §10).

## 5. Tone-mark convention `hòa` vs `hoà` — decided: new

Measured on train only, over the 421 readable train words that contain an `oa`/`oe`/`uy` vowel
pair carrying a tone:

| Convention | Example | Words |
|---|---|---:|
| old, tone on the first vowel | `hòa`, `khỏe`, `thủy` | 50 |
| new, tone on the second vowel | `hoà`, `khoẻ`, `thuỷ` | 371 |

The release is therefore **mixed**, with the new convention dominant (88.1 %). **Decided (20 Nov
2026): the new convention is canonical.**

- Train labels are converted to the new placement when the training data is built
  (`tone_normalize()` in `src/dfstr/labels/orthography.py`); the raw release is never modified.
- Validation and test labels stay exactly as shipped.
- DER, UDER, DER on the intersection set, per-component accuracy and error categories are
  computed after `tone_normalize()` on **both** prediction and ground truth, so a correctly read
  word in the other convention is not scored as a diacritic error.
- Word accuracy and 1-NED stay **strict** (no tone normalization), so they are slightly
  pessimistic for words the labels spell in the old convention. This is deliberate: it keeps an
  upper bound that does not hide anything.
- Consequence to state in the report: `WordAcc = BaseAcc × (1 − DER)` no longer holds as an
  identity, because the two sides of that equation use different normalizations. The metric code
  tests the identity on strict strings only.

`tone_normalize()` moves the tone for every `oa`/`oe`/`uy` pair, keeps uppercase (`HÒA` → `HOÀ`),
accepts NFD input, skips the `qu` onset (`quà` is not touched), and is idempotent.

## 6. Component statistics (train only, usable words)

Derived from `train_images` only (1,200 images). Test folders are never used for statistics
that feed a decision.

- Readable words: **25,766**; characters: **94,329**; distinct base-letter words: **5,788**
  (count only — the word list itself is not written anywhere).
- Word length (characters): 1 → 1,698; 2 → 4,216; 3 → 7,868; 4 → 6,855; 5 → 2,687; 6 → 1,055;
  7 → 348; 8 → 264; 9 → 162; 10 → 184; 11 → 121; 12 → 142; 13 → 81; 14 → 34; 15 → 8; 16 → 1;
  17 → 5; 18 → 9; 19 → 4; 20 → 3; 21 → 3; 22 → 8; 23 → 3; 24 → 4; 25 → 2; 27 → 1.
- Character classes: letters 81,387; digits 9,417; punctuation 3,088; symbols 52; spaces 385.
  Roughly one character in ten is a digit, which is consistent with the sign/menu images in the
  dataset and matters for the word-type grouping decision (still open).
- Tone per character position: none 81,717 (86.63 %); `sac` 3,782; `huyen` 3,215; `nang` 3,203;
  `hoi` 1,640; `nga` 772.
- Modifier per character position: none 83,171 (88.17 %); circumflex 5,620; horn 3,240;
  `đ` 1,777; breve 521.
- Both heads are strongly imbalanced: the "no diacritic" class dominates, so the tone and
  modifier heads need this number when the loss is designed (role C).

## 7. Case distribution — decided: case-sensitive

Train usable words: upper-case 17,490 (67.9 %), title-case 2,973 (11.5 %), lower-case
2,538 (9.8 %), uncased (digits/symbols only) 2,674 (10.4 %), mixed 91 (0.4 %).

Upper case dominates, so case handling changes the numbers. **Decided (20 Nov 2026): labels keep
their case and every comparison is case-sensitive.** `HOA` and `hoa` are different words; a model
that predicts the wrong case gets the word wrong in word accuracy, exact match and 1-NED. No
case-folding is applied anywhere in the metrics.

## 8. Charset coverage — decided: 228 characters, zero uncovered

Generated by `scripts/make_charset.py` into `configs/charset/baseline.txt` (one character per
line, sorted, NFC, no comments — see the format note below). Measured on the usable labels:

| Split | Usable words | Distinct characters |
|---|---:|---:|
| train | 25,766 | 226 |
| validation | 7,201 | 214 |
| test | 10,068 | 215 |
| **union** | **43,035** | **228** |

The union is exactly 228, the baseline size the proposal states. Composition: 185 letters
(upper and lower case), 10 digits, 1 space and 32 other characters (punctuation and symbols,
including `#`).

**Why not train-only (226).** `ẽ` appears only in validation (2 occurrences) and `ỵ` only in
test (3 occurrences). A charset built from train alone cannot encode them, so those words could
never be predicted exactly however good the model is — the metric would be biased before
training starts. The union is therefore used, and train reports `ẽ` and `ỵ` as entries it never
uses.

**Coverage check.** `uv run python scripts/audit_dataset.py --from_folder
data/raw/vintext/vietnamese --charset configs/charset/baseline.txt --out data/interim/audit`
reports **0 out-of-charset occurrences in all three folders**. The check counts code points and
says nothing about word content; it was approved (22 Nov 2026) for validation and test as a
one-time integrity check, and its only permitted use is to widen the charset, never to tune.

**Format note.** The file has no comment lines and no blank lines, because `#` is itself a label
character (13 occurrences) and a space is too (465), so neither can be reserved as syntax. A
space is a line holding one space. `tests/test_charset.py` guards the committed file against an
editor that trims that line.

The 228 characters are the label alphabet only; PARSeq's special tokens sit outside it (role B's
concern).

## 9. Splits — decided and written

`data/splits/` now holds the manifests, generated by `scripts/make_splits.py`:

| Split | Release folder | Images | Usable regions | Dropped |
|---|---|---:|---:|---:|
| train | `train_images` | 1,200 | 25,766 | 9,328 |
| validation | `test_image` | 300 | 7,201 | 1,536 |
| test | `unseen_test_images` | 500 | 10,068 | 2,185 |

- The assignment is a copy of the release folders. Nothing is resampled: no hashing, no shuffle,
  no crop-level split, so all crops of an image stay in that image's split (hard rule 9).
- `data/splits/{train,validation,test}.txt` list image ids only.
  `data/splits/provenance.json` records the folder mapping, the rule, the drop rule, per-split
  counts, the date, and the SHA-256 of the audit report the counts came from.
- The archive folder is named `test_image`, but it plays the **validation** role. That naming
  mismatch is intentional and recorded in the provenance file; nothing else refers to
  "test_image" as a test set.
- Regenerate with:

  ```bash
  uv run python scripts/make_splits.py --from_zip data/raw/vintext_original.zip \
      --audit-report data/interim/audit/report.json
  ```

  Re-run it after every audit run: the provenance file stores the SHA-256 of the audit report
  the counts came from, so a stale hash means the counts were taken from an older report.

- Tests: `tests/test_splits.py` (determinism, disjointness, drop counts, provenance).

**Approved and frozen (22 Nov 2026).** Person A reviewed the audit and approved these manifests;
the splits are frozen from this date. Changing them now needs explicit approval and invalidates
every downstream run. Hashes of the approved files:

| File | SHA-256 |
|---|---|
| `data/splits/train.txt` | `b394d19315cbdc9dee6b7a1c4d03ef410dc73e996f8bfe007415ab88283a84f3` |
| `data/splits/validation.txt` | `46349e7062e58076c52d3b156638aad987bf5d85fb046818ca6b5b2584281999` |
| `data/splits/test.txt` | `0066350f558ffbc4fed08de9f1f3dbf453a49c54c6d7e478e9c536afbedaacc7` |
| `data/splits/provenance.json` | `33ebe0e369efe725653066250cb179dc82d681aa711356739e47b0b4ee1b2ab2` |
| `configs/charset/baseline.txt` | `ecee416e48700696b76aafb8b74979c02e39b6374e40f228b73cc90a6a3881f8` |


## 10. Not measured yet

| Item | Why it is missing | Unblocked by |
|---|---|---|
| Image dimensions, decode failures | images are extracted, not yet opened | Phase 2 crop step |
| Crop heights after rectification | needs crop generation | Phase 2 |
| Perceptual-hash near-duplicate pairs | **skipped by decision 22 Nov 2026** (Phase 1 step 11); limitation recorded in §9 | nothing — permanent until after the test run |
| Word-type groups (with/without diacritics, English, digits) | classification rule undecided | user decision |
| Rare `(base, modifier, tone)` combinations for E7 | rule undecided | user decision, role D |

## 11. Method and tests

- Code: `src/dfstr/data/annotations.py` (format parsing, unusable-region rule),
  `unicode_stats.py` (audit-only Unicode helpers), `vintext_source.py` (unpacked tree or in-place
  zip), `audit.py` (report), `splits.py` (manifests), `charset.py` (charset read/write,
  inventory, coverage).
  Shared orthography, including `tone_normalize()`: `src/dfstr/labels/orthography.py`.
  Entry points: `scripts/audit_dataset.py`, `scripts/make_splits.py`, `scripts/make_charset.py`.
- The report contains counts, rates and distributions only; a test asserts that no fixture
  transcript appears anywhere in the serialized report.
- Tests: `tests/test_annotations.py`, `tests/test_unicode_stats.py`, `tests/test_audit.py`,
  `tests/test_orthography.py`, `tests/test_splits.py`, `tests/test_charset.py` (81 tests,
  synthetic hand-written fixtures only; one test reads the committed charset file), `uv run
  pytest`.
- The audit's Unicode helpers are measurement-only. `src/dfstr/labels/orthography.py` holds the
  shared tables and the tone-placement rule; the metrics module must import `tone_normalize()`
  from there instead of re-implementing it.
