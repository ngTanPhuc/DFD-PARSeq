# Crop generation and LMDB stores

Status: **implemented and unit-tested on synthetic fixtures; full dataset run pending**
(Phase 2, step 18 is the Human's run). Numbers marked *pending* below come from that run, not from
this file. Nothing here depends on test-set statistics.

## 1. What the pipeline does

`scripts/make_crops.py` turns each usable VinText region into one upright word crop and writes the
three PARSeq LMDB stores. Two passes, CPU only, deterministic:

1. **geometry** — read each source image's header for its size, rectify the quadrilateral of every
   usable region, and record the crop's size, orientation, and whether the quadrilateral had to be
   clipped. Produces the crop manifest and the drop counts.
2. **write** — decode each source image once, warp its crops at native resolution, PNG-encode them
   and stream them into `data/lmdb/<split>` in manifest order.

## 2. Crop rules (fix these before the first full run)

| Rule | Value | Why |
|---|---|---|
| Vertex ordering | corners sorted by angle around their centroid, then rotated to start at the most top-left corner and forced clockwise | independent of the order the annotation lists them in, stable under the label's arbitrary corner order |
| Output size | `width` = longest horizontal edge, `height` = longest vertical edge, rounded | the crop keeps the region's own aspect ratio; no fixed 32x128 here |
| Resolution | native, as measured on the source image | agreed with the training side: the DataLoader resizes to 32x128 (E0) or higher (E2) |
| Padding | none | no evidence that VinText quadrilaterals are systematically tight; inventing a border would also invent pixels the annotator did not mark |
| Minimum side | 2 px | below that no glyph survives rectification |
| Minimum area | 1 px² | protects against labels that collapsed to a line |
| Convexity | strictly convex required | a concave or self-intersecting label would rectify into a folded image |
| Clipping | corners outside the image are clamped to the image border, and the region stays if it still satisfies the checks above | keeps partially out-of-frame words (common on image edges) instead of dropping them |
| Vertical text (`height > width`) | kept in its upright orientation, **not** rotated 90° | agreed with the training side; also keeps the crop comparable to the label order |
| Interpolation | cubic, border replicate | smooth upscaling for small crops, no black edge pixels |
| Colour | BGR as decoded by OpenCV, PNG lossless | no resizing or re-encoding loss before training |

### Drop reasons

Usable transcripts are defined in `is_usable_transcript()` (empty and all-`#` transcripts, decided
20 Nov 2026). Rectification adds these reasons, one counter each per split:

| Reason | Meaning |
|---|---|
| `unreadable` | transcript is empty or all `#` (13,049 regions over the release, per the audit) |
| `duplicate_points` | the quadrilateral repeats a corner |
| `degenerate_area` | corners are (near-)collinear, area below 1 px² |
| `non_convex` | the four corners do not form a strictly convex quadrilateral |
| `out_of_bounds` | after clipping to the image the region has no area left |
| `too_small` | a side is shorter than 2 px |
| `bad_quad` | not four numeric coordinate pairs (would be an annotation error) |
| `missing_image` / `unreadable_label_file` | source file missing or not decodable |

Every drop is counted and printed as aggregate numbers only; no transcript text is printed, logged
or written to the manifest.

## 3. Commands

Count first, write nothing:

```bash
uv run python scripts/make_crops.py --from_folder data/raw/vintext/vietnamese --dry-run
```

Full run (Human, Phase 2 step 18):

```bash
uv run python scripts/make_crops.py --from_folder data/raw/vintext/vietnamese
```

Smoke run on a few images (cheap, writes to a scratch path):

```bash
uv run python scripts/make_crops.py --from_folder data/raw/vintext/vietnamese \
    --split validation --limit-images 5 \
    --manifest /tmp/opencode/crops/manifest.csv --lmdb /tmp/opencode/crops/lmdb
```

Useful flags: `--split {train,validation,test}` (repeatable), `--limit-images N`, `--map-size`
bytes, `--force` to replace a complete store, `--manifest` / `--lmdb` to relocate outputs.

## 4. Output layout

```
data/interim/crops/manifest.csv     # tracked? no — under data/, ignored
data/lmdb/{train,validation,test}/   # one LMDB per split, ignored
├── data.mdb
└── lock.mdb
```

LMDB keys, agreed with the training side (Phase 2 step 16):

| Key | Value |
|---|---|
| `num-samples` | decimal sample count |
| `image-%09d` (1-based) | PNG bytes of the native-resolution crop |
| `label-%09d` (1-based) | UTF-8 label string |

Label policy (`src/dfstr/data/training_labels.py`): every label is NFC and case-sensitive; **train**
labels get `tone_normalize()` (new placement, `hoà`/`khoẻ`/`thuỷ`), validation and test labels stay
as shipped. Any sub-character decomposition for the three-head model belongs in the tokenizer, not
in an extra LMDB field.

Manifest columns (`src/dfstr/data/crop_manifest.py`), one row per crop, sorted by split, image id,
region index:

| Column | Meaning |
|---|---|
| `crop_id` | `<image_id>_r<region_index:03d>`, e.g. `im0461_r003` |
| `image_id` | source image id, the image file stem |
| `split` | `train` / `validation` / `test` |
| `region_index` | 0-based index of the region inside its label file |
| `width`, `height` | native crop size in pixels |
| `orientation` | `horizontal` or `vertical` (informational; nothing is rotated) |
| `clamped` | `1` when the quadrilateral was clipped to the image border |

The manifest contains geometry only, so it stays free of ground-truth text. Labels live in the LMDB
under ignored paths and never in a tracked file.

## 5. Resumability and determinism

- The manifest and the PNG bytes are deterministic: same input, same output, byte for byte (tested).
- A split whose store already holds exactly as many samples as the manifest is skipped on a rerun.
  A store with a different count is rebuilt only when `--force` is passed; the script never
  overwrites a complete store silently.
- Each split is written in a single LMDB transaction, so a crashed run leaves no half store that
  could be mistaken for a complete one.
- Writes go only to `--manifest` and `--lmdb`; both default to ignored paths under `data/`.

## 6. Full run (22 Nov 2026)

Run by Person A with `uv run python scripts/make_crops.py --from_folder data/raw/vintext/vietnamese`.
Geometry counts (the dry run predicted these exactly) and the stores it wrote:

| Split | Images | Regions | Crops kept | Vertical | Clamped | Dropped: `unreadable` | `duplicate_points` | `non_convex` | `out_of_bounds` | `data.mdb` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 1,200 | 35,094 | 25,723 | 4,730 | 94 | 9,328 | 23 | 17 | 3 | 386 MiB |
| validation | 300 | 8,737 | 7,193 | 1,352 | 20 | 1,536 | 4 | 4 | 0 | 101 MiB |
| test | 500 | 12,253 | 10,065 | 1,468 | 28 | 2,185 | 1 | 2 | 0 | 178 MiB |
| total | 2,000 | 56,084 | 42,981 | 7,550 | 142 | 13,049 | 28 | 23 | 3 | 665 MiB |

`degenerate_area` and `too_small` never fired on this release. Native crop sizes: height
min/median/max = 2/28/1,975 px (train), width min/median/max = 2/46/3,979 px; 95th and 99th
percentiles are 263 and 526 px wide, so a small tail of very large regions carries the extremes.
All 2,000 source images opened, so no `missing_image` and no `unreadable_label_file`.

### Verification of the written stores

Checked against `data/interim/crops/manifest.csv` and the frozen split manifests:

| Check | Result |
|---|---|
| `num-samples` equals the manifest rows, per split | 25,723 / 7,193 / 10,065, exact |
| manifest rows in total | 42,981 = the crops in §6 |
| `crop_id` unique, and equal to `<image_id>_r<region:03d>` | yes, no duplicates |
| PNG decodes back to the declared `height x width` (300 random crops per split) | 900 of 900 |
| labels are non-empty, NFC | yes, 0 needing normalization |
| labels within the 228-character charset | 0 out-of-charset occurrences |
| old tone placement left in train labels | 0 (validation 55, test 14 stay as shipped, as intended) |
| crops or LMDB files in tracked paths | none (`*.mdb` is ignored, `git status` clean of data) |

## 7. Pending

- Loader-side confirmation from Tri: resize to 32x128 for E0, higher for E2; tokenizer decomposition
  for E2.
- Open item: a maximum-crop-size guard. 326 train / 72 validation / 144 test crops have a side over
  512 px, and 34 train crops exceed 1,024 px (largest 3,979x1,975). They are kept at native
  resolution for now; dropping or capping them is a data decision that has not been taken.
