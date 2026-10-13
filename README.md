# CK3 DNA Generator

Predict a Crusader Kings 3 character's **DNA** (the face/body "genes" that
define a portrait) from an ordinary photo, with a fully trainable,
end-to-end pipeline: face alignment -> dataset preprocessing -> a
ResNet + gated-residual-regressor model -> a CK3-ready DNA text dump.

This project is a **clean rebuild** of
[Hakim1625/ck3-image-portrait-modeling](https://github.com/Hakim1625/ck3-image-portrait-modeling),
with the gaps in that project filled in. All credit for the original idea
and architecture goes to that repo; this one re-implements it from scratch,
fixes a couple of bugs, ports it to modern dependency versions, and adds
the pieces that were missing (most importantly, an inference CLI).

## What's different from the original repo

| Area | Upstream | This rebuild |
|---|---|---|
| Inference | None -- training only | New first-class `python -m ck3dna.infer` CLI: photo + checkpoint -> DNA `.txt` |
| `gene_dicts.json` / `default_dna.txt` | Referenced by code, but not present in the repo | Derived from 85 real CK3 Ruler Designer exports (see [Gene format & assumptions](#gene-format-and-assumptions)) |
| Image -> tensor preprocessing | Implied by `training.py`'s `torch.load` dataset loader, but no script produced it | New `ck3dna.preprocess_tensors` step |
| Regressor output scale | `sigmoid(...) * 265` | **Fixed to `* 255`** (CK3 gene weights are 0-255; `265` looks like a typo) -- see `ck3dna/regressor.py` |
| pytorch-lightning hooks | `training_epoch_end(outputs)` / `validation_epoch_end(outputs)` (removed in PL >=2.0) | Ported to `on_train_epoch_end` / `on_validation_epoch_end` with manually accumulated step outputs |
| Packaging | `requirements.txt` only | `pyproject.toml` (installable `pip install -e .`) + `requirements.txt` |

## Architecture

```
portrait photo
      |
      v
 ck3dna.alignment        (dlib 68-landmark face alignment -> fixed-size crop)
      |
      v
 ck3dna.data_processing   (training only: gather/clean per-character folders,
      |                    batch-align with torchvision ImageFolder)
      v
 ck3dna.preprocess_tensors (training only: serialize aligned jpgs to .pt tensors)
      |
      v
 ck3dna.model = extractor (ResNet-50/101/152) + regressor (gated residual MLP)
      |
      v
 ck3dna.dna_parser.tensor_ToDna  ->  CK3 DNA .txt file
```

- **`ck3dna/alignment.py`** -- standard dlib-based face alignment: detect
  the largest face, rotate/scale/crop so both eyes land at a canonical
  position in a fixed-size square image.
- **`ck3dna/data_processing.py`** -- dataset discovery and cleaning
  (`clean_dirs`, `get_paths`), plus three ways to align a raw dataset:
  `dataset` (copy, no alignment), `dataset_align` (copy + align),
  `torchvision_dataset_align` (align in place via `ImageFolder`).
- **`ck3dna/preprocess_tensors.py`** *(new)* -- serializes aligned
  `angle#*.jpg` images to `angle#*.pt` tensors so training doesn't have to
  decode JPEGs every epoch.
- **`ck3dna/ck3_format.py`** -- low-level, brace-balanced parser for CK3's
  real Ruler Designer export syntax (`.ck3ruler` / `ruler_designer_<id>=`
  wrappers, color vs. categorical gene blocks).
- **`ck3dna/dna_parser.py`** -- `dna_ToTensor` / `tensor_ToDna`: the
  real-format DNA <-> tensor conversion (see below for the format).
- **`ck3dna/gene_dicts.py`** -- `get_lengths()` / `total_length()`: derives
  tensor segment sizes from `dependencies/gene_dicts.json`.
- **`ck3dna/extractor.py`** -- ResNet-50/101/152 feature extractor
  (`torchvision.models`, final FC layer replaced to output a configurable
  feature vector).
- **`ck3dna/regressor.py`** -- 4 stacked gated residual blocks
  (`input_size -> 1000 (x4) -> output_size`) + final `Linear` + `Sigmoid`,
  scaled by `DNA_WEIGHT_SCALE = 255`.
- **`ck3dna/model.py`** -- combines extractor + regressor into one
  end-to-end `pytorch_lightning.LightningModule`.
- **`ck3dna/training.py`** -- `Dataset`/`LightningDataModule`/`Net`
  (MSE loss, RMSprop + `ReduceLROnPlateau`) + an `experiment` wrapper.
- **`ck3dna/parser.py`** -- training CLI argument definitions.
- **`ck3dna/infer.py`** *(new)* -- the inference entry point described below.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -e .          # or: pip install -r requirements.txt
```

### Getting the dlib face-landmark model

Face alignment needs dlib's pretrained 68-point landmark predictor. It is
**not** shipped in this repo (large binary, separate license). Download it
from the classic dlib model repository and decompress it:

```bash
curl -L -o shape_predictor_68_face_landmarks.dat.bz2 \
    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
```

Pass the resulting `shape_predictor_68_face_landmarks.dat` path via the
`--predictor` argument everywhere alignment is used.

## Dataset format (for training)

One subdirectory per character, each containing one or more portrait
photos plus exactly one DNA export -- either a raw `.txt`
`ruler_designer_<id>=` export or a genuine `.ck3ruler` Ruler Designer save
file (see [format](#gene-format-and-assumptions)):

```
portraits/
    aethelred/
        photo1.jpg
        photo2.png
        aethelred.ck3ruler
    blanche/
        photo.jpg
        dna.txt
```

See `real_dna_samples/` for 85 real example `.ck3ruler` files plus 2
genuine (photo, DNA) pairs under `real_dna_samples/discord_pairs/`
(composite front+profile screenshots -- only the front-facing half is
usable for alignment).

## Usage: training

```bash
# 1. Clean + align raw (unaligned) portraits, copying them into the
#    portraits#i/angle#ii.jpg + dna#i.txt layout
python -c "from ck3dna.data_processing import dataset_align; \
    dataset_align('./portraits', './processed').process_and_save()"

# 2. Serialize aligned images to .pt tensors
python -m ck3dna.preprocess_tensors ./processed

# 3. Train
python -m ck3dna.training --datastore_name ./processed --epochs 50 --batch_size 16
```

> **Align your raw photos exactly once.** `dataset_align` (step 1) already
> runs dlib's face alignment, so always point it at the *original,
> unaligned* `./portraits` directory. Do not also run `process.py` /
> `torchvision_dataset_align` on the same photos first (or on `./processed`
> afterwards) -- re-running face detection on an already-aligned, tightly
> cropped face is unreliable (dlib's HOG detector frequently fails to find
> a face in a crop that's mostly just the face itself) and produces no
> benefit, since the images are already aligned. `process.py` is an
> alternative, standalone in-place aligner (useful if you only want to
> align a folder of photos without the `portraits#i/dna#i.txt`
> restructuring `dataset_align` does) -- use one or the other, never both
> on the same images.

Checkpoints land under Lightning's default `lightning_logs/` /
`tb_logs/` directories; pass one back in with `--ckpt` to resume.

> **Small/real datasets and batch size.** `PortraitDnaDataModule`'s train
> `DataLoader` uses `drop_last=True` (when the training split has more
> than one batch worth of samples) to avoid a trailing batch of size 1,
> which crashes the regressor's `BatchNorm1d` layers (`"Expected more than
> 1 value per channel when training"`) -- this matters in practice once
> you're training on real, small datasets where `len(train_set)` isn't a
> clean multiple of `--batch_size`.

### Building a dataset from the real `.ck3ruler` + portrait samples

`real_dna_samples/ruler_portraits/<RulerName>.png` holds real portrait
screenshots, name-matched to their DNA under
`real_dna_samples/rulers/<RulerName>.ck3ruler` (e.g. `Aegon I.png` <->
`Aegon I.ck3ruler`). More screenshots are added incrementally over time
(same naming convention), so `scripts/build_real_dataset.py` is written to
be safely re-run as new batches arrive -- it matches purely by filename
stem and rebuilds its output directory from scratch each time:

```bash
# 1. Pair + align every <name>.png that has a matching <name>.ck3ruler,
#    writing the portraits#i/angle#0.jpg + dna#i.txt layout directly
#    (single alignment pass, no process.py / torchvision_dataset_align
#    step needed since these screenshots are raw/unaligned).
python scripts/build_real_dataset.py \
    --predictor shape_predictor_68_face_landmarks.dat \
    --output real_dataset_aligned

# 2. Serialize to tensors
python -m ck3dna.preprocess_tensors real_dataset_aligned

# 3. Train (small-dataset-friendly settings)
python -m ck3dna.training --datastore_name real_dataset_aligned \
    --epochs 30 --val_epochs 10 --batch_size 4 --val_ratio 0.2
```

`real_dataset_aligned/` is a regenerable derived artifact (git-ignored,
not committed) -- re-run the three steps above any time more portrait
screenshots land under `real_dna_samples/ruler_portraits/`.

## Usage: inference (new)

```bash
python -m ck3dna.infer \
    --image path/to/portrait.jpg \
    --checkpoint path/to/model.ckpt \
    --output result.txt \
    --predictor shape_predictor_68_face_landmarks.dat
```

This aligns the photo, runs it through the trained model, and writes a CK3
DNA text dump to `result.txt`.

## Gene format and assumptions

**This now reflects the real Crusader Kings 3 Ruler Designer export
format**, reverse-engineered from 85 genuine sample files committed under
`real_dna_samples/` (83 `.ck3ruler` save files plus 2 real (photo, DNA)
pairs from Discord screenshots) -- not an invented approximation. An
earlier version of this project used a made-up plain-text convention
before any real sample data was available; that has been fully replaced.

### Wrapper shapes

The portrait/gene data appears in one of two equivalent wrapper shapes,
both parsed transparently by `ck3dna.ck3_format.parse_dna_file`:

```
# .ck3ruler save files -- nested one level deeper, alongside other
# character/title data:
ruler={
    ...
    portrait_info={
        type=boy
        id=0
        random_seed=0
        genes={ ... }
        override={ ... }
        entity={ 0 0 }
    }
}
```

```
# Raw Ruler Designer / Discord clipboard exports -- the portrait block is
# top-level, keyed by a numeric id:
ruler_designer_3277766055={
    type=boy
    id=0
    random_seed=0
    genes={ ... }
    override={ ... }
    entity={ 0 0 }
}
```

`type=` (`boy`/`girl`) is the only gender-like field in the real DNA
format; it is encoded as the tensor's leading value (0.0 = girl, 255.0 =
boy). There is no `age` gene in portrait DNA at all (age lives elsewhere,
in save-level character data, not in the exported genes).

### Gene shapes

Within `genes={...}`, every entry is one of:

* **Color genes** -- exactly `hair_color`, `skin_color`, `eye_color` --
  are four **raw** 0-255 integers (not named categories), e.g.:

  ```
  hair_color={ 10 3 208 191 }
  ```

* **Categorical genes** (everything else -- ~199 observed genes, e.g.
  `gene_chin_forward`, `gene_bs_nose_tip_angle`, `gene_eyebrows_shape`,
  `face_detail_cheek_def`, `expression_brow_wrinkles`, a large
  `gene_dragon_*` family, etc.) have **two** `"category" value` pairs on
  CK3's native 0-255 weight scale, blending between two (possibly
  identical) named sub-categories:

  ```
  gene_chin_forward={ "chin_forward_pos" 133 "chin_forward_pos" 133 }
  gene_bs_nose_tip_angle={ "nose_tip_angle_pos" 185 "nose_tip_angle_pos" 7 }
  ```

  The two pairs are often identical (a single clear-cut category), but can
  differ (e.g. `"nose_tip_angle_pos" 185` blended with a second,
  lower-weight sample of the same or a different category) -- this is how
  CK3 interpolates blend-shapes.

### Tensor layout

`ck3dna/dependencies/gene_dicts.json` was built by parsing all 85 sample
files (`scripts/build_gene_dicts.py`) and recording, for every gene name,
every category string observed across them. **This means the category
lists reflect only what appears in these 85 samples -- they are not
guaranteed to be exhaustive of every value the game can produce**, though
with 202 total genes and 442 total categories observed, coverage should be
fairly broad. Re-run `scripts/build_gene_dicts.py` if more real samples
become available.

The flattened DNA tensor produced by `dna_ToTensor` / expected by
`tensor_ToDna` is laid out as:

```
[gender, hair_color x4, skin_color x4, eye_color x4,
 <one segment per categorical gene, in gene_dicts.json order>]
```

For a categorical gene with N observed categories, its segment is a
length-N vector: each of its two `(category, value)` pairs sets
`segment[category_index] = value` (both pairs are preserved; a pair
sharing the same category as the other just overwrites the same slot).
`tensor_ToDna` inverts this by taking the **top two** weighted categories
per gene as the predicted `(category, value)` pairs (matching the real
two-pair blend convention), and writing the three color genes back as
their four raw, clamped-to-[0, 255] integers. `ck3dna/gene_dicts.py`'s
`get_lengths()` derives the exact per-segment sizes (sum = 455 with the
current 85-sample `gene_dicts.json`), which `ck3dna.model.Model`/
`ck3dna.regressor.Regressor` use to automatically size the network's
output layer.

### `.ck3ruler` files as training labels

Since real training data primarily comes as `.ck3ruler` save files (see
`real_dna_samples/rulers/`), `ck3dna.data_processing.clean_dirs`/
`get_paths` accept `.ck3ruler` as a valid DNA-label suffix alongside
`.txt`, and `ck3dna.dna_parser.dna_ToTensor`/`ck3dna.ck3_format.parse_dna_file`
auto-detect which of the two wrapper shapes a file uses -- no separate
code path is needed for `.ck3ruler` vs. raw `.txt` exports.

### What's still an approximation

* Category coverage is bounded by the 85 real samples available (see
  above) -- a production run against the full 83-ruler dataset (once
  paired with portraits) would likely surface a handful of additional
  categories for some genes.
* The large `gene_dragon_*` family (~50+ genes, dragon-specific
  appearance) is included in full for completeness, even though it's
  irrelevant to ordinary human portraits and defaults to neutral values
  (127/0) in every sample observed -- it is simply predicted like any
  other gene. This likely wastes a little model capacity on a
  mostly-constant target, but keeps the gene set complete and matches
  the real, full set of gene names CK3 exports.

## Base64 save-game DNA format

Besides the plaintext Ruler Designer export handled above, CK3 also stores
each character's DNA as a compact **base64 string** inside save games
(`dna="Hfwd/Jlmm..."`) and this is the format most commonly pasted on
Reddit/Discord. `ck3dna/dna_base64.py` adds a best-effort
`decode_base64_dna()` / `encode_base64_dna()` pair for this format, so
such strings can be converted into the same gene dict shape the rest of
the pipeline already uses.

**Confidence level: validated against one real ground-truth sample, with
strong results.** There is no official spec for this format. The algorithm
and reference gene-order/category-index tables
(`ck3dna/dependencies/base64_gene_index.json`) were reverse-engineered by
statically analysing `AppDnaConverter.js` from the MIT-licensed
[Kyusoo/CK3_Utility](https://github.com/Kyusoo/CK3_Utility) web app (not
by us running or modifying that tool -- just reading its logic), which
appears to target CK3 around patch ~1.5:

* A base64 string decodes to a flat byte array: **4 bytes per gene**, in a
  **fixed, hardcoded gene order** (not the same order as the plaintext
  format, and not alphabetical) -- see `gene_order["NONE"]` in
  `base64_gene_index.json` for the ~104-gene vanilla order (`"EPE"` is a
  longer, ~131-gene order for a specific mod, included for completeness).
* The three color genes (`hair_color`, `skin_color`, `eye_color`) are 4
  raw integers, exactly like the plaintext format.
* Every other gene's 4 bytes are
  `[category_index_1, value_1, category_index_2, value_2]` -- the same
  two-pair blend convention as plaintext, except the category is a raw
  integer index (per-gene category list) rather than a quoted string.
  `base64_gene_index.json["category_index"]` maps each gene's category
  names to these indices, taken directly from the converter's source.

Because our own `gene_dicts.json` was built from 85 real, modern
`.ck3ruler` samples (a newer game version with more DLC content), it has
~199 genes versus this codec's ~104-131 -- around 75 genes (mostly the
`gene_dragon_*` family, plus some renamed/added complexion and
beard/hairstyle categories from patches after ~1.5) simply have no byte
slot in this format and cannot be recovered from a base64 string. This is
an expected limitation of the only known decode logic, not a bug in the
port.

**Ground-truth validation result:** a real base64 DNA string for the
"Aelyx" character (obtained via CK3's debug `portrait_editor` "Copy DNA
string" feature) was checked against its matching
`real_dna_samples/rulers/Aelyx.ck3ruler` plaintext export with
`scripts/validate_base64_dna.py`. Of its 191 genes, 87 are outside this
codec's ~1.5-era tables entirely (expected, see above). Of the **104
genes the codec does cover, 96 (92%) decoded to an exact match**,
including all three color genes matching exactly. The remaining ~8
mismatches are isolated (a few genes like `complexion`,
`face_detail_chin_def` and the eye/teeth/eyelash accessory genes have
evidently gained or reordered categories in patches after ~1.5) rather
than one systemic pattern, so they read as per-gene category-table
version drift, not a byte-order or gene-order bug in the decoder. (The
user also noted their portrait editor was "currently bugged" when
capturing this string, so a little of that residual error may be in the
captured string itself rather than the decode.) This result is pinned as
a regression test in `tests/test_dna_base64.py`
(`test_decode_matches_real_ground_truth_sample`). Re-run the validation
script yourself any time with:

```bash
python scripts/validate_base64_dna.py --base64 "Hfwd/Jlmm..." --dna-file "real_dna_samples/rulers/Some Ruler.ck3ruler"
```

This reports, per gene, whether the decoded and expected values match,
and exits non-zero on any mismatch -- useful for checking additional
ground-truth samples as they become available, or for diagnosing a
specific gene's mismatch further.

## Testing

```bash
pip install -e ".[dev]"
pytest
```

`tests/test_ck3_format.py`, `tests/test_dna_parser.py`,
`tests/test_gene_dicts.py` and `tests/test_dna_base64.py` verify that the
real-format parser, gene-dict and base64 codec are internally consistent
and round-trip correctly, including round-trip tests against genuine
`.ck3ruler` sample files under `real_dna_samples/`. (Most of the base64
tests round-trip against the reverse-engineered reference tables
themselves, but `test_decode_matches_real_ground_truth_sample` checks a
real base64 string against a real `.ck3ruler` sample -- see "Base64
save-game DNA format" above.)

## Docker

```bash
docker build -t ck3dna .
docker run --rm -it ck3dna -m ck3dna.infer --image portrait.jpg --checkpoint model.ckpt --output result.txt
```

## Attribution

This project is a from-scratch rebuild based on the architecture and ideas
in [Hakim1625/ck3-image-portrait-modeling](https://github.com/Hakim1625/ck3-image-portrait-modeling).
No code was copied; the original repo's incomplete/missing pieces (gene
dictionaries, inference CLI, tensor preprocessing step) were designed and
implemented independently here, and the `*265` regressor scaling bug and
outdated pytorch-lightning hooks were fixed. Many thanks to the original
author for the underlying idea.

The base64 save-game DNA codec (`ck3dna/dna_base64.py`,
`ck3dna/dependencies/base64_gene_index.json`) is derived from reading the
logic of the MIT-licensed
[Kyusoo/CK3_Utility](https://github.com/Kyusoo/CK3_Utility) web app's
`AppDnaConverter.js`. Its gene-order and category-index tables were
extracted from that source (not copied verbatim as code -- re-implemented
in Python from the traced logic); see that repo's `LICENSE` for the
original terms. Thanks to Kyusoo for publishing that tool's source.
