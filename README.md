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
| `gene_dicts.json` / `default_dna.txt` | Referenced by code, but not present in the repo | Created from scratch (see [Gene format & assumptions](#gene-format-and-assumptions)) |
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
- **`ck3dna/dna_parser.py`** -- `dna_ToTensor` / `tensor_ToDna`: the
  text-DNA <-> tensor conversion (see below for the format).
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
photos plus exactly one `.txt` DNA dump (see [format](#gene-format-and-assumptions)):

```
portraits/
    aethelred/
        photo1.jpg
        photo2.png
        aethelred.txt
    blanche/
        photo.jpg
        dna.txt
```

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

Crusader Kings 3's real, in-game `dna="..."` value is a **base64-encoded
binary blob** -- there is no public specification for decoding it, and
reverse-engineering the exact binary layout was out of scope for this
rebuild. The upstream project's parser (`dna_ToTensor`/`tensor_ToDna`)
clearly expected a human-readable, regex-parseable **plain-text** dump
instead (tab-indented gene blocks with quoted sub-gene names and numeric
weights, plus special-cased `age`/`type` lines) -- so this rebuild defines
and documents its own plain-text "DNA dump" convention matching that
expectation:

```
type=male
age=30
eye_color=
{
	"blue" 200
}
hair_color=
{
	"brown" 128
}
...
```

- One line per gene category (see `ck3dna/dependencies/gene_dicts.json`
  for the full list: head/face shape, eyes, nose, mouth, chin, cheeks,
  jaw, ears, skin tone, hair color/type, etc.), each a block of one or
  more `"option_name" weight` pairs on CK3's native **0-255** weight scale.
- `type` and `age` are special-cased flat `key=value` lines (gender and
  age respectively), matching the upstream parser's description.
- `ck3dna/dependencies/default_dna.txt` is a template with a reasonable
  default option (usually `"average"`, weight 128) for every gene; `
  tensor_ToDna` substitutes predicted values into this template.

**This is a project-specific convention, not a decoder for CK3's actual
save/export format.** If you need DNA strings the game itself can read,
you will still need to translate this intermediate format into the real
base64 blob (e.g. by scripting the in-game Portrait/Ruler Designer, or by
reverse-engineering the binary format separately) -- that translation step
is intentionally out of scope here, since no authoritative public spec for
it could be found. The gene names and sub-gene option lists are themselves
reasonable approximations based on commonly documented CK3 gene categories
(ethnicity/portrait-modding community guides and the general shape of the
game's `dna_data`/portrait-modifier system), not an extraction from the
game's files.

## Testing

```bash
pip install -e ".[dev]"
pytest
```

`tests/test_dna_parser.py` and `tests/test_gene_dicts.py` verify that the
gene-dict format is internally consistent (round-trips through
`dna_ToTensor`/`tensor_ToDna`, correct tensor sizing) even without real CK3
game data available to test against.

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
