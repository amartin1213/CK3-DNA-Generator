"""Synthetic (photo, DNA) training-pair generator using the calibrated
Blender render pipeline (``ck3_render_full.py``).

For each sample:

1. Randomize a value (Gaussian around neutral=128, clipped to 0-255) for
   every gene the render pipeline can actually visualize -- both the
   hand-calibrated "coarse" bone/rotation-driven genes (mirrors
   ``ck3_render_full.py``'s ``BONE_MAP``/``ROTATION_MAP`` keys; keep the
   ``COARSE_GENES`` list below in sync if those are extended) and every
   blendshape-driven "fine detail" gene that has a matching
   ``male_bs_<attr>_max/min.mesh`` file under ``male_head/blendshapes``
   AND a matching ``gene_bs_<attr>``/``gene_<attr>`` entry in
   ``gene_dicts.json`` (computed automatically -- see ``usable_bs_genes()``).
2. Render it (front view only is needed for training) via a background
   Blender subprocess.
3. Run the same ``ck3dna.alignment.align_face`` used for real photos on the
   render, to produce a 256x256 aligned crop in the exact training format.
4. Write a real-format CK3 DNA export for the sample by patching only the
   touched gene lines into a copy of a randomly-chosen real male CK3
   character's DNA export (``real_dna_samples/rulers/*.ck3ruler``) --
   every other gene (hair, skin tone, body shape, clothes, etc.) keeps
   that donor character's real value, so the model sees genuine
   real-world diversity for genes this pipeline can't render, instead of
   one static default repeated across every synthetic sample.

Output layout matches ``build_real_dataset.py``/``ck3dna.data_processing``:
``<output>/portraits#i/{angle#0.jpg, dna#i.txt}``.

Only the male head mesh is currently rigged for rendering (see
``ck3_render_full.py``'s ``BASE_MESH``), so every synthetic sample is male.

Usage (from the repo root)::

    python scripts/generate_synthetic_dataset.py --count 300 --output synthetic_dataset_aligned --workers 4
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ck3dna.alignment import align_and_save  # noqa: E402
from ck3dna.ck3_format import format_gene_block  # noqa: E402
from ck3dna.gene_dicts import DEFAULT_GENE_DICTS_PATH, load_gene_dicts  # noqa: E402

BLENDER_EXE = r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"
RENDER_SCRIPT = REPO_ROOT / "scripts" / "ck3_render_full.py"
DEFAULT_PREDICTOR_PATH = REPO_ROOT / "shape_predictor_68_face_landmarks.dat"
# Real CK3 Ruler Designer character exports (see scripts/build_dna_prior.py's
# docstring re: why these aren't redistributed) used as "donor" templates for
# every gene this pipeline can't render (hair/skin/eye color, body shape,
# clothes, etc.). Using ONE static default template for those genes across
# every synthetic sample (the original approach) taught the model to predict
# that same fixed default for them regardless of the input photo, which
# collapsed overall prediction diversity when retrained on synthetic-heavy
# data -- confirmed via scripts/analyze_gene_variance.py (ratio dropped
# 0.439 -> 0.125). Picking a different real character per synthetic sample
# for the non-rendered genes gives the model genuine real-world diversity to
# learn from for those genes too.
REAL_RULERS_DIR = REPO_ROOT / "real_dna_samples" / "rulers"
HEAD_DIR = Path(
    r"D:\SteamLibrary\steamapps\common\Crusader Kings III\game\gfx\models\portraits\male_head"
)
BLENDSHAPES_DIR = HEAD_DIR / "blendshapes"

# Mirrors ck3_render_full.py's BONE_MAP/ROTATION_MAP keys -- the hand-
# calibrated coarse genes driven by bone-pose translation/rotation rather
# than a blendshape file. Keep this list in sync if those maps are extended.
COARSE_GENES = [
    "chin_height", "chin_width", "chin_forward",
    "jaw_height", "jaw_width", "jaw_forward",
    "eye_depth", "eye_distance", "eye_height", "eye_angle", "eye_shut",
    "cheek_forward", "cheek_height", "cheek_width",
    "forehead_height", "forehead_width", "forehead_angle", "forehead_roundness", "forehead_brow_height",
    "head_height", "head_width", "head_profile", "head_top_height", "head_top_width",
    "mouth_corner_depth", "mouth_corner_height", "mouth_forward", "mouth_height",
    "mouth_lower_lip_size", "mouth_upper_lip_size", "mouth_open", "mouth_width",
    "neck_length", "neck_width",
    "eyebrows_fullness", "eyebrows_shape",
]


def usable_bs_genes(genes: dict) -> dict:
    """Blendshape-driven ("fine detail") genes the render pipeline can
    actually visualize: every ``gene_bs_<attr>`` in gene_dicts.json with a
    matching ``male_bs_<attr>_max.mesh``/``_min.mesh`` file on disk.
    """
    files = {p.name for p in BLENDSHAPES_DIR.iterdir()} if BLENDSHAPES_DIR.exists() else set()
    usable = {}
    for gene_key, options in genes.items():
        if not gene_key.startswith("gene_bs_"):
            continue
        attr = gene_key[len("gene_bs_"):]
        if f"male_bs_{attr}_max.mesh" in files or f"male_bs_{attr}_min.mesh" in files:
            usable[attr] = (gene_key, options)
    return usable


def build_gene_index(genes: dict) -> dict:
    """attr name -> (gene_key, options) for every gene this pipeline can
    both render and write a DNA label for."""
    index = dict(usable_bs_genes(genes))
    for attr in COARSE_GENES:
        if attr in index:
            continue  # already resolved via its gene_bs_<attr> entry above
        gene_key = f"gene_{attr}"
        options = genes.get(gene_key)
        if options is not None:
            index[attr] = (gene_key, options)
        else:
            print(f"WARN: no gene_dicts entry for coarse gene '{attr}' -- skipped")
    return index


def sample_value(rng: random.Random) -> int:
    """Gaussian around neutral (128), clipped to the valid 0-255 range --
    keeps most samples face-like while still covering the extremes."""
    return max(0, min(255, int(round(rng.gauss(128, 55)))))


def encode_attr_value(options, value: int):
    """Convert a 0-255 slider-style value into a (category, weight) DNA
    pair, matching the real game's encoding seen in real_dna_samples/
    (e.g. ``gene_chin_height={ "chin_height_pos" 116 "chin_height_pos" 116 }``
    -- the dominant category repeated at its weight)."""
    value = max(0, min(255, int(round(value))))
    pos = [o for o in options if o.endswith("_pos")]
    neg = [o for o in options if o.endswith("_neg")]
    if len(options) == 2 and len(pos) == 1 and len(neg) == 1:
        if value >= 128:
            weight = round((value - 128) / 127 * 255)
            category = pos[0]
        else:
            weight = round((128 - value) / 128 * 255)
            category = neg[0]
        return category, max(0, min(255, weight))
    # Generic categorical gene (no single bipolar pos/neg pair, e.g.
    # nose_profile's 7 named shapes) -- bucket the value into one option.
    idx = min(len(options) - 1, value * len(options) // 256)
    return options[idx], 255


_LINE_GENE_RE = re.compile(r"^(\s*)(\w+)\s*=\s*\{")


def list_male_ruler_paths() -> list:
    """Real CK3 Ruler Designer exports with ``type=boy`` -- used as per-
    sample donor templates for every gene this pipeline can't render (see
    REAL_RULERS_DIR comment above). Restricted to male characters since the
    synthetic renders are male-only (see module docstring); mixing in a
    female donor's body/clothes genes under a forced ``type=boy`` could
    teach spurious gender/gene correlations.
    """
    paths = []
    for p in sorted(REAL_RULERS_DIR.glob("*.ck3ruler")):
        text = p.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"\btype\s*=\s*boy\b", text):
            paths.append(p)
    return paths


def write_dna(attrs: dict, gene_index: dict, out_path: Path, template_path: Path, gender: str = "boy") -> None:
    output_genes = {}
    for attr, value in attrs.items():
        gene_key, options = gene_index[attr]
        category, weight = encode_attr_value(options, value)
        output_genes[gene_key] = [(category, weight), (category, weight)]

    template = Path(template_path).read_text(encoding="utf-8")
    out_lines = []
    for line in template.split("\n"):
        m = _LINE_GENE_RE.match(line)
        if m and m.group(2) in output_genes:
            out_lines.append(m.group(1) + format_gene_block(m.group(2), output_genes[m.group(2)]))
        else:
            out_lines.append(line)
    text = "\n".join(out_lines)
    text = text.replace("type=boy", f"type={gender}", 1).replace("type=girl", f"type={gender}", 1)
    out_path.write_text(text, encoding="utf-8")


def render_one(attrs: dict, tmp_dir: Path, index: int) -> Path:
    """Renders `attrs` via a background Blender subprocess, returns the
    front-view PNG path (raises on failure)."""
    attr_file = tmp_dir / f"attrs_{index}.json"
    attr_file.write_text(json.dumps(attrs), encoding="utf-8")
    out_prefix = tmp_dir / f"render_{index}"
    result = subprocess.run(
        [
            BLENDER_EXE, "--background", "--python", str(RENDER_SCRIPT),
            "--", str(attr_file), str(out_prefix),
        ],
        capture_output=True, text=True, timeout=180,
    )
    front_png = Path(f"{out_prefix}_front.png")
    if not front_png.exists():
        raise RuntimeError(f"render failed for sample {index}: {result.stdout[-2000:]}\n{result.stderr[-2000:]}")
    return front_png


def generate_one(
    index: int, gene_index: dict, predictor_path: str, output_dir: Path, rng_seed: int, donor_paths: list
) -> str:
    rng = random.Random(rng_seed)
    attrs = {attr: sample_value(rng) for attr in gene_index}
    template_path = rng.choice(donor_paths)

    with tempfile.TemporaryDirectory(prefix=f"ck3synth_{index}_") as tmp:
        tmp_dir = Path(tmp)
        try:
            front_png = render_one(attrs, tmp_dir, index)
        except Exception as exc:  # noqa: BLE001 - surface render failures per-sample
            return f"[fail] sample {index}: {exc}"

        sample_dir = output_dir / f"portraits#{index}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        ok = align_and_save(front_png, predictor_path, sample_dir / "angle#0.jpg", output_size=256)
        if not ok:
            shutil.rmtree(sample_dir, ignore_errors=True)
            return f"[skip] sample {index}: no face detected in render"

        write_dna(attrs, gene_index, sample_dir / f"dna#{index}.txt", template_path)
        return f"[ok]   sample {index} -> {sample_dir.name}"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=100, help="Number of synthetic samples to generate")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "synthetic_dataset_aligned")
    parser.add_argument("--predictor", default=str(DEFAULT_PREDICTOR_PATH))
    parser.add_argument("--workers", type=int, default=4, help="Concurrent Blender render processes")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clean", action="store_true", help="Remove --output before generating")
    args = parser.parse_args(argv)

    genes = load_gene_dicts(DEFAULT_GENE_DICTS_PATH)
    gene_index = build_gene_index(genes)
    print(f"Renderable+labelable genes: {len(gene_index)}")

    donor_paths = list_male_ruler_paths()
    if not donor_paths:
        raise SystemExit(f"No male .ck3ruler donor templates found under {REAL_RULERS_DIR}")
    print(f"Donor templates (real male characters): {len(donor_paths)}")

    if args.clean and args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    written = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(generate_one, i, gene_index, args.predictor, args.output, args.seed + i, donor_paths)
            for i in range(args.count)
        ]
        for fut in as_completed(futures):
            msg = fut.result()
            print(msg)
            if msg.startswith("[ok]"):
                written += 1

    print(f"Wrote {written}/{args.count} synthetic (portrait, dna) pairs to {args.output}")


if __name__ == "__main__":
    main()
