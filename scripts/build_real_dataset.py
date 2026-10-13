"""Assemble a trainable dataset from the real (portrait, DNA) pairs under
``real_dna_samples/``.

Three source layouts are supported and combined into one dataset:

* ``real_dna_samples/ruler_portraits/<Name>.png`` + matching
  ``real_dna_samples/rulers/<Name>.ck3ruler`` -- official in-game
  screenshots, name-matched by filename stem. These are solo, front-facing
  shots (as verified when each batch was captured).
* ``real_dna_samples/discord_pairs/pairN_portrait.png`` + matching
  ``real_dna_samples/discord_pairs/pairN_dna.txt`` -- community-shared
  (Reddit/Discord) pairs, matched by the shared ``pairN`` stem. Many of
  these portraits are composite front+profile shots (two renders side by
  side) or full Ruler Designer UI screenshots with the face(s) off-center;
  see the note on face selection below for how the profile/UI-noise half
  is excluded without any hand-tuned cropping.
* ``real_dna_samples/discord_pairs/pairN_reference_photo.{png,jpg}`` + the
  same ``pairN_dna.txt`` -- a handful of pairs additionally include a
  genuine real-life photo the user says the CK3 likeness was based on.
  These are ingested as their own extra sample (same DNA label, a second
  "photo of the same face"), since a real photograph is arguably *more*
  representative of the eventual real-world inference use case than a
  rendered in-game portrait.

Face selection note (no composite cropping needed): earlier revisions of
this script cropped composite portraits to their left ~58% width, on the
assumption the front-facing render was always on the left. That assumption
turned out to be false -- at least one sample (`pair20_portrait.png`) has
the front-facing render on the *right*. `ck3dna.alignment.align_face` now
picks the *highest-confidence* dlib detection (rather than the largest by
bounding-box area) across the whole, uncropped image, which reliably
identifies the true frontal face regardless of which side it's on, or
whether it's sitting in the middle of a wider UI screenshot with empty
space/panels around it. So every portrait (ruler screenshot, discord
composite, or reference photo) is simply handed to `align_face` as-is.

Either way, each pair is aligned with ``ck3dna.alignment.align_face`` and
written into the ``portraits#i/angle#0.jpg`` + ``portraits#i/dna#i.txt``
layout expected by ``ck3dna.data_processing`` / ``ck3dna.preprocess_tensors``
/ ``ck3dna.training`` (see their module docstrings). Note `dna_parser.py`'s
`parse_dna_file` auto-detects both the ``.ck3ruler`` and raw
``ruler_designer_<id>=`` wrapper shapes, so copying either kind of source
file to ``dna#i.txt`` works regardless of its original extension.

This is incremental and idempotent: pairs are matched purely by filename
stem, so re-running this script after more
``real_dna_samples/ruler_portraits/*.png`` screenshots or
``real_dna_samples/discord_pairs/pairN_*`` pairs are added will simply pick
up the new pairs and rebuild the full output directory from scratch -- it
is not a one-off/manual script.

Usage (from the repo root, after acquiring the dlib shape predictor -- see
README.md)::

    python scripts/build_real_dataset.py \\
        --predictor shape_predictor_68_face_landmarks.dat \\
        --output real_dataset_aligned

Then serialize to tensors for training::

    python -m ck3dna.preprocess_tensors real_dataset_aligned
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ck3dna.alignment import align_face  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
PORTRAITS_DIR = REPO_ROOT / "real_dna_samples" / "ruler_portraits"
RULERS_DIR = REPO_ROOT / "real_dna_samples" / "rulers"
DISCORD_PAIRS_DIR = REPO_ROOT / "real_dna_samples" / "discord_pairs"


def find_ruler_pairs(portraits_dir: Path = PORTRAITS_DIR, rulers_dir: Path = RULERS_DIR):
    """Returns ``[(portrait_path, ck3ruler_path), ...]`` for every portrait
    screenshot whose filename stem matches a ``.ck3ruler`` file's stem.

    Portraits without a name-matched ``.ck3ruler`` file (or vice versa) are
    silently skipped, so this stays safe to re-run as new, partial batches
    of screenshots arrive.
    """
    rulers_by_stem = {p.stem: p for p in sorted(rulers_dir.glob("*.ck3ruler"))}
    pairs = []
    for portrait in sorted(portraits_dir.glob("*.png")) + sorted(portraits_dir.glob("*.jpg")):
        ruler = rulers_by_stem.get(portrait.stem)
        if ruler is not None:
            pairs.append((portrait, ruler))
    return pairs


# Reference photos flagged by the person sourcing them as a weak/
# low-confidence likeness match (e.g. a joke/satirical image with no
# genuine resemblance to the DNA it's attached to), rather than a genuine
# "this is who I based the look on" reference. Training on these would
# teach the model an incorrect photo<->gene association, so they're
# excluded here even though their sibling `pairN_portrait.*` render is
# still included normally. pair32: a satirical "funny pope" stock photo
# with exaggerated regalia, attached to DNA with no matching headgear
# gene -- explicitly flagged as weak by the person who supplied it.
_EXCLUDED_REFERENCE_PHOTOS = {"pair32_reference_photo"}


def find_discord_pairs(discord_dir: Path = DISCORD_PAIRS_DIR):
    """Returns ``[(portrait_path, dna_txt_path), ...]`` for every
    ``pairN_portrait.{png,jpg}`` / ``pairN_dna.txt`` pair, AND every
    ``pairN_reference_photo.{png,jpg}`` / ``pairN_dna.txt`` pair (when a
    reference photo exists alongside the rendered portrait for the same
    ``pairN``), under `discord_dir`.

    A ``pairN`` with both a rendered portrait and a reference photo
    contributes two samples (same DNA label, two photos of the same face)
    -- unless the reference photo's stem is in `_EXCLUDED_REFERENCE_PHOTOS`
    (a known weak/joke pairing), in which case only the rendered portrait
    is used for that pair.

    Safe to re-run as more ``pairN_*`` files are added incrementally.
    """
    if not discord_dir.exists():
        return []
    dna_by_stem = {
        p.stem[: -len("_dna")]: p
        for p in sorted(discord_dir.glob("*_dna.txt"))
        if p.stem.endswith("_dna")
    }
    pairs = []
    for suffix in ("_portrait", "_reference_photo"):
        for ext in ("png", "jpg"):
            for photo in sorted(discord_dir.glob(f"*{suffix}.{ext}")):
                if photo.stem in _EXCLUDED_REFERENCE_PHOTOS:
                    continue
                stem = photo.stem[: -len(suffix)]
                dna_path = dna_by_stem.get(stem)
                if dna_path is not None:
                    pairs.append((photo, dna_path))
    return pairs


def build_dataset(
    output_dir,
    predictor_path: str,
    output_size: int = 256,
    portraits_dir: Path = PORTRAITS_DIR,
    rulers_dir: Path = RULERS_DIR,
    discord_dir: Path = DISCORD_PAIRS_DIR,
    include_discord: bool = True,
) -> int:
    """Aligns every matched (portrait, dna) pair -- from both the
    ruler_portraits/rulers source and (unless `include_discord` is False)
    the discord_pairs source (including any `pairN_reference_photo` files)
    -- and writes each to ``output_dir/portraits#i/{angle#0.jpg, dna#i.txt}``.

    Returns the number of pairs successfully written (face alignment can
    fail to detect a face, in which case that pair is skipped with a
    warning rather than aborting the whole run).
    """
    ruler_pairs = [(p, d, False) for p, d in find_ruler_pairs(portraits_dir, rulers_dir)]
    discord_pairs: List[Tuple[Path, Path, bool]] = []
    if include_discord:
        discord_pairs = [(p, d, True) for p, d in find_discord_pairs(discord_dir)]
    pairs = ruler_pairs + discord_pairs
    if not pairs:
        raise SystemExit(
            f"No (portrait, dna) pairs found under {portraits_dir} / {rulers_dir} "
            f"or {discord_dir}. Ruler portrait filenames must match a .ck3ruler "
            "file's stem exactly, e.g. 'Aegon I.png' <-> 'Aegon I.ck3ruler'; "
            "discord pairs must share a 'pairN' stem, e.g. 'pair1_portrait.png' "
            "<-> 'pair1_dna.txt'."
        )

    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    import cv2  # local import so a missing cv2 only breaks this path

    written = 0
    for portrait_path, dna_path, _is_discord in pairs:
        aligned = align_face(portrait_path, predictor_path, output_size=output_size)
        if aligned is None:
            print(f"  [skip] no face detected in {portrait_path.name}")
            continue

        sample_dir = output_dir / f"portraits#{written}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(sample_dir / "angle#0.jpg"), aligned)
        shutil.copy(dna_path, sample_dir / f"dna#{written}.txt")
        written += 1
        print(f"  [ok]   {portrait_path.name} -> {sample_dir.name} (from {dna_path.name})")

    print(
        f"Wrote {written}/{len(pairs)} aligned (portrait, dna) pairs to {output_dir} "
        f"({len(ruler_pairs)} ruler_portraits + {len(discord_pairs)} discord_pairs/reference_photos)"
    )
    return written


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--portraits_dir",
        type=Path,
        default=PORTRAITS_DIR,
        help="Directory of <RulerName>.png/.jpg screenshots (default: real_dna_samples/ruler_portraits)",
    )
    parser.add_argument(
        "--rulers_dir",
        type=Path,
        default=RULERS_DIR,
        help="Directory of <RulerName>.ck3ruler exports (default: real_dna_samples/rulers)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "real_dataset_aligned",
        help="Output directory for the aligned portraits#i/ layout",
    )
    parser.add_argument(
        "--predictor",
        default="shape_predictor_68_face_landmarks.dat",
        help="Path to dlib's 68-landmark shape predictor (see README.md for how to obtain it)",
    )
    parser.add_argument("--output_size", type=int, default=256, help="Alignment crop size in pixels")
    parser.add_argument(
        "--discord_dir",
        type=Path,
        default=DISCORD_PAIRS_DIR,
        help="Directory of pairN_portrait.{png,jpg}/pairN_dna.txt pairs (default: real_dna_samples/discord_pairs)",
    )
    parser.add_argument(
        "--no-discord",
        action="store_true",
        help="Exclude real_dna_samples/discord_pairs/ from the assembled dataset",
    )
    return parser


def main(argv=None) -> None:
    args = _build_arg_parser().parse_args(argv)
    build_dataset(
        output_dir=args.output,
        predictor_path=args.predictor,
        output_size=args.output_size,
        portraits_dir=args.portraits_dir,
        rulers_dir=args.rulers_dir,
        discord_dir=args.discord_dir,
        include_discord=not args.no_discord,
    )


if __name__ == "__main__":
    main()
