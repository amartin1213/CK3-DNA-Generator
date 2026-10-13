"""Assemble a trainable dataset from the real (portrait, .ck3ruler) pairs
under ``real_dna_samples/``.

Pairs each ``real_dna_samples/ruler_portraits/<Name>.png`` screenshot with
its name-matched ``real_dna_samples/rulers/<Name>.ck3ruler`` DNA export,
aligns the portrait with ``ck3dna.alignment.align_face``, and writes the
result into the ``portraits#i/angle#0.jpg`` + ``portraits#i/dna#i.txt``
layout expected by ``ck3dna.data_processing`` / ``ck3dna.preprocess_tensors``
/ ``ck3dna.training`` (see their module docstrings).

This is incremental and idempotent: portraits are matched purely by
filename stem, so re-running this script after more
``real_dna_samples/ruler_portraits/*.png`` screenshots are added (same
naming convention, i.e. ``<RulerName>.png`` matching
``real_dna_samples/rulers/<RulerName>.ck3ruler``) will simply pick up the
new pairs and rebuild the full output directory from scratch -- it is not a
one-off/manual script.

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ck3dna.alignment import align_face  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
PORTRAITS_DIR = REPO_ROOT / "real_dna_samples" / "ruler_portraits"
RULERS_DIR = REPO_ROOT / "real_dna_samples" / "rulers"


def find_pairs(portraits_dir: Path = PORTRAITS_DIR, rulers_dir: Path = RULERS_DIR):
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


def build_dataset(
    output_dir,
    predictor_path: str,
    output_size: int = 256,
    portraits_dir: Path = PORTRAITS_DIR,
    rulers_dir: Path = RULERS_DIR,
) -> int:
    """Aligns every matched (portrait, .ck3ruler) pair and writes it to
    ``output_dir/portraits#i/{angle#0.jpg, dna#i.txt}``.

    Returns the number of pairs successfully written (face alignment can
    fail to detect a face, in which case that pair is skipped with a
    warning rather than aborting the whole run).
    """
    pairs = find_pairs(portraits_dir, rulers_dir)
    if not pairs:
        raise SystemExit(
            f"No (portrait, .ck3ruler) pairs found under {portraits_dir} / {rulers_dir}. "
            "Portrait filenames must match a .ck3ruler file's stem exactly, e.g. "
            "'Aegon I.png' <-> 'Aegon I.ck3ruler'."
        )

    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    written = 0
    for name_index, (portrait_path, ruler_path) in enumerate(pairs):
        aligned = align_face(portrait_path, predictor_path, output_size=output_size)
        if aligned is None:
            print(f"  [skip] no face detected in {portrait_path.name}")
            continue

        sample_dir = output_dir / f"portraits#{written}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        import cv2  # local import so a missing cv2 only breaks this path

        cv2.imwrite(str(sample_dir / "angle#0.jpg"), aligned)
        shutil.copy(ruler_path, sample_dir / f"dna#{written}.txt")
        written += 1
        print(f"  [ok]   {portrait_path.name} -> {sample_dir.name} (from {ruler_path.name})")

    print(f"Wrote {written}/{len(pairs)} aligned (portrait, dna) pairs to {output_dir}")
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
    return parser


def main(argv=None) -> None:
    args = _build_arg_parser().parse_args(argv)
    build_dataset(
        output_dir=args.output,
        predictor_path=args.predictor,
        output_size=args.output_size,
        portraits_dir=args.portraits_dir,
        rulers_dir=args.rulers_dir,
    )


if __name__ == "__main__":
    main()
