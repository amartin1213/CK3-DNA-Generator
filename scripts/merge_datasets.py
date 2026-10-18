"""Merge one or more synthetic Blender-rendered datasets with the real
106-sample ``real_dataset_aligned`` set into a single combined training
datastore, renumbering ``portraits#i``/``dna#i.txt`` sequentially and
dropping a ``SYNTHETIC`` marker file into every synthetic sample's folder
so ``ck3dna/training.py``'s ``--synthetic_weight`` down-weighting can find
them. The real samples always come first (indices 0..N-1) so a partial/
truncated run of this script still has a complete, usable real-only
prefix.

Usage:
    python scripts/merge_datasets.py --real real_dataset_aligned \
        --synthetic synthetic_dataset_oat --output combined_dataset_oat
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SYNTHETIC_MARKER_NAME = "SYNTHETIC"


def iter_portrait_dirs(root: Path):
    return sorted(
        (p for p in root.iterdir() if p.is_dir() and p.name.startswith("portraits#")),
        key=lambda p: int(p.name.split("#", 1)[1]),
    )


def copy_sample(src_dir: Path, dest_root: Path, index: int, mark_synthetic: bool) -> None:
    dest_dir = dest_root / f"portraits#{index}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    for src_file in src_dir.iterdir():
        name = src_file.name
        if name.startswith("dna#"):
            name = f"dna#{index}.txt"
        shutil.copy2(src_file, dest_dir / name)
    if mark_synthetic:
        (dest_dir / SYNTHETIC_MARKER_NAME).touch()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--real", type=Path, default=REPO_ROOT / "real_dataset_aligned")
    parser.add_argument("--synthetic", type=Path, nargs="+", required=True, help="One or more synthetic dataset dirs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clean", action="store_true", help="Remove --output before merging")
    args = parser.parse_args(argv)

    if args.clean and args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    index = 0
    real_dirs = iter_portrait_dirs(args.real)
    for src_dir in real_dirs:
        copy_sample(src_dir, args.output, index, mark_synthetic=False)
        index += 1
    print(f"Copied {len(real_dirs)} real samples (indices 0..{index - 1})")

    for synth_root in args.synthetic:
        synth_dirs = iter_portrait_dirs(synth_root)
        start = index
        for src_dir in synth_dirs:
            copy_sample(src_dir, args.output, index, mark_synthetic=True)
            index += 1
        print(f"Copied {len(synth_dirs)} synthetic samples from {synth_root} (indices {start}..{index - 1})")

    print(f"Wrote {index} total samples to {args.output}")


if __name__ == "__main__":
    main()
