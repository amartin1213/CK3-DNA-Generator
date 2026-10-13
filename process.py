"""Thin CLI wrapper mirroring the upstream project's `process.py`: aligns
every portrait image found under a directory of per-character subfolders,
in place, using dlib.

Usage::

    python process.py --directory ./portraits --predictor shape_predictor_68_face_landmarks.dat
"""
from __future__ import annotations

import argparse

from ck3dna.data_processing import torchvision_dataset_align


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default="./portraits", help="Directory of portrait subfolders to align in place")
    parser.add_argument(
        "--predictor",
        default="shape_predictor_68_face_landmarks.dat",
        help="Path to dlib's 68-landmark shape predictor (see README.md)",
    )
    parser.add_argument("--output_size", type=int, default=256)
    args = parser.parse_args()

    dataset = torchvision_dataset_align(
        directory=args.directory,
        predictor_path=args.predictor,
        output_size=args.output_size,
    )
    dataset.process_and_save()


if __name__ == "__main__":
    main()
