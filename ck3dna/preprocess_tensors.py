"""ck3dna.preprocess_tensors

New preprocessing step (missing upstream): converts the aligned
``portraits#i/angle#ii.jpg`` images produced by `ck3dna.data_processing`
into serialized `.pt` tensors, so `ck3dna.training`'s
`torch.load`-based dataset loader has something to load. The original
project's `utils/trainning.py` assumed this preprocessing had already
happened, but never actually performed it.

Usage::

    python -m ck3dna.preprocess_tensors ./processed_portraits
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

# Matches the transform used at inference time in `ck3dna.infer`.
DEFAULT_TRANSFORM = transforms.Compose([transforms.ToTensor()])


def preprocess_directory(directory, transform=DEFAULT_TRANSFORM, remove_source_images: bool = False) -> int:
    """Serialize every ``angle#*.jpg`` portrait image under `directory` to a
    sibling `.pt` tensor file (e.g. ``angle#0.jpg`` -> ``angle#0.pt``).

    Returns the number of images converted.
    """
    directory = Path(directory)
    count = 0
    for image_path in sorted(directory.rglob("angle#*.jpg")):
        with Image.open(image_path) as img:
            tensor = transform(img.convert("RGB"))
        torch.save(tensor, image_path.with_suffix(".pt"))
        if remove_source_images:
            image_path.unlink()
        count += 1
    return count


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        help="Directory produced by ck3dna.data_processing (contains portraits#i/ subfolders)",
    )
    parser.add_argument(
        "--remove-source-images",
        action="store_true",
        help="Delete the original .jpg after serializing it to .pt",
    )
    return parser


def main(argv=None) -> None:
    args = _build_arg_parser().parse_args(argv)
    n = preprocess_directory(args.directory, remove_source_images=args.remove_source_images)
    print(f"Serialized {n} image(s) to .pt tensors under {args.directory}")


if __name__ == "__main__":
    main()
