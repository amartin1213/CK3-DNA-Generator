"""ck3dna.data_processing

Dataset discovery, cleaning and face alignment for the training pipeline.

Expected raw layout -- one subdirectory per character, each containing one
or more portrait images plus a single CK3 DNA text dump::

    portraits/
        some_character/
            photo1.jpg
            photo2.png
            character.txt   # CK3 DNA dump, see ck3dna.dna_parser
        another_character/
            photo.jpg
            dna.txt

`clean_dirs` discards subdirectories missing either an image or a `.txt`
DNA file. `get_paths` builds the ``[{"dna": path, "images": [paths, ...]}]``
list consumed by the dataset classes below.
"""
from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import cv2
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from .alignment import align_face

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def clean_dirs(directory) -> None:
    """Remove subdirectories of `directory` lacking an image or a .txt DNA file."""
    directory = Path(directory)
    for sub in sorted(p for p in directory.iterdir() if p.is_dir()):
        children = list(sub.iterdir())
        has_image = any(p.suffix.lower() in IMAGE_SUFFIXES for p in children)
        has_dna = any(p.suffix.lower() == ".txt" for p in children)
        if not (has_image and has_dna):
            shutil.rmtree(sub)


def get_paths(directory) -> List[Dict]:
    """Build ``[{"dna": Path, "images": [Path, ...]}, ...]`` for every valid subdirectory."""
    directory = Path(directory)
    entries: List[Dict] = []
    for sub in sorted(p for p in directory.iterdir() if p.is_dir()):
        children = list(sub.iterdir())
        images = sorted(p for p in children if p.suffix.lower() in IMAGE_SUFFIXES)
        dna_files = sorted(p for p in children if p.suffix.lower() == ".txt")
        if images and dna_files:
            entries.append({"dna": dna_files[0], "images": images})
    return entries


class dataset_processor(ABC):
    """Common interface for the dataset preprocessing classes below."""

    @abstractmethod
    def process_and_save(self) -> None:
        ...


@dataclass
class dataset(dataset_processor):
    """Gathers {dna, images} entries and copies them into a normalized layout.

    Output layout: ``save_directory/portraits#i/angle#ii.jpg`` +
    ``save_directory/portraits#i/dna#i.txt``.
    """

    directory: str
    save_directory: str

    def __post_init__(self):
        clean_dirs(self.directory)
        self.entries = get_paths(self.directory)

    def process_and_save(self) -> None:
        Path(self.save_directory).mkdir(parents=True, exist_ok=True)
        for i, entry in enumerate(self.entries):
            out_dir = Path(self.save_directory) / f"portraits#{i}"
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(entry["dna"], out_dir / f"dna#{i}.txt")
            for ii, image_path in enumerate(entry["images"]):
                shutil.copy(image_path, out_dir / f"angle#{ii}.jpg")


@dataclass
class dataset_align(dataset):
    """Like `dataset`, but aligns each image with dlib before saving it."""

    predictor_path: str = "./shape_predictor_68_face_landmarks.dat"
    output_size: int = 256

    def process_and_save(self) -> None:
        Path(self.save_directory).mkdir(parents=True, exist_ok=True)
        for i, entry in enumerate(self.entries):
            out_dir = Path(self.save_directory) / f"portraits#{i}"
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(entry["dna"], out_dir / f"dna#{i}.txt")
            ii = 0
            for image_path in entry["images"]:
                aligned = align_face(image_path, self.predictor_path, output_size=self.output_size)
                if aligned is None:
                    continue
                cv2.imwrite(str(out_dir / f"angle#{ii}.jpg"), aligned)
                ii += 1


@dataclass
class torchvision_dataset_align(dataset_processor):
    """Batch-aligns every image under `directory` in place, via a torchvision
    `ImageFolder` + `DataLoader` (mirrors the upstream `process.py` workflow
    of ``torchvision_dataset_align(directory='./portraits')``).

    Unlike `dataset_align`, this does not copy into a separate
    `save_directory`: it overwrites each image with its aligned version (or
    deletes it if no face was detected), directly under `directory`.
    """

    directory: str
    predictor_path: str = "./shape_predictor_68_face_landmarks.dat"
    output_size: int = 256
    batch_size: int = 1
    workers: int = 0

    def __post_init__(self):
        clean_dirs(self.directory)

    def process_and_save(self) -> None:
        image_dataset = ImageFolder(root=self.directory, loader=lambda p: p)
        loader = DataLoader(
            image_dataset,
            batch_size=self.batch_size,
            num_workers=self.workers,
            collate_fn=lambda batch: batch,
        )
        for batch in loader:
            for path, _label in batch:
                aligned = align_face(path, self.predictor_path, output_size=self.output_size)
                if aligned is None:
                    Path(path).unlink(missing_ok=True)
                    continue
                cv2.imwrite(path, aligned)
