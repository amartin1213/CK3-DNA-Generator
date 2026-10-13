"""ck3dna.gene_dicts

Helpers for sizing model inputs/outputs from the gene dictionary file.

`gene_dicts.json` is derived from real Crusader Kings 3 Ruler Designer
exports (see `scripts/build_gene_dicts.py` and README.md's "Gene format and
assumptions" section). It has two top-level keys:

* ``_meta``: provenance info (sample count, which genes are raw "color"
  genes rather than named-category genes).
* ``genes``: maps each CK3 gene name (e.g. ``gene_chin_forward``) to the
  ordered list of its observed named sub-gene categories.

The three color genes (`hair_color`, `skin_color`, `eye_color`) are *not*
listed in ``genes`` -- they use four raw 0-255 integers instead of named
categories (see `ck3dna.ck3_format.COLOR_GENE_NAMES`), and are sized via
`COLOR_GENE_LENGTH` below.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from .ck3_format import COLOR_GENE_NAMES

DEFAULT_GENE_DICTS_PATH = Path(__file__).parent / "dependencies" / "gene_dicts.json"

# Every color gene is four raw 0-255 integers (see ck3_format.COLOR_GENE_NAMES).
COLOR_GENE_LENGTH = 4
# Leading gender slot (encoded as 0.0="girl"/255.0="boy", matching CK3's
# `type=` portrait field -- see ck3dna.dna_parser).
GENDER_LENGTH = 1


def load_gene_dicts(path=DEFAULT_GENE_DICTS_PATH) -> Dict[str, List[str]]:
    """Load the gene name -> ordered sub-gene category list mapping.

    Only the categorical genes are returned (see module docstring); color
    genes are handled separately since they have no named categories.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["genes"]


def get_lengths(path=DEFAULT_GENE_DICTS_PATH) -> List[int]:
    """Per-segment lengths of the flattened DNA tensor.

    Layout (shared with `ck3dna.dna_parser.dna_ToTensor` /
    `tensor_ToDna`): ``[gender, <hair_color x4>, <skin_color x4>,
    <eye_color x4>, <one segment per categorical gene in gene_dicts.json
    order>]``.

    The sum of this list is the required output dimensionality of
    `ck3dna.regressor.Regressor`.
    """
    genes = load_gene_dicts(path)
    color_lengths = [COLOR_GENE_LENGTH] * len(COLOR_GENE_NAMES)
    return [GENDER_LENGTH] + color_lengths + [len(options) for options in genes.values()]


def total_length(path=DEFAULT_GENE_DICTS_PATH) -> int:
    """Convenience wrapper: sum(get_lengths(path))."""
    return sum(get_lengths(path))
