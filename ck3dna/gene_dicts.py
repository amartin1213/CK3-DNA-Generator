"""ck3dna.gene_dicts

Helpers for sizing model inputs/outputs from the gene dictionary file.

`gene_dicts.json` maps each CK3-style gene name to the ordered list of its
possible named sub-gene options (see README.md for the documented
simplifying assumptions behind this file, since the real game format is an
undocumented base64 binary blob).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

DEFAULT_GENE_DICTS_PATH = Path(__file__).parent / "dependencies" / "gene_dicts.json"


def load_gene_dicts(path=DEFAULT_GENE_DICTS_PATH) -> Dict[str, List[str]]:
    """Load the gene name -> ordered sub-gene option list mapping."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_lengths(path=DEFAULT_GENE_DICTS_PATH) -> List[int]:
    """Per-segment lengths of the flattened DNA tensor.

    Layout (shared with `ck3dna.dna_parser.dna_ToTensor` /
    `tensor_ToDna`): a leading ``[age, gender]`` pair (length 1 each),
    followed by one segment per gene in `gene_dicts.json` (in file order),
    sized to that gene's number of sub-gene options.

    The sum of this list is the required output dimensionality of
    `ck3dna.regressor.Regressor`.
    """
    genes = load_gene_dicts(path)
    return [1, 1] + [len(options) for options in genes.values()]


def total_length(path=DEFAULT_GENE_DICTS_PATH) -> int:
    """Convenience wrapper: sum(get_lengths(path))."""
    return sum(get_lengths(path))
