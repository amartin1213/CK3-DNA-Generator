"""ck3dna.dna_parser

Parsing between CK3-style DNA text dumps and fixed-length tensors.

FORMAT NOTE: Crusader Kings 3's real on-disk/clipboard ``dna="..."`` value
is a base64-encoded binary blob -- there is no public specification for
decoding it. This project instead defines its own plain-text "DNA dump"
convention (documented in README.md) that mirrors the *shape* of CK3 gene
data: a special-cased `type` (gender) line, a special-cased `age` line, and
one block per named gene containing one or more weighted sub-gene options
on CK3's native 0-255 weight scale, e.g.::

    type=male
    age=30
    eye_color=
    {
        "blue" 200
    }

This keeps the pipeline trainable and inspectable end-to-end, while being
explicit that it is a project-specific convention rather than a decoder for
the game's actual binary format.

Tensor layout produced by `dna_ToTensor` / expected by `tensor_ToDna`:
``[age, gender, <one segment per gene in gene_dicts.json order>]`` -- see
`ck3dna.gene_dicts.get_lengths` for the exact per-segment sizes. All values
are kept on CK3's native 0-255 weight scale (age and gender included), to
match `ck3dna.regressor.Regressor`'s `sigmoid() * 255` output range.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import torch

from .gene_dicts import DEFAULT_GENE_DICTS_PATH, load_gene_dicts

DEFAULT_DNA_TEMPLATE_PATH = Path(__file__).parent / "dependencies" / "default_dna.txt"

# A gene block looks like `gene_name=\n{\n\t"option" weight\n\t...\n}`.
# This is intentionally a *flat*, single-level-brace scan (no nested
# `genes={...}` wrapper) to keep the regex-based parsing below unambiguous.
_GENE_BLOCK_RE = re.compile(r"(?m)^[ \t]*(\w+)=\s*\n?[ \t]*\{([^{}]*)\}")
_SUBGENE_RE = re.compile(r'"([^"]+)"\s*([-+]?\d+(?:\.\d+)?)')
_AGE_RE = re.compile(r"(?m)^[ \t]*age\s*=\s*(\d+(?:\.\d+)?)")
_TYPE_RE = re.compile(r'(?m)^[ \t]*type\s*=\s*"?(\w+)"?')

# Backwards-compatible alias matching the naming used in the upstream repo.
load_dict = load_gene_dicts


def dna_ToTensor(path, genes: Optional[Dict[str, List[str]]] = None) -> torch.Tensor:
    """Parse a DNA text dump at `path` into a single concatenated tensor.

    See the module docstring for the exact tensor layout.
    """
    if genes is None:
        genes = load_gene_dicts(DEFAULT_GENE_DICTS_PATH)
    text = Path(path).read_text(encoding="utf-8")

    age_match = _AGE_RE.search(text)
    age = float(age_match.group(1)) if age_match else 0.0

    type_match = _TYPE_RE.search(text)
    gender = 255.0 if type_match and type_match.group(1).lower().startswith("m") else 0.0

    blocks = {name: content for name, content in _GENE_BLOCK_RE.findall(text)}

    gene_tensors = []
    for gene_name, options in genes.items():
        vec = torch.zeros(len(options))
        block_content = blocks.get(gene_name)
        if block_content:
            for sub_name, weight in _SUBGENE_RE.findall(block_content):
                if sub_name in options:
                    vec[options.index(sub_name)] = float(weight)
        gene_tensors.append(vec)

    return torch.cat([torch.tensor([age, gender], dtype=torch.float32)] + gene_tensors)


def tensor_ToDna(
    tensor: torch.Tensor,
    path,
    genes: Optional[Dict[str, List[str]]] = None,
    template_path=DEFAULT_DNA_TEMPLATE_PATH,
) -> None:
    """Inverse of `dna_ToTensor`: write a DNA text file at `path` from `tensor`.

    For each gene, the predicted sub-gene is the argmax of its slice, and
    its weight is the (clamped-to-[0, 255]) value at that index, rounded to
    the nearest integer.
    """
    if genes is None:
        genes = load_gene_dicts(DEFAULT_GENE_DICTS_PATH)

    tensor = tensor.detach().to("cpu").flatten()
    age = int(round(max(0.0, min(255.0, float(tensor[0])))))
    gender = "male" if float(tensor[1]) >= 127.5 else "female"

    offset = 2
    predictions: Dict[str, tuple] = {}
    for gene_name, options in genes.items():
        length = len(options)
        segment = tensor[offset : offset + length]
        offset += length
        if length == 0:
            continue
        idx = int(torch.argmax(segment).item())
        weight = int(round(max(0.0, min(255.0, float(segment[idx])))))
        predictions[gene_name] = (options[idx], weight)

    template = Path(template_path).read_text(encoding="utf-8")
    template = _TYPE_RE.sub(f"type={gender}", template, count=1)
    template = _AGE_RE.sub(f"age={age}", template, count=1)

    def _replace_block(match: "re.Match[str]") -> str:
        gene_name = match.group(1)
        if gene_name in predictions:
            sub_name, weight = predictions[gene_name]
            return f'{gene_name}=\n{{\n\t"{sub_name}" {weight}\n}}'
        return match.group(0)

    template = _GENE_BLOCK_RE.sub(_replace_block, template)

    Path(path).write_text(template, encoding="utf-8")
