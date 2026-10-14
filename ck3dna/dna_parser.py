"""ck3dna.dna_parser

Parsing between CK3's real Ruler Designer DNA export format and fixed
length tensors.

FORMAT: this module reads/writes the *real* CK3 Ruler Designer export
format, as found in genuine ``.ck3ruler`` save files and raw
``ruler_designer_<id>={...}`` text exports (see
`real_dna_samples/` for 85 authentic samples this was reverse-engineered
from, and `ck3dna.ck3_format` for the low-level block parser). This
supersedes an earlier, invented plain-text convention used before real
sample data was available.

Every gene in the real format is one of:

* A **color gene** (`hair_color`, `skin_color`, `eye_color`): four raw
  0-255 integers, e.g. ``hair_color={ 10 3 208 191 }``.
* A **categorical gene** (everything else, e.g. `gene_chin_forward`): two
  ``"category" value`` pairs on CK3's native 0-255 weight scale, blending
  between two (possibly identical) named sub-categories, e.g.::

    gene_chin_forward={ "chin_forward_pos" 133 "chin_forward_pos" 133 }
    gene_bs_nose_tip_angle={ "nose_tip_angle_pos" 185 "nose_tip_angle_pos" 7 }

Tensor layout produced by `dna_ToTensor` / expected by `tensor_ToDna` (see
`ck3dna.gene_dicts.get_lengths` for the exact per-segment sizes)::

    [gender, hair_color x4, skin_color x4, eye_color x4,
     <one segment per categorical gene in gene_dicts.json order>]

For a categorical gene with N observed categories, its segment is a
length-N vector: each of the gene's two (category, value) pairs sets
``segment[category_index] = value`` (the second pair overwrites the first
if they name the same category). All values are kept on CK3's native
0-255 weight scale, matching `ck3dna.regressor.Regressor`'s
``sigmoid() * 255`` output range (see that module for the *265 -> *255
scaling bugfix carried over from the original project).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import torch

from .ck3_format import COLOR_GENE_NAMES, format_gene_block, parse_dna_file
from .gene_dicts import DEFAULT_GENE_DICTS_PATH, load_gene_dicts

DEFAULT_DNA_TEMPLATE_PATH = Path(__file__).parent / "dependencies" / "default_dna.txt"

# Backwards-compatible alias matching the naming used in the upstream repo.
load_dict = load_gene_dicts


def dna_ToTensor(path, genes: Optional[Dict[str, List[str]]] = None) -> torch.Tensor:
    """Parse a real CK3 DNA export (``.ck3ruler`` or raw
    ``ruler_designer_<id>=`` ``.txt``) at `path` into a single concatenated
    tensor. See the module docstring for the exact tensor layout.

    Genes absent from the file (the real samples are not all perfectly
    uniform -- some genes were added in later game patches) are left as
    zeros rather than raising an error.
    """
    if genes is None:
        genes = load_gene_dicts(DEFAULT_GENE_DICTS_PATH)

    parsed = parse_dna_file(path)
    # Real CK3 exports always use "boy"/"girl" for `type=`, but several of the
    # hand-curated Discord-sourced training pairs instead wrote "male"/
    # "female". Treat both spellings as synonyms -- previously only "boy" was
    # recognized, so every "male"-labeled sample (true males) silently fell
    # through to gender=0.0 ("girl"), corrupting ~1/4 of the training labels
    # and teaching the model to regress male photos toward a female output.
    gender_str = (parsed["gender"] or "").strip().lower()
    gender = 255.0 if gender_str in ("boy", "male") else 0.0
    parsed_genes = parsed["genes"]

    segments = [torch.tensor([gender], dtype=torch.float32)]

    for color_name in COLOR_GENE_NAMES:
        raw = parsed_genes.get(color_name)
        vec = torch.zeros(4, dtype=torch.float32)
        if raw:
            vec[: len(raw)] = torch.tensor([float(v) for v in raw[:4]])
        segments.append(vec)

    for gene_name, options in genes.items():
        vec = torch.zeros(len(options), dtype=torch.float32)
        pairs = parsed_genes.get(gene_name)
        if pairs:
            for category, weight in pairs:
                if category in options:
                    vec[options.index(category)] = float(weight)
        segments.append(vec)

    return torch.cat(segments)


def tensor_ToDna(
    tensor: torch.Tensor,
    path,
    genes: Optional[Dict[str, List[str]]] = None,
    template_path=DEFAULT_DNA_TEMPLATE_PATH,
) -> None:
    """Inverse of `dna_ToTensor`: write a real-format CK3 DNA export at
    `path` from `tensor`.

    For each categorical gene, the top two weighted categories (by tensor
    value) become the gene's two ``"category" value`` pairs, matching the
    real format's blend-shape convention (for a gene with only one
    possible category, both pairs trivially repeat it). Color genes are
    written back as their four raw, clamped-to-[0, 255] integers.
    """
    if genes is None:
        genes = load_gene_dicts(DEFAULT_GENE_DICTS_PATH)

    tensor = tensor.detach().to("cpu").flatten()

    def _clamped_int(value: float) -> int:
        return int(round(max(0.0, min(255.0, float(value)))))

    offset = 0
    gender = "boy" if float(tensor[offset]) >= 127.5 else "girl"
    offset += 1

    output_genes: Dict[str, object] = {}
    for color_name in COLOR_GENE_NAMES:
        segment = tensor[offset : offset + 4]
        offset += 4
        output_genes[color_name] = [_clamped_int(v) for v in segment.tolist()]

    for gene_name, options in genes.items():
        length = len(options)
        segment = tensor[offset : offset + length]
        offset += length
        if length == 0:
            continue
        if length == 1:
            top_indices = [0, 0]
        else:
            top_indices = torch.topk(segment, k=min(2, length)).indices.tolist()
            if len(top_indices) < 2:
                top_indices.append(top_indices[0])
        pairs = []
        for idx in top_indices:
            pairs.append((options[idx], _clamped_int(segment[idx])))
        output_genes[gene_name] = pairs

    template = Path(template_path).read_text(encoding="utf-8")
    template = template.replace("type=boy", f"type={gender}", 1).replace("type=girl", f"type={gender}", 1)

    _line_gene_re = re.compile(r"^(\w+)\s*=\s*\{")

    def _replace_gene_lines(text: str) -> str:
        lines = text.split("\n")
        out_lines = []
        for line in lines:
            stripped = line.lstrip()
            indent = line[: len(line) - len(stripped)]
            match = _line_gene_re.match(stripped)
            gene_name = match.group(1) if match else None
            if gene_name in output_genes:
                out_lines.append(indent + format_gene_block(gene_name, output_genes[gene_name]))
            else:
                out_lines.append(line)
        return "\n".join(out_lines)

    template = _replace_gene_lines(template)

    Path(path).write_text(template, encoding="utf-8")
