"""ck3dna.ck3_format

Low-level helpers for reading CK3's native (Clausewitz-script-like) DNA
export format, as found in real `.ck3ruler` files and raw
`ruler_designer_<id>={...}` text exports.

Two concrete wrapper shapes are supported, both containing the same inner
`type=`/`id=`/`random_seed=`/`genes={...}`/`override={...}`/`entity={...}`
structure:

* `.ck3ruler` save files -- the portrait data is nested one level deeper,
  under ``ruler={ ... portrait_info={ ... } }``.
* Raw Ruler Designer / Discord exports -- the portrait data is the
  top-level block, usually keyed as ``ruler_designer_<id>={ ... }``, but
  some real-world exports use a non-numeric placeholder key instead (e.g.
  ``TODO_choose_portait_key={ ... }``, observed in a community-shared
  sample). Any top-level ``<identifier>={ ... }`` block containing a nested
  ``genes={...}`` block is accepted as a fallback for this case.

Within the portrait block, every gene is one of two shapes:

* **Color genes** (`hair_color`, `skin_color`, `eye_color`): four raw
  integers, e.g. ``hair_color={ 10 3 208 191 }``.
* **Categorical genes** (everything else, e.g. `gene_chin_forward`,
  `gene_bs_nose_tip_angle`, ...): exactly two ``"category" value`` pairs,
  e.g. ``gene_chin_forward={ "chin_forward_pos" 133 "chin_forward_pos" 133 }``.
  The two pairs blend between (usually adjacent, sometimes identical)
  named sub-categories -- this is how CK3 interpolates blend-shapes.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# A categorical gene's parsed value: [(category, value), (category, value)].
CategoricalGene = List[Tuple[str, int]]
# A color gene's parsed value: four raw 0-255 ints.
ColorGene = List[int]
GeneValue = Union[CategoricalGene, ColorGene]

COLOR_GENE_NAMES = ("hair_color", "skin_color", "eye_color")

_PORTRAIT_INFO_KEY_RE = re.compile(r"portrait_info\s*=\s*\{")
_RULER_DESIGNER_KEY_RE = re.compile(r"ruler_designer_\d+\s*=\s*\{")
# Fallback for raw exports whose top-level key isn't the usual
# ``ruler_designer_<id>=`` -- observed in the wild as
# ``TODO_choose_portait_key=`` (a placeholder left by whatever tool produced
# the export, seemingly when it wasn't tied to a saved Ruler Designer slot).
# Any ``<identifier>={`` block is accepted here; `find_portrait_block` only
# falls back to it after the two known-good keys fail to match, and the
# caller still requires a nested ``genes={...}`` block to parse
# successfully, so this stays narrow in practice.
_GENERIC_TOP_LEVEL_KEY_RE = re.compile(r"(?m)^\s*\w+\s*=\s*\{")
_GENES_KEY_RE = re.compile(r"(?<![\w])genes\s*=\s*\{")
_TYPE_RE = re.compile(r"(?m)^\s*type\s*=\s*(\w+)")
_ID_RE = re.compile(r"(?m)^\s*id\s*=\s*(\d+)")
_RANDOM_SEED_RE = re.compile(r"(?m)^\s*random_seed\s*=\s*(\d+)")

_GENE_ENTRY_RE = re.compile(r"(\w+)\s*=\s*\{([^{}]*)\}")
_QUOTED_PAIR_RE = re.compile(r'"([^"]+)"\s*(-?\d+)')
_RAW_INTS_RE = re.compile(r"-?\d+")


def find_block(text: str, key_regex: "re.Pattern[str]", start: int = 0) -> Optional[Tuple[str, int, int]]:
    """Find the first ``key={`` match (per `key_regex`) at/after `start`, and
    return ``(content, content_start, content_end)`` where `content` is the
    brace-balanced text between the matching ``{`` and ``}`` (exclusive).

    Returns None if no match is found.
    """
    match = key_regex.search(text, start)
    if match is None:
        return None
    content_start = match.end()  # just past the opening '{'
    depth = 1
    i = content_start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth != 0:
        raise ValueError("Unbalanced braces while scanning for a block")
    content_end = i - 1  # index of the matching closing '}'
    return text[content_start:content_end], content_start, content_end


def find_portrait_block(text: str) -> Tuple[str, str]:
    """Locate the portrait-info block in `text`, regardless of which of the
    two supported wrapper shapes it uses.

    Returns ``(wrapper_kind, content)`` where `wrapper_kind` is
    ``"portrait_info"`` (``.ck3ruler`` files) or ``"ruler_designer"`` (raw
    exports), and `content` is the full text between that block's braces.
    """
    found = find_block(text, _PORTRAIT_INFO_KEY_RE)
    if found is not None:
        return "portrait_info", found[0]
    found = find_block(text, _RULER_DESIGNER_KEY_RE)
    if found is not None:
        return "ruler_designer", found[0]
    # Last resort: some raw exports use a non-numeric placeholder top-level
    # key instead of ruler_designer_<id>= (e.g. "TODO_choose_portait_key=").
    # Structurally identical otherwise, so treat it the same as the
    # ruler_designer wrapper -- but only accept it if it actually contains a
    # nested genes={...} block, to avoid matching an unrelated top-level key.
    found = find_block(text, _GENERIC_TOP_LEVEL_KEY_RE)
    if found is not None and _GENES_KEY_RE.search(found[0]):
        return "ruler_designer", found[0]
    raise ValueError("No portrait_info={...} or ruler_designer_<id>={...} block found")


def parse_gene_block_content(genes_content: str) -> Dict[str, GeneValue]:
    """Parse the flat content of a `genes={...}` block into a mapping of
    gene name -> parsed value (see module docstring for the two shapes).
    """
    genes: Dict[str, GeneValue] = {}
    for name, body in _GENE_ENTRY_RE.findall(genes_content):
        pairs = _QUOTED_PAIR_RE.findall(body)
        if pairs:
            genes[name] = [(category, int(value)) for category, value in pairs]
        else:
            genes[name] = [int(v) for v in _RAW_INTS_RE.findall(body)]
    return genes


def parse_dna_text(text: str) -> Dict[str, object]:
    """Parse a full `.ck3ruler` or raw `ruler_designer_<id>=` DNA export.

    Returns a dict with keys ``wrapper`` ("portrait_info"/"ruler_designer"),
    ``gender`` (the raw `type=` value, e.g. "boy"/"girl"), ``id``,
    ``random_seed`` and ``genes`` (see `parse_gene_block_content`).
    """
    wrapper, portrait_content = find_portrait_block(text)

    type_match = _TYPE_RE.search(portrait_content)
    gender = type_match.group(1) if type_match else None

    id_match = _ID_RE.search(portrait_content)
    char_id = int(id_match.group(1)) if id_match else None

    seed_match = _RANDOM_SEED_RE.search(portrait_content)
    random_seed = int(seed_match.group(1)) if seed_match else None

    genes_found = find_block(portrait_content, _GENES_KEY_RE)
    if genes_found is None:
        raise ValueError("No genes={...} block found inside the portrait block")
    genes = parse_gene_block_content(genes_found[0])

    return {
        "wrapper": wrapper,
        "gender": gender,
        "id": char_id,
        "random_seed": random_seed,
        "genes": genes,
    }


def parse_dna_file(path) -> Dict[str, object]:
    """Convenience wrapper around `parse_dna_text` for a file on disk.

    Works for both `.ck3ruler` save files and raw `.txt`
    `ruler_designer_<id>=` exports -- the wrapper shape is auto-detected.
    """
    text = Path(path).read_text(encoding="utf-8")
    return parse_dna_text(text)


def format_gene_block(name: str, value: GeneValue) -> str:
    """Render a single parsed gene value back to CK3's native text form."""
    if name in COLOR_GENE_NAMES:
        ints = " ".join(str(v) for v in value)
        return f"{name}={{ {ints} }}"
    pairs = " ".join(f'"{category}" {weight}' for category, weight in value)
    return f"{name}={{ {pairs} }}"
