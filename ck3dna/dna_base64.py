"""ck3dna.dna_base64

Best-effort decoder/encoder for CK3's *other* DNA format: the compact
base64 string (``dna="Hfwd/Jlmm..."``) embedded in save games and commonly
shared on Reddit/Discord, as opposed to the plaintext Ruler Designer
``ruler_designer_<id>={ genes={...} }`` format handled by
`ck3dna.ck3_format` / `ck3dna.dna_parser`.

CONFIDENCE LEVEL -- please read before relying on this module
================================================================
This is a **best-effort reverse-engineering**, not an officially documented
format, but it has now been validated against one real ground-truth
sample with strong results:

* The core algorithm (base64 -> raw bytes -> 4 bytes per gene, in a fixed
  hardcoded gene order) and the full gene-order/category-index tables in
  ``dependencies/base64_gene_index.json`` were extracted by statically
  analysing ``AppDnaConverter.js`` from the MIT-licensed
  `Kyusoo/CK3_Utility <https://github.com/Kyusoo/CK3_Utility>`_ project
  (a community CK3 DNA converter web app). That tool's logic is itself
  unofficial, and appears to target CK3 roughly around patch 1.5 -- it
  covers ~104 "vanilla" genes (the ``NONE`` gene order below) plus an
  extended ~131-gene order for a specific mod it calls ``EPE``.
* **Validated against a real base64 DNA string** for the "Aelyx" character
  (obtained via CK3's debug ``portrait_editor`` "Copy DNA string" feature)
  checked against its matching `real_dna_samples/rulers/Aelyx.ck3ruler`
  plaintext export (see `scripts/validate_base64_dna.py` and
  `tests/test_dna_base64.py::test_decode_matches_real_ground_truth_sample`).
  Of the 104 genes this codec's tables cover, **96 (92%) decoded to an
  exact match**, including all three color genes exactly. The remaining
  ~8 mismatches are isolated to a handful of genes whose category lists
  appear to have changed in patches after ~1.5 (e.g. `complexion`,
  `face_detail_chin_def`, the eye/teeth/eyelash accessory genes) rather
  than following one systemic pattern -- consistent with per-gene
  category-table version drift, not a byte-order or gene-order bug. (The
  source capture was also flagged by the user as coming from a
  momentarily-buggy portrait editor, so a little of that residual error
  may be in the captured string rather than the decode.)
* Our own `gene_dicts.json` (built from 85 real, modern `.ck3ruler`
  samples) has ~199 genes, including ~75 that do not appear in this
  base64 codec's tables at all (mostly newer-DLC genes such as the
  ``gene_dragon_*`` family, and a handful of renamed/added complexion and
  beard/hairstyle categories from patches after ~1.5). Those genes simply
  cannot be recovered from a base64 string with this decoder -- they will
  be silently absent from the decoded gene dict. This is expected and is
  not a bug; it reflects a real limitation of the only known decode logic,
  not of this port.

Byte layout (decode direction)
-------------------------------
1. Base64-decode the string to raw bytes (this is a plain, standard
   base64 decode -- despite `AppDnaConverter.js` routing through an
   unnecessary hex-string round trip, the net effect is exactly
   ``bytes(base64.b64decode(s))``).
2. The byte array is a flat sequence of 4-byte groups, one per gene, in a
   **fixed, hardcoded gene order** (not alphabetical, and not the same
   order as the plaintext format) -- see ``gene_order`` in
   ``base64_gene_index.json``. ``"NONE"`` is the vanilla/unmodded order;
   ``"EPE"`` is a longer order for a specific mod (included for
   completeness, not the primary target for Reddit-shared vanilla DNA
   strings).
3. For the three color genes (`hair_color`, `skin_color`, `eye_color`),
   the 4 bytes are used directly as raw integers -- exactly like the
   plaintext format.
4. For every other (categorical) gene, the 4 bytes are
   ``[category_index_1, value_1, category_index_2, value_2]``: the same
   two-pair blend convention as the plaintext format, except the category
   is a raw integer index into that specific gene's category list (per
   ``category_index`` in ``base64_gene_index.json``) rather than a named
   string. Decoding looks up the index back to its category name;
   encoding does the reverse.

The decoded/encoded dict shape matches `ck3dna.ck3_format.parse_dna_text`'s
output (``wrapper``/``gender``/``id``/``random_seed``/``genes``), so it can
be fed directly into `ck3dna.dna_parser`'s tensor conversion. The base64
payload itself carries no gender/id/random_seed metadata (those live
elsewhere in a save file), so those fields are returned as ``None``.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .ck3_format import COLOR_GENE_NAMES, GeneValue

DEFAULT_BASE64_GENE_INDEX_PATH = Path(__file__).parent / "dependencies" / "base64_gene_index.json"

_BASE64_GENE_INDEX_CACHE: Optional[dict] = None


def load_base64_gene_index(path=DEFAULT_BASE64_GENE_INDEX_PATH) -> dict:
    """Load (and cache, for the default path) the gene-order / category-index
    reference table extracted from `Kyusoo/CK3_Utility`'s converter.
    """
    global _BASE64_GENE_INDEX_CACHE
    if path == DEFAULT_BASE64_GENE_INDEX_PATH and _BASE64_GENE_INDEX_CACHE is not None:
        return _BASE64_GENE_INDEX_CACHE
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if path == DEFAULT_BASE64_GENE_INDEX_PATH:
        _BASE64_GENE_INDEX_CACHE = data
    return data


def _reverse_category_index(categories: Dict[str, int]) -> Dict[int, str]:
    """Invert a `{category_name: index}` map to `{index: category_name}`,
    keeping the *first* name seen for a given index (matching the JS
    source's `Object.keys(e).find(n => e[n] === t)` lookup order) in the
    rare case of a duplicate index within one gene.
    """
    reverse: Dict[int, str] = {}
    for name, index in categories.items():
        reverse.setdefault(index, name)
    return reverse


def decode_base64_dna(s: str, mod: str = "NONE", index: Optional[dict] = None) -> Dict[str, object]:
    """Decode a CK3 save-game base64 DNA string into the same dict shape
    produced by `ck3dna.ck3_format.parse_dna_text`.

    Args:
        s: the base64 string (e.g. the value of a save file's ``dna="..."``
            field). Whitespace is stripped; no ``dna=`` prefix/quotes are
            expected -- pass just the string contents.
        mod: ``"NONE"`` (default, vanilla gene order) or ``"EPE"`` (a
            longer gene order for a specific mod). See the module
            docstring's confidence-level notes.
        index: optional pre-loaded `base64_gene_index.json` dict (mainly
            for testing); defaults to loading the bundled reference table.

    Returns:
        A dict with ``wrapper="base64"``, ``gender=None``, ``id=None``,
        ``random_seed=None`` (the base64 payload does not encode these),
        and ``genes`` in the same shape as `ck3_format.parse_dna_text`.

    Raises:
        ValueError: if the decoded byte length is shorter than this gene
            order requires (longer inputs are truncated, matching the
            original converter's behaviour, since trailing padding/unused
            bytes have been observed in the wild).
    """
    if index is None:
        index = load_base64_gene_index()

    gene_order: List[str] = index["gene_order"][mod]
    category_index: Dict[str, Dict[str, int]] = index["category_index"]

    raw = base64.b64decode(s.strip())
    data = list(raw)

    required_length = 4 * len(gene_order)
    if len(data) < required_length:
        raise ValueError(
            f"Base64 DNA string decodes to {len(data)} bytes, "
            f"but the '{mod}' gene order expects at least {required_length} "
            f"(4 bytes x {len(gene_order)} genes)."
        )
    data = data[:required_length]

    genes: Dict[str, GeneValue] = {}
    for gene_idx, gene_name in enumerate(gene_order):
        chunk = data[gene_idx * 4 : gene_idx * 4 + 4]
        if gene_name in COLOR_GENE_NAMES:
            genes[gene_name] = list(chunk)
        else:
            categories = category_index.get(gene_name)
            if not categories:
                # No known category table for this gene (shouldn't happen
                # for anything in gene_order, but fail soft rather than
                # crash on an unexpected/future gene).
                continue
            reverse = _reverse_category_index(categories)
            cat1_idx, value1, cat2_idx, value2 = chunk
            cat1 = reverse.get(cat1_idx, f"<unknown_index_{cat1_idx}>")
            cat2 = reverse.get(cat2_idx, f"<unknown_index_{cat2_idx}>")
            genes[gene_name] = [(cat1, value1), (cat2, value2)]

    return {
        "wrapper": "base64",
        "gender": None,
        "id": None,
        "random_seed": None,
        "genes": genes,
    }


def encode_base64_dna(
    genes: Dict[str, GeneValue],
    mod: str = "NONE",
    index: Optional[dict] = None,
) -> str:
    """Best-effort inverse of `decode_base64_dna`: encode a `genes` dict
    (same shape as `ck3_format.parse_dna_text`'s ``"genes"`` value) back
    into a base64 DNA string.

    Genes in `gene_order` that are missing from `genes` are encoded as
    four zero bytes (CK3's "first category, weight 0" default). Genes
    present in `genes` but not in `gene_order` for this `mod` (e.g. the
    ~75 newer/modded genes our `gene_dicts.json` knows about but this
    codec's tables predate) are silently dropped -- there is no byte slot
    for them in this format, per the module docstring's confidence notes.
    """
    if index is None:
        index = load_base64_gene_index()

    gene_order: List[str] = index["gene_order"][mod]
    category_index: Dict[str, Dict[str, int]] = index["category_index"]

    out = bytearray()
    for gene_name in gene_order:
        value = genes.get(gene_name)
        if gene_name in COLOR_GENE_NAMES:
            ints = [0, 0, 0, 0]
            if value:
                for i, v in enumerate(value[:4]):
                    ints[i] = int(v) & 0xFF
            out.extend(ints)
        else:
            categories = category_index.get(gene_name, {})
            if value:
                (cat1, val1), (cat2, val2) = value[0], value[1] if len(value) > 1 else value[0]
                idx1 = categories.get(cat1, 0)
                idx2 = categories.get(cat2, 0)
                out.extend([idx1 & 0xFF, int(val1) & 0xFF, idx2 & 0xFF, int(val2) & 0xFF])
            else:
                out.extend([0, 0, 0, 0])

    return base64.b64encode(bytes(out)).decode("ascii")
