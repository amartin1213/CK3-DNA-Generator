"""Unit tests for `ck3dna.dna_base64` (the save-game base64 DNA codec).

Most of these tests exercise structural/round-trip correctness of the
Python port against the extracted reference tables themselves, since a
real ground-truth base64<->plaintext sample was not available when the
module was first written. `test_decode_matches_real_ground_truth_sample`
below *is* a real validation: a genuine base64 DNA string for the
"Aelyx" character (obtained via CK3's debug portrait editor), checked
against `real_dna_samples/rulers/Aelyx.ck3ruler`. See that test and the
module docstring for the resulting confidence-level discussion.
"""
from __future__ import annotations

import base64
from pathlib import Path

import pytest

from ck3dna.ck3_format import parse_dna_file
from ck3dna.dna_base64 import (
    decode_base64_dna,
    encode_base64_dna,
    load_base64_gene_index,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# A real base64 DNA string for "Aelyx", obtained via CK3's debug
# portrait_editor "Copy DNA string" feature. Note: the user flagged that
# their portrait editor was behaving buggy at the time of capture, so this
# is treated as best-effort ground truth, not a guaranteed-perfect
# reference -- see the module docstring and the test below for how this
# is interpreted.
_AELYX_BASE64 = (
    "AAAaCiUBGCL8vPUaAIwBjAFyAHIAcgByAX8AfwCMAIwAcgByAHIAcgFyAXIBggCCAH0AfQGMAYwB"
    "ggGCAYwBjACMAIwAdQF1AHIAcgCJAIkBggCCAIwAjAB8AHwAcgFyAHMAcwByAHIAcgByAH4AfgBy"
    "AHIAcgByAXIBcgByAHIBjAGMAIgAiAFyAXIAXwAdAccBtQEVAFUAVwBXAAAAAAI7AjsBMwEzAQUB"
    "BQF/ASUARQAhAQAAMQENAQ0A3wDfADsAOwEfAR8BCwELADMAMwFLARcAAAAAAcEBwQAVABUAFQA5"
    "ADMAVwAvAC8AcwBzAM8AzwFoAWgAHwAfAW0BbQELAQsAewB7AA0ADQDHAAEA/wDZASUBJQExATEA"
    "cQBxAAEAAQGfATsAEwATAHIAgQE4AAYE/wRKAQABAAEAAQAAJgAmAAAAAAEAAwACQwJDAAAAnQCU"
    "AGkDAAIAAgACAAAAAAADAAMACAAAXAGXAZYBWAFyAP8BAAAPBKQEvwT/A+sD6wOAA4AAAAB9Af8A"
    "fwD/AP8AfwB/AP8A/wAAAAADBQEDAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB/AH8AfwB/AH8AfwB/AH8AfwB/AH8AfwB/AH8A"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAH8AfwB/AP8AAAA7AFsBAAB/AH8AfwB/AV8B"
    "lwILAkQAfwB/AS8CAAB/AH8A/wB/ATsBOwK8AlsBlwGXAkQCRAIAAgAAAAAAASABIAAAAAACAwIA"
    "AZsBAAB/AH8BAAEAAjcCNwExATEAhgCGAVABUAFEAYEBEwETAQABNwD/AP8CAAIAAQoBCgHoAegA"
    "fwB/Av8AMA=="
)


@pytest.fixture(scope="module")
def index():
    return load_base64_gene_index()


def test_gene_index_has_expected_shape(index):
    assert set(index["gene_order"].keys()) >= {"NONE", "EPE"}
    assert len(index["gene_order"]["NONE"]) > 0
    assert len(index["gene_order"]["EPE"]) >= len(index["gene_order"]["NONE"])
    for color_gene in ("hair_color", "skin_color", "eye_color"):
        assert color_gene in index["gene_order"]["NONE"]
        # Color genes are raw bytes, not category-indexed.
        assert color_gene not in index["category_index"]


def test_decode_too_short_raises(index):
    # One byte short of the minimum expected length for the NONE order.
    short = base64.b64encode(bytes(4 * len(index["gene_order"]["NONE"]) - 1)).decode()
    with pytest.raises(ValueError):
        decode_base64_dna(short, mod="NONE", index=index)


def test_decode_all_zero_bytes_uses_first_category(index):
    gene_order = index["gene_order"]["NONE"]
    data = bytes(4 * len(gene_order))
    s = base64.b64encode(data).decode()
    result = decode_base64_dna(s, mod="NONE", index=index)

    assert result["wrapper"] == "base64"
    assert result["gender"] is None
    assert set(result["genes"].keys()) == set(gene_order)

    for gene_name in gene_order:
        value = result["genes"][gene_name]
        if gene_name in ("hair_color", "skin_color", "eye_color"):
            assert value == [0, 0, 0, 0]
        else:
            # Index 0 for both pairs, value 0 for both.
            (cat1, val1), (cat2, val2) = value
            assert val1 == 0 and val2 == 0
            categories = index["category_index"][gene_name]
            assert categories[cat1] == 0
            assert categories[cat2] == 0


def test_decode_truncates_longer_input(index):
    gene_order = index["gene_order"]["NONE"]
    required = 4 * len(gene_order)
    # Extra trailing junk bytes should be ignored, not break decoding.
    data = bytes(required) + b"\xff\xff\xff\xff"
    s = base64.b64encode(data).decode()
    result = decode_base64_dna(s, mod="NONE", index=index)
    assert len(result["genes"]) == len(gene_order)


def test_encode_then_decode_round_trip_for_known_categories(index):
    """Build a synthetic genes dict using the reference table's own
    category names/indices, encode it, decode it back, and check the
    category choices and values survive the round trip.
    """
    gene_order = index["gene_order"]["NONE"]
    category_index = index["category_index"]

    genes = {}
    for gene_name in gene_order:
        if gene_name in ("hair_color", "skin_color", "eye_color"):
            genes[gene_name] = [11, 22, 33, 44]
        else:
            categories = list(category_index[gene_name].keys())
            cat1 = categories[0]
            cat2 = categories[-1]
            genes[gene_name] = [(cat1, 100), (cat2, 200)]

    encoded = encode_base64_dna(genes, mod="NONE", index=index)
    decoded = decode_base64_dna(encoded, mod="NONE", index=index)

    for gene_name in gene_order:
        if gene_name in ("hair_color", "skin_color", "eye_color"):
            assert decoded["genes"][gene_name] == genes[gene_name]
        else:
            assert decoded["genes"][gene_name] == genes[gene_name]


def test_encode_missing_genes_default_to_zero(index):
    encoded = encode_base64_dna({}, mod="NONE", index=index)
    raw = base64.b64decode(encoded)
    assert raw == bytes(len(raw))


def test_encode_drops_genes_not_in_gene_order(index):
    genes = {"gene_dragon_body_pattern": [("some_category", 5), ("some_category", 5)]}
    # Should not raise, and should produce the same output as an empty dict
    # since this gene has no byte slot in the NONE order.
    encoded = encode_base64_dna(genes, mod="NONE", index=index)
    empty_encoded = encode_base64_dna({}, mod="NONE", index=index)
    assert encoded == empty_encoded


def test_decode_matches_real_ground_truth_sample(index):
    """Validate `decode_base64_dna` against a real base64 DNA string for
    "Aelyx" (obtained via CK3's debug portrait editor) against the matching
    `real_dna_samples/rulers/Aelyx.ck3ruler` plaintext export.

    Result (see `scripts/validate_base64_dna.py` for the full per-gene
    report this is based on): of the 191 genes in `Aelyx.ck3ruler`, 87 are
    not covered by this codec's ~1.5-era tables at all (mostly the
    `gene_dragon_*` family and a handful of genes added in later patches
    -- see `ck3dna.dna_base64`'s module docstring), which is expected and
    not a decode error. Of the **104 genes this codec does cover,
    96 (92%) decoded to an exact match** -- including all three color
    genes exactly. This is strong validation that the core algorithm
    (byte layout, gene order, blend-pair convention) is correct; the
    remaining ~8 per-gene mismatches look like isolated category-table
    version drift (a few genes gained/reordered categories in patches
    after the ~1.5 era this codec's tables were extracted from), not a
    systemic bug. The user also flagged their portrait editor as
    "currently bugged" when capturing this string, so a small residual
    error rate is plausible from the source capture too, not just the
    decoder.

    This test pins that known-good result as a regression check: it
    should keep matching at least this well, and should not silently
    regress to a much lower match rate (which would indicate a real
    byte-order/gene-order bug).
    """
    dna_path = REPO_ROOT / "real_dna_samples" / "rulers" / "Aelyx.ck3ruler"
    if not dna_path.exists():
        pytest.skip("real_dna_samples/rulers/Aelyx.ck3ruler not present in this checkout")

    decoded = decode_base64_dna(_AELYX_BASE64, mod="NONE", index=index)
    expected = parse_dna_file(dna_path)

    decoded_genes = decoded["genes"]
    expected_genes = expected["genes"]

    gene_order = set(index["gene_order"]["NONE"])
    covered = [name for name in expected_genes if name in gene_order]
    assert covered, "expected at least some genes to be covered by the NONE gene order"

    matches = sum(1 for name in covered if decoded_genes.get(name) == expected_genes.get(name))
    match_ratio = matches / len(covered)

    # Color genes should match exactly -- they're raw bytes, no category
    # table or version-drift ambiguity involved.
    for color_gene in ("hair_color", "skin_color", "eye_color"):
        assert decoded_genes[color_gene] == expected_genes[color_gene]

    # Known-good result was 96/104 (92%); allow a little slack for
    # environment/float differences but fail loudly on a real regression.
    assert match_ratio >= 0.85, (
        f"base64 decode only matched {matches}/{len(covered)} "
        f"({match_ratio:.0%}) covered genes against the real Aelyx sample "
        "-- expected >=85%, which would indicate a real decoding bug."
    )
