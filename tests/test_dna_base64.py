"""Unit tests for `ck3dna.dna_base64` (the save-game base64 DNA codec).

No ground-truth base64<->plaintext sample is available yet (see module
docstring), so these tests exercise structural/round-trip correctness of
the Python port against the extracted reference tables themselves, rather
than validating against real CK3 data.
"""
from __future__ import annotations

import base64

import pytest

from ck3dna.dna_base64 import (
    decode_base64_dna,
    encode_base64_dna,
    load_base64_gene_index,
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
