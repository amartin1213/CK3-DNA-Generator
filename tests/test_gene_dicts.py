"""Consistency tests for `ck3dna.gene_dicts.get_lengths`."""
from __future__ import annotations

from ck3dna.gene_dicts import get_lengths, load_gene_dicts, total_length


def test_lengths_match_gene_dicts():
    genes = load_gene_dicts()
    lengths = get_lengths()

    # [age, gender] leading pair + one segment per gene.
    assert lengths[0] == 1
    assert lengths[1] == 1
    assert lengths[2:] == [len(options) for options in genes.values()]
    assert len(lengths) == len(genes) + 2


def test_total_length_matches_sum():
    assert total_length() == sum(get_lengths())


def test_all_gene_options_are_non_empty_and_unique():
    genes = load_gene_dicts()
    for gene_name, options in genes.items():
        assert len(options) > 0, f"{gene_name} has no sub-gene options"
        assert len(options) == len(set(options)), f"{gene_name} has duplicate sub-gene options"
