"""Consistency tests for `ck3dna.gene_dicts.get_lengths`."""
from __future__ import annotations

from ck3dna.ck3_format import COLOR_GENE_NAMES
from ck3dna.gene_dicts import COLOR_GENE_LENGTH, GENDER_LENGTH, get_lengths, load_gene_dicts, total_length


def test_lengths_match_gene_dicts():
    genes = load_gene_dicts()
    lengths = get_lengths()

    # [gender] + one length-4 segment per color gene + one segment per
    # categorical gene.
    assert lengths[0] == GENDER_LENGTH
    num_color = len(COLOR_GENE_NAMES)
    assert lengths[1 : 1 + num_color] == [COLOR_GENE_LENGTH] * num_color
    assert lengths[1 + num_color :] == [len(options) for options in genes.values()]
    assert len(lengths) == len(genes) + 1 + num_color


def test_total_length_matches_sum():
    assert total_length() == sum(get_lengths())


def test_all_gene_options_are_non_empty_and_unique():
    genes = load_gene_dicts()
    for gene_name, options in genes.items():
        assert len(options) > 0, f"{gene_name} has no sub-gene options"
        assert len(options) == len(set(options)), f"{gene_name} has duplicate sub-gene options"


def test_color_genes_not_duplicated_in_categorical_dict():
    genes = load_gene_dicts()
    for color_name in COLOR_GENE_NAMES:
        assert color_name not in genes, f"{color_name} should not appear in the categorical gene dict"
