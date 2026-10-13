"""Unit tests for the low-level `ck3dna.ck3_format` block parser."""
from __future__ import annotations

import re

import pytest

from ck3dna.ck3_format import (
    find_block,
    find_portrait_block,
    format_gene_block,
    parse_dna_text,
    parse_gene_block_content,
)


def test_find_block_balances_nested_braces():
    text = "outer={ inner={ a=1 } more={ b=2 } } trailing=1"
    content, start, end = find_block(text, re.compile(r"outer\s*=\s*\{"))
    assert content.strip() == "inner={ a=1 } more={ b=2 }"


def test_find_portrait_block_detects_wrapper_kind():
    text = "ruler_designer_1={ type=boy genes={ hair_color={ 1 2 3 4 } } }"
    wrapper, content = find_portrait_block(text)
    assert wrapper == "ruler_designer"
    assert "hair_color" in content


def test_parse_gene_block_content_categorical_and_color():
    content = (
        ' hair_color={ 10 3 208 191 } '
        ' gene_chin_forward={ "chin_forward_pos" 133 "chin_forward_pos" 133 } '
        ' gene_bs_nose_tip_angle={ "nose_tip_angle_pos" 185 "nose_tip_angle_pos" 7 } '
    )
    genes = parse_gene_block_content(content)
    assert genes["hair_color"] == [10, 3, 208, 191]
    assert genes["gene_chin_forward"] == [("chin_forward_pos", 133), ("chin_forward_pos", 133)]
    assert genes["gene_bs_nose_tip_angle"] == [("nose_tip_angle_pos", 185), ("nose_tip_angle_pos", 7)]


def test_parse_dna_text_ck3ruler_wrapper():
    text = """
    version=1
    ruler={
        config={ age=30 }
        portrait_info={
            type=boy
            id=0
            random_seed=0
            genes={
                hair_color={ 1 2 3 4 }
                gene_chin_forward={ "chin_forward_pos" 100 "chin_forward_pos" 100 }
            }
            override={ portrait_modifier_overrides={ } }
            entity={ 0 0 }
        }
    }
    """
    parsed = parse_dna_text(text)
    assert parsed["wrapper"] == "portrait_info"
    assert parsed["gender"] == "boy"
    assert parsed["genes"]["hair_color"] == [1, 2, 3, 4]
    assert parsed["genes"]["gene_chin_forward"] == [("chin_forward_pos", 100), ("chin_forward_pos", 100)]


def test_parse_dna_text_ruler_designer_wrapper():
    text = """
    ruler_designer_123={
        type=girl
        id=0
        random_seed=0
        genes={
            skin_color={ 5 3 64 80 }
            gene_eye_angle={ "eye_angle_pos" 140 "eye_angle_pos" 140 }
        }
        override={ portrait_modifier_overrides={ } }
        entity={ 0 0 }
    }
    """
    parsed = parse_dna_text(text)
    assert parsed["wrapper"] == "ruler_designer"
    assert parsed["gender"] == "girl"
    assert parsed["genes"]["skin_color"] == [5, 3, 64, 80]


def test_parse_dna_text_raises_without_portrait_block():
    with pytest.raises(ValueError):
        parse_dna_text("not_a_valid_dna_file=1")


def test_format_gene_block_round_trip():
    color_text = format_gene_block("hair_color", [10, 3, 208, 191])
    assert color_text == "hair_color={ 10 3 208 191 }"

    categorical_text = format_gene_block(
        "gene_chin_forward", [("chin_forward_pos", 133), ("chin_forward_pos", 133)]
    )
    assert categorical_text == 'gene_chin_forward={ "chin_forward_pos" 133 "chin_forward_pos" 133 }'
