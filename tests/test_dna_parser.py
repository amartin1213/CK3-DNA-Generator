"""Round-trip tests for `ck3dna.dna_parser` against the project's
`gene_dicts.json` / `default_dna.txt` dependency files.
"""
from __future__ import annotations

from pathlib import Path

import torch

from ck3dna.dna_parser import DEFAULT_DNA_TEMPLATE_PATH, dna_ToTensor, tensor_ToDna
from ck3dna.gene_dicts import get_lengths, load_gene_dicts, total_length


def test_default_dna_parses_without_error():
    tensor = dna_ToTensor(DEFAULT_DNA_TEMPLATE_PATH)
    assert tensor.shape == (total_length(),)


def test_default_dna_round_trip(tmp_path: Path):
    """Parsing the default template to a tensor and back should reproduce
    the same (gender, age, per-gene argmax option) choices.
    """
    tensor = dna_ToTensor(DEFAULT_DNA_TEMPLATE_PATH)

    out_path = tmp_path / "round_trip.txt"
    tensor_ToDna(tensor, out_path)

    round_tripped = dna_ToTensor(out_path)
    assert torch.equal(tensor, round_tripped)


def test_tensor_to_dna_selects_gene_argmax(tmp_path: Path):
    genes = load_gene_dicts()
    lengths = get_lengths()
    tensor = torch.zeros(sum(lengths))

    # age, gender
    tensor[0] = 45.0
    tensor[1] = 255.0  # male

    offset = 2
    expected = {}
    for (gene_name, options), length in zip(genes.items(), lengths[2:]):
        segment_start = offset
        offset += length
        # Pick the last option as the "winner" for this gene.
        winner_index = length - 1
        tensor[segment_start + winner_index] = 200.0
        expected[gene_name] = options[winner_index]

    out_path = tmp_path / "predicted.txt"
    tensor_ToDna(tensor, out_path)

    text = out_path.read_text(encoding="utf-8")
    assert "type=male" in text
    assert "age=45" in text
    for gene_name, option in expected.items():
        assert f'{gene_name}=\n{{\n\t"{option}" 200\n}}' in text


def test_gender_female_round_trip(tmp_path: Path):
    lengths = get_lengths()
    tensor = torch.zeros(sum(lengths))
    tensor[0] = 20.0
    tensor[1] = 0.0  # female

    out_path = tmp_path / "female.txt"
    tensor_ToDna(tensor, out_path)
    text = out_path.read_text(encoding="utf-8")
    assert "type=female" in text
    assert "age=20" in text
