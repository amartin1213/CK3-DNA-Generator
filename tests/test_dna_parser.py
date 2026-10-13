"""Round-trip tests for `ck3dna.dna_parser` / `ck3dna.ck3_format` against
both the project's `default_dna.txt` template and real CK3 Ruler Designer
sample exports under `real_dna_samples/`.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ck3dna.ck3_format import parse_dna_file
from ck3dna.dna_parser import DEFAULT_DNA_TEMPLATE_PATH, dna_ToTensor, tensor_ToDna
from ck3dna.gene_dicts import get_lengths, load_gene_dicts, total_length

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = REPO_ROOT / "real_dna_samples"

REAL_RULER_SAMPLES = sorted((SAMPLES_DIR / "rulers").glob("*.ck3ruler"))
REAL_DISCORD_SAMPLES = sorted((SAMPLES_DIR / "discord_pairs").glob("*.txt"))


def test_default_dna_parses_without_error():
    tensor = dna_ToTensor(DEFAULT_DNA_TEMPLATE_PATH)
    assert tensor.shape == (total_length(),)


def test_default_dna_round_trip(tmp_path: Path):
    """Parsing the default template to a tensor and back should reproduce
    the same (gender, per-gene category/value) choices.
    """
    tensor = dna_ToTensor(DEFAULT_DNA_TEMPLATE_PATH)

    out_path = tmp_path / "round_trip.txt"
    tensor_ToDna(tensor, out_path)

    round_tripped = dna_ToTensor(out_path)
    assert torch.equal(tensor, round_tripped)


def test_tensor_to_dna_selects_gene_top_categories(tmp_path: Path):
    genes = load_gene_dicts()
    lengths = get_lengths()
    tensor = torch.zeros(sum(lengths))

    tensor[0] = 255.0  # boy

    offset = 1 + 4 * 3  # gender + hair/skin/eye color segments (each 4 tensor slots)
    expected = {}
    for (gene_name, options), length in zip(genes.items(), lengths[4:]):
        segment_start = offset
        offset += length
        winner_index = length - 1
        tensor[segment_start + winner_index] = 200.0
        expected[gene_name] = options[winner_index]

    out_path = tmp_path / "predicted.txt"
    tensor_ToDna(tensor, out_path)

    parsed = parse_dna_file(out_path)
    assert parsed["gender"] == "boy"
    for gene_name, option in expected.items():
        pairs = parsed["genes"][gene_name]
        # The single-winner category must come first (highest weight); the
        # second slot is whatever other category had the next-highest
        # weight (0, since this test only sets the winner's weight).
        assert pairs[0] == (option, 200)


def test_gender_girl_round_trip(tmp_path: Path):
    lengths = get_lengths()
    tensor = torch.zeros(sum(lengths))
    tensor[0] = 0.0  # girl

    out_path = tmp_path / "girl.txt"
    tensor_ToDna(tensor, out_path)
    parsed = parse_dna_file(out_path)
    assert parsed["gender"] == "girl"


@pytest.mark.parametrize("sample_path", REAL_RULER_SAMPLES[:5], ids=lambda p: p.stem)
def test_real_ck3ruler_samples_round_trip(sample_path: Path, tmp_path: Path):
    """Parsing a genuine `.ck3ruler` sample to a tensor and back to text
    should reproduce exactly the same gender + per-gene values.
    """
    tensor = dna_ToTensor(sample_path)
    assert tensor.shape == (total_length(),)

    out_path = tmp_path / f"{sample_path.stem}.txt"
    tensor_ToDna(tensor, out_path)

    round_tripped = dna_ToTensor(out_path)
    assert torch.equal(tensor, round_tripped)


@pytest.mark.parametrize("sample_path", REAL_DISCORD_SAMPLES, ids=lambda p: p.stem)
def test_real_discord_exports_round_trip(sample_path: Path, tmp_path: Path):
    """The 2 genuine (photo, DNA) discord pairs use the flat
    ``ruler_designer_<id>=`` wrapper (vs. `.ck3ruler`'s nested
    ``ruler={ portrait_info={...} }``) -- make sure both wrapper shapes
    round-trip correctly.
    """
    tensor = dna_ToTensor(sample_path)
    assert tensor.shape == (total_length(),)

    out_path = tmp_path / f"{sample_path.stem}.txt"
    tensor_ToDna(tensor, out_path)

    round_tripped = dna_ToTensor(out_path)
    assert torch.equal(tensor, round_tripped)


def test_ck3ruler_wrapper_detected_as_portrait_info():
    parsed = parse_dna_file(REAL_RULER_SAMPLES[0])
    assert parsed["wrapper"] == "portrait_info"


def test_discord_wrapper_detected_as_ruler_designer():
    parsed = parse_dna_file(REAL_DISCORD_SAMPLES[0])
    assert parsed["wrapper"] == "ruler_designer"
