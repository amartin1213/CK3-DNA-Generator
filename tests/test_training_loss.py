"""Tests for ck3dna.training's PerGeneMSELoss (per-gene-normalized loss)."""
import torch

from ck3dna.training import PerGeneMSELoss


def test_per_gene_mse_zero_for_identical_tensors():
    lengths = [1, 4, 10]
    loss_fn = PerGeneMSELoss(lengths)
    batch_size = 3
    total = sum(lengths)
    x = torch.rand(batch_size, total)
    assert loss_fn(x, x).item() == 0.0


def test_per_gene_mse_weights_small_and_large_segments_equally():
    """A segment-length-1 gene and a segment-length-100 gene should
    contribute equally to the loss, unlike plain element-wise MSE where the
    100-dim gene would dominate by sheer element count.
    """
    lengths = [1, 100]
    loss_fn = PerGeneMSELoss(lengths)

    # Perfect prediction on the big segment, off-by-10 on the tiny one.
    target = torch.zeros(1, 101)
    pred = torch.zeros(1, 101)
    pred[0, 0] = 10.0  # error only in the length-1 segment

    per_gene = loss_fn(pred, target).item()
    plain = torch.nn.functional.mse_loss(pred, target).item()

    # Plain MSE dilutes the length-1 gene's error across all 101 elements;
    # per-gene MSE should report a much larger loss for the same error,
    # since that gene's full weight is 1/2 (not 1/101) of the total.
    assert per_gene > plain
    # The length-1 gene alone has MSE 100 (10**2); averaged over 2 genes
    # (the length-100 gene contributes 0) that's 50.
    assert per_gene == 50.0


def test_per_gene_mse_matches_elementwise_when_segments_equal_length():
    lengths = [5, 5, 5]
    loss_fn = PerGeneMSELoss(lengths)
    target = torch.rand(2, 15)
    pred = torch.rand(2, 15)
    per_gene = loss_fn(pred, target).item()
    plain = torch.nn.functional.mse_loss(pred, target).item()
    assert abs(per_gene - plain) < 1e-5
