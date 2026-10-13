"""ck3dna.regressor

Gated-residual MLP head that maps a fixed-size image feature vector to the
flattened DNA tensor (see `ck3dna.dna_parser` / `ck3dna.gene_dicts` for the
tensor layout).

BUG FIX vs. upstream: the original project's regressor multiplied the final
sigmoid output by ``265``. CK3 gene weights live on a 0-255 scale, so this
is almost certainly a typo for ``255`` -- fixed here as `DNA_WEIGHT_SCALE`.
"""
from __future__ import annotations

import pytorch_lightning as pl
import torch
from torch import nn

# Was `265` upstream (likely a typo) -- CK3 gene weights are 0-255.
DNA_WEIGHT_SCALE = 255.0


class _GatedResidualBlock(nn.Module):
    """One gated residual block: a main branch (linear->bn->ELU->linear->bn)
    is multiplied elementwise by a sigmoid gate branch
    (linear->ELU->linear->sigmoid), then passed through a final ELU.
    """

    def __init__(self, size: int):
        super().__init__()
        self.main_branch = nn.Sequential(
            nn.Linear(size, size),
            nn.BatchNorm1d(size),
            nn.ELU(),
            nn.Linear(size, size),
            nn.BatchNorm1d(size),
        )
        self.gate_branch = nn.Sequential(
            nn.Linear(size, size),
            nn.ELU(),
            nn.Linear(size, size),
            nn.Sigmoid(),
        )
        self.out_act = nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_act(self.main_branch(x) * self.gate_branch(x))


class Regressor(pl.LightningModule):
    """4 stacked gated residual blocks narrowing/widening
    `input_size -> hidden_size (x num_blocks) -> output_size`, followed by a
    final Linear + Sigmoid scaled by `DNA_WEIGHT_SCALE`.
    """

    def __init__(self, input_size: int, output_size: int, hidden_size: int = 1000, num_blocks: int = 4):
        super().__init__()
        self.save_hyperparameters()
        blocks = []
        in_size = input_size
        for _ in range(num_blocks):
            blocks.append(nn.Sequential(nn.Linear(in_size, hidden_size), _GatedResidualBlock(hidden_size)))
            in_size = hidden_size
        self.blocks = nn.ModuleList(blocks)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.output_activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        x = self.output_activation(self.output_layer(x))
        return x * DNA_WEIGHT_SCALE
