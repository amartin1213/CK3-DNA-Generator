"""ck3dna.model

End-to-end model: ResNet feature extractor -> gated-residual regressor ->
flattened DNA tensor (see `ck3dna.gene_dicts.get_lengths` for its layout).
"""
from __future__ import annotations

import pytorch_lightning as pl
import torch

from .extractor import Extractor
from .gene_dicts import get_lengths
from .regressor import Regressor


class Model(pl.LightningModule):
    def __init__(self, feature_size: int = 512, resnet_depth: int = 50, pretrained: bool = True):
        super().__init__()
        self.save_hyperparameters()
        output_size = sum(get_lengths())
        self.extractor = Extractor(num_classes=feature_size, depth=resnet_depth, pretrained=pretrained)
        self.regressor = Regressor(input_size=feature_size, output_size=output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extractor(x)
        return self.regressor(features)
