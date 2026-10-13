"""ck3dna.extractor

Thin pytorch_lightning wrapper around a torchvision ResNet (50/101/152,
bottleneck-block variants) used as an image feature extractor. The final
fully-connected layer is replaced so the module outputs a feature vector of
size `num_classes` (despite the name -- kept for interface parity with the
upstream repo -- this is a feature dimensionality, not a classification
head).
"""
from __future__ import annotations

import pytorch_lightning as pl
import torch
import torchvision

_RESNET_BUILDERS = {
    50: torchvision.models.resnet50,
    101: torchvision.models.resnet101,
    152: torchvision.models.resnet152,
}


class Extractor(pl.LightningModule):
    def __init__(self, num_classes: int = 512, depth: int = 50, pretrained: bool = True):
        super().__init__()
        self.save_hyperparameters()
        if depth not in _RESNET_BUILDERS:
            raise ValueError(
                f"Unsupported ResNet depth {depth}; choose one of {sorted(_RESNET_BUILDERS)}"
            )
        weights = "DEFAULT" if pretrained else None
        self.backbone = _RESNET_BUILDERS[depth](weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
