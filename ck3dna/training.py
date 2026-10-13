"""ck3dna.training

Dataset / LightningDataModule / LightningModule wiring for training the
end-to-end model, plus an `experiment` convenience wrapper driven by CLI
args (see `ck3dna.parser`).

Expects data preprocessed by `ck3dna.data_processing` (alignment, producing
``portraits#i/angle#ii.jpg`` + ``portraits#i/dna#i.txt``) followed by
`ck3dna.preprocess_tensors` (serializing the aligned images to `.pt`
tensors), laid out as::

    datastore/
        portraits#0/
            angle#0.pt
            dna#0.txt
        portraits#1/
            angle#0.pt
            dna#1.txt
        ...

Simplifying assumption: if a ``portraits#i`` folder has multiple
pre-aligned angle images, only the first (lexicographically, e.g.
``angle#0.pt``) is used per training sample, for a clean 1:1 image<->DNA
pairing.

PYTORCH-LIGHTNING VERSION NOTE: the upstream project used the
``training_epoch_end(outputs)`` / ``validation_epoch_end(outputs)`` hooks,
which were removed in pytorch-lightning >=2.0. This rebuild instead uses the
current ``on_train_epoch_end`` / ``on_validation_epoch_end`` hooks together
with manually accumulated step-output lists, so it works with modern
pytorch-lightning without needing to pin an old version.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset, random_split

from .dna_parser import dna_ToTensor
from .model import Model


def _portrait_loader(path) -> torch.Tensor:
    """Loader for the image side of the paired dataset: load a saved `.pt` tensor."""
    return torch.load(path)


class PortraitDnaDataset(Dataset):
    """Pairs one serialized portrait tensor with its parsed DNA tensor, one
    sample per ``portraits#i`` folder (see module docstring for the expected
    layout and the 1-image-per-folder simplifying assumption).
    """

    def __init__(self, root: str):
        self.root = Path(root)
        self.samples = []
        for portrait_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            image_tensors = sorted(portrait_dir.glob("angle#*.pt"))
            dna_files = sorted(portrait_dir.glob("dna#*.txt"))
            if image_tensors and dna_files:
                self.samples.append((image_tensors[0], dna_files[0]))
        if not self.samples:
            raise RuntimeError(
                f"No (image, dna) pairs found under {root}. Did you run "
                "ck3dna.data_processing and ck3dna.preprocess_tensors first?"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, dna_path = self.samples[index]
        image = _portrait_loader(image_path)
        dna = dna_ToTensor(dna_path)
        return image, dna


class PortraitDnaDataModule(pl.LightningDataModule):
    def __init__(self, datastore_name: str, batch_size: int = 16, workers: int = 0, val_ratio: float = 0.2):
        super().__init__()
        self.datastore_name = datastore_name
        self.batch_size = batch_size
        self.workers = workers
        self.val_ratio = val_ratio

    def setup(self, stage: Optional[str] = None):
        full = PortraitDnaDataset(self.datastore_name)
        val_size = max(1, int(len(full) * self.val_ratio))
        train_size = len(full) - val_size
        self.train_set, self.val_set = random_split(full, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.workers)


class Net(pl.LightningModule):
    """MSE-trained wrapper around `ck3dna.model.Model`, with an RMSprop
    optimizer and a `ReduceLROnPlateau` scheduler monitoring validation loss.
    """

    def __init__(self, lr: float = 1e-3, feature_size: int = 512, resnet_depth: int = 50, pretrained: bool = True):
        super().__init__()
        self.save_hyperparameters()
        self.model = Model(feature_size=feature_size, resnet_depth=resnet_depth, pretrained=pretrained)
        self.loss_fn = torch.nn.MSELoss()
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        preds = self(images)
        loss = self.loss_fn(preds, targets)
        self.training_step_outputs.append(loss.detach())
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        preds = self(images)
        loss = self.loss_fn(preds, targets)
        self.validation_step_outputs.append(loss.detach())
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self) -> None:
        if self.training_step_outputs:
            avg_loss = torch.stack(self.training_step_outputs).mean()
            self.log("train_loss_epoch_avg", avg_loss)
            self.training_step_outputs.clear()

    def on_validation_epoch_end(self) -> None:
        if self.validation_step_outputs:
            avg_loss = torch.stack(self.validation_step_outputs).mean()
            self.log("val_loss_epoch_avg", avg_loss)
            self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }


class experiment:
    """Wires together the data module, model and trainer from CLI args
    (see `ck3dna.parser.build_arg_parser`)."""

    def __init__(self, args):
        self.args = args
        self.datamodule = PortraitDnaDataModule(
            datastore_name=args.datastore_name,
            batch_size=args.batch_size,
            workers=args.workers,
            val_ratio=args.val_ratio,
        )
        self.net = Net(lr=args.lr)
        callbacks = [EarlyStopping(monitor="val_loss", patience=args.val_epochs)]
        logger = TensorBoardLogger(save_dir="tb_logs", name="ck3dna")
        self.trainer = pl.Trainer(
            max_epochs=args.epochs,
            devices=args.gpus if args.gpus else "auto",
            accelerator="gpu" if args.gpus else "auto",
            callbacks=callbacks,
            logger=logger,
        )

    def run(self) -> None:
        self.trainer.fit(self.net, datamodule=self.datamodule, ckpt_path=self.args.ckpt or None)


def main(argv=None) -> None:
    from .parser import build_arg_parser

    args = build_arg_parser().parse_args(argv)
    experiment(args).run()


if __name__ == "__main__":
    main()
