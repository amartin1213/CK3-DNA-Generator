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

import json
from pathlib import Path
from typing import Dict, List, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset, random_split

from .dna_parser import dna_ToTensor
from .gene_dicts import get_lengths, load_gene_dicts
from .model import Model

# Built by `scripts/build_dna_prior.py` from a corpus of genuine CK3 Ruler
# Designer `.ck3ruler` character files (see that script's docstring). Only
# aggregate per-gene standard deviations are stored here, never the source
# character files themselves.
DEFAULT_DIVERSITY_PRIOR_PATH = Path(__file__).parent / "dependencies" / "dna_prior_std.json"


class PerGeneMSELoss(torch.nn.Module):
    """MSE loss that first averages squared error *within* each gene
    segment, then averages those per-gene means across genes (rather than
    averaging over every raw tensor element, which is what plain
    ``nn.MSELoss`` does).

    Why: the flattened DNA tensor mixes very different segment sizes --
    4-dim color genes, 1-2-dim categorical genes, and 10+-dim categorical
    genes all concatenated together (see `ck3dna.gene_dicts.get_lengths`).
    Plain element-wise MSE implicitly weights each *gene* by its segment
    length, so a handful of high-cardinality genes can dominate the
    gradient purely by dimension count, while low-cardinality genes get
    diluted. Averaging per-gene first gives every gene equal weight
    regardless of how many categories it has.

    `gene_weights` (name -> multiplier, matched by exact gene name) lets
    specific genes get *more* than that equal share. Added after
    `scripts/analyze_gene_variance.py` found that mouth/jaw/chin genes
    stayed collapsed (predictions clustering near a fixed blend rather than
    responding to the input photo -- reported in practice as "puffy lips",
    "no jawline", "chin too high") even with equal per-gene weighting,
    worse than other genes of similar segment size. `names`/`group_weights`
    is a convenience that applies a multiplier to every gene whose name
    contains a given substring (e.g. ``{"mouth": 3.0, "jaw": 3.0, "chin": 3.0}``).
    """

    def __init__(
        self,
        lengths,
        names: Optional[List[str]] = None,
        group_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.lengths = list(lengths)
        weights = [1.0] * len(self.lengths)
        if names is not None and group_weights:
            assert len(names) == len(self.lengths), "names must align with lengths"
            for i, name in enumerate(names):
                for substr, mult in group_weights.items():
                    if substr in name:
                        weights[i] *= mult
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        offset = 0
        per_gene_losses = []
        for length in self.lengths:
            pred_seg = pred[:, offset : offset + length]
            target_seg = target[:, offset : offset + length]
            offset += length
            per_gene_losses.append(torch.nn.functional.mse_loss(pred_seg, target_seg))
        losses = torch.stack(per_gene_losses)
        # Normalize by gene *count*, not by the weights' sum. Dividing by
        # weights.sum() turns this into a weighted *average* where adding a
        # new boosted group (e.g. --lip_weight) grows the denominator and
        # proportionally shrinks every gene's absolute loss contribution --
        # including genes boosted by an *earlier*, separate group (e.g.
        # --chin_weight, --eye_weight). That was caught empirically: adding
        # --lip_weight 2.5 on top of an already-tuned --chin_weight/
        # --eye_weight config measurably regressed chin/eye diversity
        # ratios that a prior retrain (without --lip_weight) had fixed.
        # Normalizing by count instead keeps every unweighted gene's
        # absolute contribution constant (matches plain equal-weighting when
        # all weights are 1.0) and makes each group's boost purely additive,
        # so independently-tuned weight groups compose without interfering.
        return (losses * self.weights).sum() / len(self.weights)


class GenderBCELoss(torch.nn.Module):
    """Binary-cross-entropy loss on the DNA tensor's leading gender channel
    (encoded 0.0="girl"/255.0="boy", see module docstring), instead of
    folding it into the generic per-gene MSE.

    Why: gender is fundamentally a binary classification target (CK3's
    `type=` field is always exactly "boy" or "girl", never a blend), but
    plain squared error lets the model hedge near the 0-255 midpoint for
    visually ambiguous photos -- that minimizes *expected* squared error,
    but produces exactly the "misgendered" predictions seen in practice.
    BCE's gradient grows steeply as a confident-but-wrong prediction
    approaches the opposite extreme, pushing the model to commit to one
    side rather than average toward the training set's gender ratio.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_prob = (pred[:, 0] / 255.0).clamp(self.eps, 1.0 - self.eps)
        target_prob = (target[:, 0] / 255.0).clamp(0.0, 1.0)
        return torch.nn.functional.binary_cross_entropy(pred_prob, target_prob)


def get_gene_names() -> List[str]:
    """Return the flattened DNA tensor's per-segment gene names, in the same
    order as `ck3dna.gene_dicts.get_lengths()` (the 4 fixed leading segments
    -- gender, hair/skin/eye color -- followed by every categorical gene in
    `gene_dicts.json` order).
    """
    gene_dicts = load_gene_dicts()
    return ["gender", "hair_color", "skin_color", "eye_color"] + list(gene_dicts.keys())


def load_diversity_prior_stds(path=DEFAULT_DIVERSITY_PRIOR_PATH, lengths: Optional[List[int]] = None) -> List[float]:
    """Load the per-gene-segment target standard deviations produced by
    `scripts/build_dna_prior.py`, in the same gene order as `get_lengths()`
    (which `DiversityPriorLoss` expects)."""
    if lengths is None:
        lengths = get_lengths()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    names = get_gene_names()
    per_gene_std = data["per_gene_std"]
    stds = [per_gene_std[name] for name in names]
    assert len(stds) == len(lengths), "diversity prior gene count doesn't match gene_dicts"
    return stds


class DiversityPriorLoss(torch.nn.Module):
    """Regularizer that penalizes a training batch's predictions for having
    *less* per-gene spread (standard deviation across the batch) than real
    CK3 characters typically exhibit, pushing the model away from
    mode-collapsed/near-identical output ("every face looks the same") and
    toward the natural diversity found in a reference corpus of genuine DNA
    samples (see `scripts/build_dna_prior.py` and
    `dependencies/dna_prior_std.json`).

    Deliberately one-sided (via `relu`): predictions *more* diverse than the
    reference aren't penalized, only collapsed ones. This is a secondary
    regularizer, not a substitute for the primary per-gene MSE loss -- on
    its own it can be driven to zero by any sufficiently-spread-out batch,
    irrespective of per-gene accuracy against ground truth.
    """

    def __init__(self, lengths: List[int], target_stds: List[float]):
        super().__init__()
        self.lengths = list(lengths)
        self.register_buffer("target_stds", torch.tensor(target_stds, dtype=torch.float32))

    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        if pred.size(0) < 2:
            # Batch-wise std is undefined for a single sample; skip rather
            # than propagate a spurious zero/NaN penalty.
            return pred.new_zeros(())
        offset = 0
        penalties = []
        for length, target_std in zip(self.lengths, self.target_stds.tolist()):
            seg = pred[:, offset : offset + length]
            offset += length
            seg_std = seg.std(dim=0).mean()
            penalties.append(torch.relu(target_std - seg_std) ** 2)
        return torch.stack(penalties).mean()


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
            dna_files = sorted(portrait_dir.glob("dna#*.txt")) + sorted(portrait_dir.glob("dna#*.ck3ruler"))
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
        # drop_last=True avoids a trailing batch of size 1, which would crash
        # the regressor's BatchNorm1d layers (they require >1 sample per
        # channel in training mode) -- this matters in practice on small
        # real-world datasets where len(train_set) isn't a multiple of
        # batch_size (e.g. 17 samples / batch_size=4 -> a final batch of 1).
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.workers,
            shuffle=True,
            drop_last=len(self.train_set) > self.batch_size,
        )

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.workers)


class Net(pl.LightningModule):
    """`PerGeneMSELoss`-trained (per-gene-normalized MSE by default) wrapper
    around `ck3dna.model.Model`, with an RMSprop optimizer and a
    `ReduceLROnPlateau` scheduler monitoring validation loss.
    """

    def __init__(
        self,
        lr: float = 3e-3,
        feature_size: int = 512,
        resnet_depth: int = 50,
        pretrained: bool = True,
        per_gene_loss: bool = True,
        diversity_weight: float = 0.0,
        diversity_prior_path: Optional[str] = None,
        gender_weight: float = 300.0,
        facial_structure_weight: float = 1.0,
        chin_weight: float = 1.0,
        eye_weight: float = 1.0,
        lip_weight: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = Model(feature_size=feature_size, resnet_depth=resnet_depth, pretrained=pretrained)
        # See `PerGeneMSELoss` docstring: normalizes per-gene instead of
        # per-raw-element so high-cardinality genes don't drown out
        # low-cardinality ones in the gradient. `per_gene_loss=False`
        # restores the original flat `nn.MSELoss` behavior for comparison.
        # `facial_structure_weight` (>1.0) additionally upweights mouth/jaw/
        # chin genes specifically: `analyze_gene_variance.py` found these
        # stayed collapsed (near-constant blended predictions, reported as
        # "puffy lips"/"no jawline"/"chin too high") even under equal
        # per-gene weighting, worse than other genes of similar size.
        # `chin_weight` stacks an *additional* multiplier on top of that for
        # "chin"-named genes specifically: even after a 3x facial_structure_
        # weight boost, gene_chin_height stayed one of the single most
        # collapsed genes in the whole model (ratio ~0.245) while jaw_height
        # with the same 3x reached ~0.431 -- so chin height needed an extra
        # push beyond the shared mouth/jaw/chin multiplier.
        # `eye_weight` is a separate group for "eye"-named genes, added after
        # a user report that eye angle/shape looked off; this group wasn't
        # covered by facial_structure_weight at all previously.
        # `lip_weight` stacks a further multiplier on top of the shared
        # "mouth" group for "lip"-named genes specifically (e.g.
        # mouth_upper_lip_*, mouth_lower_lip_*, mouth_*_lip_forward/profile/
        # glamour_lips/philtrum): a user report of "lack of lip similarity",
        # overly-forward mouth corners, and a visible seam under the lower
        # lip traced back to these lip-shape sub-genes staying badly
        # collapsed (ratios ~0.10-0.31) even under the overall 3x "mouth"
        # boost from facial_structure_weight -- much worse than other
        # mouth genes like mouth_height (~0.45) receiving the same
        # multiplier, so the lip-shape genes needed an extra push. Since
        # "lip" is a distinct dict key from "mouth"/"chin"/"jaw"/"eye", it
        # composes automatically in `PerGeneMSELoss` (multiplicative across
        # every substring that matches a gene's name) without needing an
        # explicit merge like `chin_weight` does.
        if per_gene_loss:
            group_weights: Dict[str, float] = {}
            if facial_structure_weight != 1.0:
                group_weights["mouth"] = facial_structure_weight
                group_weights["jaw"] = facial_structure_weight
                group_weights["chin"] = facial_structure_weight
            if chin_weight != 1.0:
                group_weights["chin"] = group_weights.get("chin", 1.0) * chin_weight
            if eye_weight != 1.0:
                group_weights["eye"] = eye_weight
            if lip_weight != 1.0:
                group_weights["lip"] = lip_weight
            self.loss_fn = PerGeneMSELoss(
                get_lengths(), names=get_gene_names(), group_weights=group_weights or None
            )
        else:
            self.loss_fn = torch.nn.MSELoss()
        # Dedicated gender classification loss (see `GenderBCELoss`), on by
        # default since plain per-gene MSE was found to systematically
        # mispredict gender for ambiguous/small-training-set photos (the
        # gender channel's own MSE term, bounded by [0,255]^2, only gets a
        # 1/n_genes share of the averaged per-gene loss, same as every other
        # gene -- but BCE values are O(0.1-5), roughly 4-5 orders of
        # magnitude smaller than that averaged MSE term (O(1e3-1e4)), so
        # gender_weight defaults high (2000.0) to give it an outsized,
        # deliberately non-"equal-weight" share of the gradient; set
        # gender_weight=0.0 to disable and fall back to plain per-gene MSE.
        self.gender_loss = GenderBCELoss() if gender_weight > 0 else None
        # Optional secondary regularizer (see `DiversityPriorLoss`):
        # disabled by default (diversity_weight=0.0) to keep prior behavior
        # unchanged unless explicitly opted into via the CLI.
        self.diversity_loss: Optional[DiversityPriorLoss]
        if diversity_weight > 0:
            lengths = get_lengths()
            target_stds = load_diversity_prior_stds(
                diversity_prior_path or DEFAULT_DIVERSITY_PRIOR_PATH, lengths
            )
            self.diversity_loss = DiversityPriorLoss(lengths, target_stds)
        else:
            self.diversity_loss = None
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _diversity_penalty(self, preds: torch.Tensor, stage: str) -> torch.Tensor:
        if self.diversity_loss is None:
            return preds.new_zeros(())
        penalty = self.diversity_loss(preds)
        self.log(f"{stage}_diversity_penalty", penalty, prog_bar=False, on_step=False, on_epoch=True)
        return penalty

    def _gender_penalty(self, preds: torch.Tensor, targets: torch.Tensor, stage: str) -> torch.Tensor:
        if self.gender_loss is None:
            return preds.new_zeros(())
        penalty = self.gender_loss(preds, targets)
        self.log(f"{stage}_gender_bce", penalty, prog_bar=False, on_step=False, on_epoch=True)
        return penalty

    def training_step(self, batch, batch_idx):
        images, targets = batch
        preds = self(images)
        loss = (
            self.loss_fn(preds, targets)
            + self.hparams.diversity_weight * self._diversity_penalty(preds, "train")
            + self.hparams.gender_weight * self._gender_penalty(preds, targets, "train")
        )
        self.training_step_outputs.append(loss.detach())
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        preds = self(images)
        loss = (
            self.loss_fn(preds, targets)
            + self.hparams.diversity_weight * self._diversity_penalty(preds, "val")
            + self.hparams.gender_weight * self._gender_penalty(preds, targets, "val")
        )
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
        self.net = Net(
            lr=args.lr,
            diversity_weight=getattr(args, "diversity_weight", 0.0),
            diversity_prior_path=getattr(args, "diversity_prior", None),
            gender_weight=getattr(args, "gender_weight", 300.0),
            facial_structure_weight=getattr(args, "facial_structure_weight", 1.0),
            chin_weight=getattr(args, "chin_weight", 1.0),
            eye_weight=getattr(args, "eye_weight", 1.0),
            lip_weight=getattr(args, "lip_weight", 1.0),
        )
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=args.val_epochs),
            # Without an explicit monitored ModelCheckpoint, Lightning's default
            # checkpointing behavior only saves the *last* epoch seen before
            # training stops -- not the epoch with the lowest val_loss. On this
            # dataset's small size, val_loss is volatile late in training (it
            # can drift back up during the EarlyStopping patience window before
            # the patience counter finally expires), so the "last" epoch can be
            # measurably worse than the best one seen a few epochs earlier.
            # Saving top_k=1 by val_loss ensures the checkpoint actually used
            # for inference is the best one, not whatever epoch happened to be
            # current when patience ran out.
            ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, filename="best-{epoch}-{val_loss:.4f}"),
        ]
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
