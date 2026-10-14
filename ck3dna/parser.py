"""ck3dna.parser

Shared argparse CLI definition for the training entry point
(``python -m ck3dna.training``).
"""
from __future__ import annotations

import argparse


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the CK3 DNA prediction model")
    parser.add_argument(
        "--datastore_name",
        type=str,
        required=True,
        help="Path to the preprocessed dataset (see ck3dna.preprocess_tensors)",
    )
    parser.add_argument("--workers", type=int, default=0, help="DataLoader worker processes")
    parser.add_argument("--epochs", type=int, default=100, help="Max training epochs")
    parser.add_argument(
        "--val_epochs",
        type=int,
        default=5,
        help="EarlyStopping patience, in validation epochs without improvement",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-3,
        help=(
            "Learning rate for the RMSprop optimizer. Empirically, 1e-3 "
            "(the original default) converges too slowly within a "
            "practical epoch budget on small real-world datasets (tens to "
            "~100 samples) and the model collapses to near-constant, "
            "low-diversity predictions across different input photos; "
            "3e-3 converged meaningfully further within the same epoch "
            "budget in that regime -- see README's 'Training notes: output "
            "diversity' section."
        ),
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.2, help="Fraction of the dataset reserved for validation"
    )
    parser.add_argument(
        "--gpus", type=int, default=0, help="Number of GPUs to use (0 = CPU / let Lightning auto-detect)"
    )
    parser.add_argument("--ckpt", type=str, default=None, help="Optional checkpoint path to resume training from")
    parser.add_argument(
        "--diversity_weight",
        type=float,
        default=0.0,
        help=(
            "Weight for the DiversityPriorLoss regularizer (0 = disabled, "
            "matching prior behavior). Penalizes a training batch's "
            "predictions for having less per-gene spread than real CK3 "
            "characters typically do, using an independent reference "
            "corpus (see scripts/build_dna_prior.py) far larger than the "
            "small photo-paired training set -- a secondary defense "
            "against the model collapsing to near-identical output. "
            "Try a small value such as 0.05-0.2."
        ),
    )
    parser.add_argument(
        "--diversity_prior",
        type=str,
        default=None,
        help=(
            "Path to a diversity prior JSON built by "
            "scripts/build_dna_prior.py. Defaults to the checked-in "
            "ck3dna/dependencies/dna_prior_std.json."
        ),
    )
    parser.add_argument(
        "--gender_weight",
        type=float,
        default=300.0,
        help=(
            "Weight for the dedicated GenderBCELoss term on the DNA tensor's "
            "binary gender channel (0 = disabled, falling back to treating "
            "gender as a plain per-gene MSE regression target like every "
            "other gene). BCE values are several orders of magnitude smaller "
            "than the averaged per-gene MSE loss, so a small weight would "
            "leave it negligible -- but an initial 2000.0 default was found "
            "empirically to be *too* dominant: it meaningfully regressed "
            "overall prediction diversity (0.380->0.218 on the real104 "
            "diversity-ratio check), hitting mouth/chin genes hardest since "
            "their small (1-3 dim) segments get the least gradient signal "
            "relative to gender's now-oversized term. 300.0 recovers most of "
            "that diversity (ratio 0.309) for a small (~5pp) full-dataset "
            "gender-accuracy tradeoff; combine with --diversity_weight "
            "0.2-0.3 to recover further."
        ),
    )
    parser.add_argument(
        "--facial_structure_weight",
        type=float,
        default=1.0,
        help=(
            "Extra multiplier (applied within PerGeneMSELoss) for every "
            "gene whose name contains 'mouth', 'jaw', or 'chin' (1.0 = "
            "disabled, same equal per-gene weighting as every other gene). "
            "analyze_gene_variance.py found these specific gene groups stay "
            "collapsed -- predictions clustering near a fixed blend "
            "regardless of the input photo -- worse than other genes of "
            "similar segment size, reported in practice as puffy/overfull "
            "lips, a missing jawline, and chin height not tracking the "
            "photo. Try 2.0-4.0; combine with --gender_weight and "
            "--diversity_weight as needed."
        ),
    )
    parser.add_argument(
        "--chin_weight",
        type=float,
        default=1.0,
        help=(
            "Extra multiplier (stacked on top of --facial_structure_weight) "
            "applied only to genes whose name contains 'chin' (1.0 = "
            "disabled). Added after gene_chin_height stayed one of the "
            "single most collapsed genes in the whole model (ratio ~0.245) "
            "even with a 3x --facial_structure_weight boost, while "
            "gene_jaw_height improved much more (~0.431) under the same "
            "multiplier -- chin height specifically needed more push. Try "
            "2.0-3.0 on top of --facial_structure_weight 3.0."
        ),
    )
    parser.add_argument(
        "--eye_weight",
        type=float,
        default=1.0,
        help=(
            "Multiplier applied to genes whose name contains 'eye' (1.0 = "
            "disabled, same equal per-gene weighting as every other gene). "
            "Added after a user report that eye angle/shape looked off; "
            "this gene group was not covered by --facial_structure_weight. "
            "Try 2.0-3.0."
        ),
    )
    parser.add_argument(
        "--lip_weight",
        type=float,
        default=1.0,
        help=(
            "Extra multiplier (stacked on top of --facial_structure_weight) "
            "applied only to genes whose name contains 'lip' (1.0 = "
            "disabled). Added after lip-shape sub-genes (upper/lower lip "
            "forward, profile, glamour_lips, center_curve, philtrum_shape) "
            "stayed badly collapsed (ratios ~0.10-0.31) even under the "
            "overall 3x 'mouth' boost from --facial_structure_weight -- "
            "much worse than other mouth genes like mouth_height (~0.45) "
            "under the same multiplier. Reported in practice as mismatched "
            "lip shape, overly-forward mouth corners, and a visible seam "
            "under the lower lip. Try 2.0-3.0."
        ),
    )
    return parser


if __name__ == "__main__":
    build_arg_parser().parse_args()
