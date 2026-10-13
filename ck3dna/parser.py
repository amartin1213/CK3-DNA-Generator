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
    return parser


if __name__ == "__main__":
    build_arg_parser().parse_args()
