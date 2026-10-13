"""Diagnostic script (not part of the package): compares per-gene-segment
variance between ground-truth DNA tensors and model predictions across the
full real dataset, to find which genes the model has "collapsed" to
near-constant predictions vs. which genes it predicts with real diversity.

Usage::

    python scripts/analyze_gene_variance.py --checkpoint checkpoints/foo.ckpt \
        --datastore real_dataset_aligned
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ck3dna.dna_parser import dna_ToTensor
from ck3dna.gene_dicts import load_gene_dicts, get_lengths
from ck3dna.training import Net, _portrait_loader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--datastore", default="real_dataset_aligned")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    root = Path(args.datastore)
    samples = []
    for portrait_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        image_tensors = sorted(portrait_dir.glob("angle#*.pt"))
        dna_files = sorted(portrait_dir.glob("dna#*.txt")) + sorted(portrait_dir.glob("dna#*.ck3ruler"))
        if image_tensors and dna_files:
            samples.append((image_tensors[0], dna_files[0]))
    print(f"Found {len(samples)} samples")

    net = Net.load_from_checkpoint(args.checkpoint, map_location="cpu")
    net.eval()

    gt_tensors = []
    pred_tensors = []
    with torch.no_grad():
        for image_path, dna_path in samples:
            gt = dna_ToTensor(dna_path)
            gt_tensors.append(gt)
            image = _portrait_loader(image_path).unsqueeze(0)
            pred = net(image).squeeze(0)
            pred_tensors.append(pred)

    gt_mat = torch.stack(gt_tensors)
    pred_mat = torch.stack(pred_tensors)

    gene_dicts = load_gene_dicts()
    lengths = get_lengths()
    names = ["gender", "hair_color", "skin_color", "eye_color"] + list(gene_dicts.keys())

    offset = 0
    rows = []
    for name, length in zip(names, lengths):
        gt_seg = gt_mat[:, offset : offset + length]
        pred_seg = pred_mat[:, offset : offset + length]
        offset += length
        gt_std = gt_seg.std(dim=0).mean().item()
        pred_std = pred_seg.std(dim=0).mean().item()
        rows.append((name, length, gt_std, pred_std))

    overall_gt_std = gt_mat.std(dim=0).mean().item()
    overall_pred_std = pred_mat.std(dim=0).mean().item()
    print(f"\nOverall mean per-dim std: ground truth={overall_gt_std:.2f}  predicted={overall_pred_std:.2f}")
    print(f"Ratio (predicted/ground truth): {overall_pred_std / overall_gt_std:.3f}\n")

    # Sort by how much the model under-predicts diversity relative to the
    # genuine diversity present in the dataset for that gene.
    def collapse_ratio(row):
        _, _, gt_std, pred_std = row
        return pred_std / gt_std if gt_std > 1e-6 else float("inf")

    rows_sorted = sorted(rows, key=collapse_ratio)

    print(f"{'gene':35s} {'len':>4s} {'gt_std':>8s} {'pred_std':>9s} {'ratio':>7s}")
    print("-" * 70)
    for name, length, gt_std, pred_std in rows_sorted[: args.top]:
        ratio = pred_std / gt_std if gt_std > 1e-6 else float("inf")
        print(f"{name:35s} {length:4d} {gt_std:8.2f} {pred_std:9.2f} {ratio:7.3f}")

    print("\n...most diverse predictions (highest ratio)...\n")
    for name, length, gt_std, pred_std in rows_sorted[-args.top :]:
        ratio = pred_std / gt_std if gt_std > 1e-6 else float("inf")
        print(f"{name:35s} {length:4d} {gt_std:8.2f} {pred_std:9.2f} {ratio:7.3f}")

    # Separate shape (categorical, non-color) vs color gene stats.
    color_rows = [r for r in rows if r[0] in ("hair_color", "skin_color", "eye_color")]
    shape_rows = [r for r in rows if r[0] not in ("gender", "hair_color", "skin_color", "eye_color")]

    def agg(rows_subset):
        gt = sum(r[2] for r in rows_subset) / len(rows_subset)
        pred = sum(r[3] for r in rows_subset) / len(rows_subset)
        return gt, pred

    color_gt, color_pred = agg(color_rows)
    shape_gt, shape_pred = agg(shape_rows)
    print(f"\nColor genes:  mean gt_std={color_gt:.2f}  mean pred_std={color_pred:.2f}  ratio={color_pred/color_gt:.3f}")
    print(f"Shape genes:  mean gt_std={shape_gt:.2f}  mean pred_std={shape_pred:.2f}  ratio={shape_pred/shape_gt:.3f}")


if __name__ == "__main__":
    main()
