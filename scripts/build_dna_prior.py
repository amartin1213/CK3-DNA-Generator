"""Build a per-gene diversity prior from a folder of genuine CK3 Ruler
Designer ``.ck3ruler`` character files, independent of (and typically much
larger than) the project's small photo-paired training set.

This intentionally stores ONLY the resulting aggregate per-gene standard
deviations (see ``ck3dna/dependencies/dna_prior_std.json``), never the
source character files or their names -- those are user-authored
likenesses of copyrighted fictional characters/actors not meant for
redistribution. Only the statistical summary is checked into the repo, and
is consumed by ``ck3dna.training.DiversityPriorLoss`` as a training-time
regularizer that discourages the model's predictions from collapsing to
near-identical output ("every face looks the same") by requiring each
gene's batch-wise spread to be at least as wide as real CK3 characters
typically exhibit.

Usage::

    python scripts/build_dna_prior.py --rulers_dir "C:\\...\\rulers" \\
        --out ck3dna/dependencies/dna_prior_std.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ck3dna.dna_parser import dna_ToTensor
from ck3dna.gene_dicts import get_lengths, load_gene_dicts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rulers_dir", required=True, help="Folder containing .ck3ruler files")
    ap.add_argument("--out", default="ck3dna/dependencies/dna_prior_std.json")
    args = ap.parse_args()

    files = sorted(Path(args.rulers_dir).glob("*.ck3ruler"))
    if not files:
        raise SystemExit(f"No .ck3ruler files found under {args.rulers_dir}")

    tensors = []
    failed = []
    for f in files:
        try:
            tensors.append(dna_ToTensor(f))
        except Exception as e:  # noqa: BLE001 - report and continue
            failed.append((f.name, str(e)))

    if failed:
        print(f"Skipped {len(failed)} file(s) that failed to parse:")
        for name, err in failed:
            print(f"  {name}: {err}")

    if not tensors:
        raise SystemExit("No files parsed successfully; nothing to build.")

    mat = torch.stack(tensors)
    print(f"Loaded {mat.shape[0]} real DNA samples ({mat.shape[1]}-dim each)")

    gene_dicts = load_gene_dicts()
    lengths = get_lengths()
    names = ["gender", "hair_color", "skin_color", "eye_color"] + list(gene_dicts.keys())

    offset = 0
    per_gene_std = {}
    for name, length in zip(names, lengths):
        seg = mat[:, offset : offset + length]
        offset += length
        per_gene_std[name] = seg.std(dim=0).mean().item()

    out_data = {
        "_meta": {
            "sample_count": mat.shape[0],
            "skipped_count": len(failed),
            "description": (
                "Per-gene mean standard deviation across a corpus of genuine "
                "CK3 Ruler Designer DNA exports, used as a diversity floor "
                "target during training (see "
                "ck3dna.training.DiversityPriorLoss). Source character files "
                "are NOT included/redistributed, only this aggregate "
                "statistic."
            ),
        },
        "per_gene_std": per_gene_std,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_data, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} ({len(per_gene_std)} genes)")


if __name__ == "__main__":
    main()
