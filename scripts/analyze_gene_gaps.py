"""Finds under-represented value ranges ("gaps") for every renderable gene
in the real training set, so `generate_synthetic_dataset.py` can target
synthetic samples at those specific gaps instead of resampling uniformly
at random.

Why: plain Gaussian(128, 55) resampling (the prior default) spreads new
synthetic samples across the whole 0-255 range in proportion to a generic
bell curve, regardless of where the *real* 106-sample training set already
has (or lacks) coverage. If the real set happens to already cover the
middle of a gene's range well but has almost no samples near one extreme,
blanket Gaussian resampling wastes most of its budget re-covering the
already-dense middle instead of filling the actual gap -- and, per the
dilution findings in README.md, every extra synthetic sample has a real
cost (diluting the real-photo-correlated gradient signal), so sampling
efficiency matters.

For each gene this pipeline can render (see
`generate_synthetic_dataset.build_gene_index`), this script:

1. Decodes every real sample in `real_dataset_aligned`'s DNA label back to
   an approximate 0-255 slider value (the exact inverse of
   `generate_synthetic_dataset.encode_attr_value`).
2. Buckets those values into `--bins` equal-width bins across [0, 255].
3. Assigns each bin an inverse-frequency sampling weight (rarer bins get
   proportionally more weight), written to `--output` as JSON for
   `generate_synthetic_dataset.py --gap_bins <path>` to consume.

Usage (from the repo root)::

    python scripts/analyze_gene_gaps.py --datastore real_dataset_aligned --bins 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ck3dna.ck3_format import parse_dna_file  # noqa: E402
from ck3dna.gene_dicts import DEFAULT_GENE_DICTS_PATH, load_gene_dicts  # noqa: E402

from generate_synthetic_dataset import build_gene_index  # noqa: E402


def decode_attr_value(options, category: str, weight: int) -> float:
    """Inverse of `generate_synthetic_dataset.encode_attr_value`: recover
    an approximate 0-255 slider value from a real (category, weight) pair.
    """
    pos = [o for o in options if o.endswith("_pos")]
    neg = [o for o in options if o.endswith("_neg")]
    if len(options) == 2 and len(pos) == 1 and len(neg) == 1:
        if category == pos[0]:
            return 128.0 + (weight / 255.0) * 127.0
        if category == neg[0]:
            return 128.0 - (weight / 255.0) * 128.0
        return 128.0
    if category in options:
        idx = options.index(category)
        # Center of the bucket `encode_attr_value` would have mapped to
        # this index: idx = value * len // 256 -> value in
        # [idx*256/len, (idx+1)*256/len).
        lo = idx * 256.0 / len(options)
        hi = (idx + 1) * 256.0 / len(options)
        return (lo + hi) / 2.0
    return 128.0


def collect_real_values(datastore: Path, gene_index: dict) -> dict:
    """attr -> list of decoded 0-255 values, one per real sample that has
    that gene present in its DNA label."""
    values = {attr: [] for attr in gene_index}
    dna_files = []
    for portrait_dir in sorted(p for p in datastore.iterdir() if p.is_dir()):
        dna_files.extend(sorted(portrait_dir.glob("dna#*.txt")))
        dna_files.extend(sorted(portrait_dir.glob("dna#*.ck3ruler")))
    for dna_path in dna_files:
        try:
            parsed = parse_dna_file(dna_path)
        except Exception:  # noqa: BLE001 - skip unparseable/legacy files
            continue
        parsed_genes = parsed["genes"]
        for attr, (gene_key, options) in gene_index.items():
            pairs = parsed_genes.get(gene_key)
            if not pairs:
                continue
            # Real exports blend two (category, weight) pairs; take the
            # dominant (higher-weight) one as this sample's value.
            category, weight = max(pairs, key=lambda cw: cw[1])
            values[attr].append(decode_attr_value(options, category, weight))
    return values


def compute_gap_bins(values: list, num_bins: int, smoothing: float = 1.0) -> dict:
    """Equal-width [0, 255] bins + inverse-frequency sampling weights."""
    edges = [i * 256.0 / num_bins for i in range(num_bins + 1)]
    counts = [0] * num_bins
    for v in values:
        idx = min(num_bins - 1, max(0, int(v / 256.0 * num_bins)))
        counts[idx] += 1
    inv = [1.0 / (c + smoothing) for c in counts]
    total = sum(inv)
    weights = [w / total for w in inv]
    return {"edges": edges, "counts": counts, "weights": weights}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datastore", type=Path, default=REPO_ROOT / "real_dataset_aligned")
    ap.add_argument("--bins", type=int, default=10, help="Number of equal-width value bins per gene")
    ap.add_argument("--output", type=Path, default=REPO_ROOT / "scripts" / "gene_gap_bins.json")
    args = ap.parse_args(argv)

    genes = load_gene_dicts(DEFAULT_GENE_DICTS_PATH)
    gene_index = build_gene_index(genes)
    real_values = collect_real_values(args.datastore, gene_index)

    gap_bins = {}
    worst_gaps = []
    for attr, values in real_values.items():
        info = compute_gap_bins(values, args.bins)
        gap_bins[attr] = info
        empty_bins = sum(1 for c in info["counts"] if c == 0)
        worst_gaps.append((empty_bins, attr, info["counts"]))

    args.output.write_text(json.dumps(gap_bins, indent=2), encoding="utf-8")

    worst_gaps.sort(reverse=True)
    print(f"Analyzed {len(gene_index)} renderable genes across {args.datastore}")
    print(f"Wrote per-gene gap bins to {args.output}")
    print("\nGenes with the most empty bins (biggest coverage gaps):")
    for empty_bins, attr, counts in worst_gaps[:20]:
        print(f"  {attr:35s} empty_bins={empty_bins}/{args.bins}  counts={counts}")


if __name__ == "__main__":
    main()
