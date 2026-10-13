"""Regenerate ck3dna/dependencies/gene_dicts.json and default_dna.txt from the
real CK3 Ruler Designer samples under ``real_dna_samples/``.

This is a one-off/occasional developer tool, not part of the runtime
package. Re-run it if more real `.ck3ruler` / Ruler Designer export samples
are added later, to broaden the observed category coverage.

Usage (from the repo root):

    python scripts/build_gene_dicts.py

It overwrites:
    ck3dna/dependencies/gene_dicts.json
    ck3dna/dependencies/default_dna.txt
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ck3dna.ck3_format import COLOR_GENE_NAMES, parse_dna_file  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = REPO_ROOT / "real_dna_samples"
DEPENDENCIES_DIR = REPO_ROOT / "ck3dna" / "dependencies"


def collect_sample_files():
    rulers = sorted(glob.glob(str(SAMPLES_DIR / "rulers" / "*.ck3ruler")))
    discord = sorted(glob.glob(str(SAMPLES_DIR / "discord_pairs" / "*.txt")))
    return rulers + discord


def build_gene_dicts(files):
    """Returns (gene_order, gene_categories, samples) where `samples` is a
    list of (path, parsed_dict) used later to synthesize a default template.
    """
    gene_order = []
    gene_categories = {}
    samples = []
    for f in files:
        parsed = parse_dna_file(f)
        samples.append((f, parsed))
        for name, value in parsed["genes"].items():
            if name not in gene_order:
                gene_order.append(name)
                gene_categories[name] = [] if name not in COLOR_GENE_NAMES else None
            if name in COLOR_GENE_NAMES:
                continue
            for category, _weight in value:
                if category not in gene_categories[name]:
                    gene_categories[name].append(category)
    return gene_order, gene_categories, samples


def build_default_values(gene_order, samples):
    """Picks a concrete default value for every gene in `gene_order` by
    scanning samples in order and taking the first one that has it.
    """
    defaults = {}
    for _path, parsed in samples:
        for name, value in parsed["genes"].items():
            if name not in defaults:
                defaults[name] = value
        if len(defaults) == len(gene_order):
            break
    return defaults


def main():
    files = collect_sample_files()
    if not files:
        raise SystemExit(f"No sample files found under {SAMPLES_DIR}")

    gene_order, gene_categories, samples = build_gene_dicts(files)

    gene_dicts = {
        "_meta": {
            "source": (
                "Derived from real Crusader Kings 3 Ruler Designer exports "
                f"({len(files)} sample files: .ck3ruler save files + raw "
                "ruler_designer_<id> text exports). Category lists reflect "
                "only the values observed in these samples and may not be "
                "fully exhaustive of every value the game can produce."
            ),
            "num_samples": len(files),
            "color_genes": list(COLOR_GENE_NAMES),
        },
        "genes": {name: gene_categories[name] for name in gene_order if name not in COLOR_GENE_NAMES},
    }

    DEPENDENCIES_DIR.mkdir(parents=True, exist_ok=True)
    gene_dicts_path = DEPENDENCIES_DIR / "gene_dicts.json"
    gene_dicts_path.write_text(json.dumps(gene_dicts, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {gene_dicts_path} ({len(gene_order)} genes, "
          f"{sum(len(c) for c in gene_categories.values() if c is not None)} total categories)")

    defaults = build_default_values(gene_order, samples)

    from ck3dna.ck3_format import format_gene_block

    lines = ["ruler_designer_0={", "\ttype=boy", "\tid=0", "\trandom_seed=0", "\tgenes={"]
    for name in gene_order:
        lines.append("\t\t" + format_gene_block(name, defaults[name]))
    lines.append("\t}")
    lines.append("\toverride={")
    lines.append("\t\tportrait_modifier_overrides={")
    lines.append("\t\t}")
    lines.append("\t}")
    lines.append("\tentity={ 0 0 }")
    lines.append("}")

    default_dna_path = DEPENDENCIES_DIR / "default_dna.txt"
    default_dna_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {default_dna_path}")


if __name__ == "__main__":
    main()
