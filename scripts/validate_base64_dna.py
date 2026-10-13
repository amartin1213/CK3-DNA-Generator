"""scripts/validate_base64_dna.py

Validate `ck3dna.dna_base64.decode_base64_dna` against a known-correct
ground-truth pair: a real base64 DNA string (e.g. from CK3's debug
``portrait_editor`` "Copy DNA string" feature, or a save file's
``dna="..."`` field) and the matching plaintext `.ck3ruler` /
``ruler_designer_<id>=`` export for the *same character*.

This is the tool referenced in the base64 reverse-engineering work: since
no ground-truth sample was available when the decoder was first written,
this script exists so one can be checked the moment it is obtained,
without needing another round of development.

Usage (from the repo root)::

    python scripts/validate_base64_dna.py --base64 "Hfwd/Jlmm..." --dna-file real_dna_samples/rulers/Some Ruler.ck3ruler
    python scripts/validate_base64_dna.py --base64-file string.txt --dna-file pair1_dna.txt --mod NONE

It reports, per gene:
* MATCH       -- both category choices and both blend values agree.
* VALUE DIFF  -- categories agree but a blend value differs.
* CATEGORY DIFF -- a category choice differs (most informative for
  diagnosing gene-order/category-index table mistakes).
* MISSING     -- the gene is absent from one side (e.g. a gene this
  codec's tables don't know about -- see `ck3dna.dna_base64`'s module
  docstring for the ~75 known-missing newer/modded genes).

A non-zero exit code is returned if any mismatches are found, so this can
also be used as a quick pass/fail smoke test once a real sample arrives.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ck3dna.ck3_format import parse_dna_file  # noqa: E402
from ck3dna.dna_base64 import decode_base64_dna  # noqa: E402


def _read_base64(args: argparse.Namespace) -> str:
    if args.base64:
        return args.base64.strip()
    return Path(args.base64_file).read_text(encoding="utf-8").strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base64", help="The raw base64 DNA string.")
    parser.add_argument("--base64-file", help="Path to a text file containing the base64 DNA string.")
    parser.add_argument("--dna-file", required=True, help="Path to the matching .ck3ruler / ruler_designer_<id>=.txt export.")
    parser.add_argument("--mod", default="NONE", choices=["NONE", "EPE"], help="Which gene order to decode with (default: NONE).")
    args = parser.parse_args()

    if not args.base64 and not args.base64_file:
        parser.error("Provide --base64 or --base64-file.")

    base64_string = _read_base64(args)
    decoded = decode_base64_dna(base64_string, mod=args.mod)
    expected = parse_dna_file(args.dna_file)

    decoded_genes = decoded["genes"]
    expected_genes = expected["genes"]

    all_names = sorted(set(decoded_genes) | set(expected_genes))
    mismatches = 0
    matches = 0
    for name in all_names:
        d = decoded_genes.get(name)
        e = expected_genes.get(name)
        if d is None or e is None:
            print(f"MISSING        {name}: base64={d!r} dna_file={e!r}")
            mismatches += 1
            continue
        if d == e:
            matches += 1
            continue
        # Categorical genes: list of (category, value) tuples.
        if isinstance(d, list) and d and isinstance(d[0], tuple):
            d_cats = [c for c, _ in d]
            e_cats = [c for c, _ in e]
            if d_cats != e_cats:
                print(f"CATEGORY DIFF  {name}: base64={d} dna_file={e}")
            else:
                print(f"VALUE DIFF     {name}: base64={d} dna_file={e}")
        else:
            print(f"VALUE DIFF     {name}: base64={d} dna_file={e}")
        mismatches += 1

    print(f"\n{matches} matched, {mismatches} mismatched, out of {len(all_names)} total genes.")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
