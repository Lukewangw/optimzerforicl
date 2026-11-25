"""Where data/vit_page_data.json comes from.

usage: python extract_data.py RESULTS_PAGE.html [OUT.json]      (default OUT: data/vit_page_data.json)

The ViT study keeps its analysis in one results page, an HTML file whose charts are all drawn from a single
block of numbers (`const DATA = {...};`). The study's own scripts compute that block from the per-run read-outs:
validation accuracy, the few-shot tests at every checkpoint and after every layer, and the weight spectra.
This script copies the block out of the page into a JSON file without changing a number, and stops if the block
holds anything that looks like a file path, an address or a user name.

The per-run read-outs (156 MB), the checkpoints, the results page and the scripts that compute its numbers are
not in this repository, so this script documents the provenance of the data file; the figures themselves need
only the data file.
"""
import json
import os
import sys

MARK = "const DATA = "
FORBIDDEN = ("/mnt", "/home", "/users", "http", "@", "\\\\")


def main(page, out):
    with open(page, encoding="utf-8") as f:
        src = f.read()
    if src.count(MARK) != 1:
        raise SystemExit(f"extract_data: {page} must hold `{MARK}` exactly once")
    data, _ = json.JSONDecoder().raw_decode(src[src.index(MARK) + len(MARK):])
    blob = json.dumps(data, separators=(",", ":"))
    found = [w for w in FORBIDDEN if w in blob.lower()]
    if found:
        raise SystemExit(f"extract_data: the block holds {found}; not written")
    with open(out, "w") as f:
        f.write(blob)
    print(f"extract_data: {len(data)} groups of numbers, {len(blob) // 1024} KB -> {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    here = os.path.dirname(os.path.abspath(__file__))
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "data", "vit_page_data.json"))
