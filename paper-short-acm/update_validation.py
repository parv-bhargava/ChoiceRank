"""Refresh validation.json after rebuilding the PDFs: page counts and SHA-256 hashes of the sources and outputs.

Run from the repository root after `tectonic main.tex` in both paper/ and paper-short-acm/:
    python paper-short-acm/update_validation.py
Fields not listed in HASHED are kept as they are.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper-short-acm" / "validation.json"
HASHED = {
    "main_tex_sha256": "paper-short-acm/main.tex",
    "main_pdf_sha256": "paper-short-acm/main.pdf",
    "longer_main_pdf_sha256": "paper/main.pdf",
    "chart_script_sha256": "paper-short-acm/make_charts.py",
    "clean_analysis_script_sha256": "scripts/musique_clean_subset.py",
    "overlap_audit_sha256": "results/musique_split_overlap_audit.json",
    "clean_metadata_sha256": "results/musique_clean_subset_metadata.json",
}


def sha256(path: Path) -> str:
    data = path.read_bytes()
    if path.suffix != ".pdf":  # hash text as committed (LF), so Windows CRLF checkouts give the same value
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def main():
    v = json.loads(OUT.read_text(encoding="utf-8"))
    v["pdf_pages"] = len(PdfReader(ROOT / "paper-short-acm" / "main.pdf").pages)
    v["longer_pdf_pages"] = len(PdfReader(ROOT / "paper" / "main.pdf").pages)
    for key, rel in HASHED.items():
        v[key] = sha256(ROOT / rel)
    OUT.write_text(json.dumps(v, indent=2) + "\n", encoding="utf-8")
    print(f"updated {OUT.relative_to(ROOT)}: {v['pdf_pages']} + {v['longer_pdf_pages']} pages")


if __name__ == "__main__":
    main()
