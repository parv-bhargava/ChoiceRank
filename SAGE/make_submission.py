"""Build the JIS (SAGE) upload files from SAGE/ChoiceRank and check that the manuscript is anonymous.

python SAGE/make_submission.py [--tectonic PATH] -> SAGE/dist/
  manuscript_source.zip  anonymized LaTeX source (sent to reviewers): everything except the title page
  title_page.tex         identifying details (not sent to reviewers), uploaded on its own
  *.pdf                  compiled documents, when tectonic or latexmk is available

Exits with status 1 if a manuscript source file or compiled PDF names an author, an affiliation or the repository.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "ChoiceRank"
DIST = HERE / "dist"
TITLE_PAGE = "title_page.tex"
IDENTIFYING = re.compile(r"Bhargava|Jafari|Aihan|\bParv\b|parv-bhargava|George Washington|\bGWU\b|Northwestern"
                         r"|zenodo", re.IGNORECASE)


def manuscript_files():
    return [p for p in sorted(SRC.rglob("*")) if p.is_file() and p.name != TITLE_PAGE]


def leaks(name, text):
    return [f"{name}: {m.group(0)!r} in ...{text[max(0, m.start() - 40):m.end() + 40]!r}..."
            for m in IDENTIFYING.finditer(text)]


def build(tex, engine):
    out = DIST / "build"
    out.mkdir(parents=True, exist_ok=True)
    if Path(engine).name.lower().startswith("tectonic"):
        cmd = [engine, "--keep-logs", "-o", str(out), tex]
    else:
        cmd = [engine, "-pdf", "-interaction=nonstopmode", f"-outdir={out}", tex]
    subprocess.run(cmd, cwd=SRC, check=True, capture_output=True)
    pdf = out / (Path(tex).stem + ".pdf")
    shutil.copy2(pdf, DIST / pdf.name)
    return DIST / pdf.name


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tectonic", help="path to tectonic (default: tectonic or latexmk on PATH)")
    args = parser.parse_args()
    DIST.mkdir(exist_ok=True)

    problems = []
    files = manuscript_files()
    for p in files:
        if p.suffix in (".tex", ".bib", ".cls", ".bst"):
            problems += leaks(p.relative_to(SRC).as_posix(), p.read_text(encoding="utf-8"))

    with zipfile.ZipFile(DIST / "manuscript_source.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(SRC).as_posix())
    shutil.copy2(SRC / TITLE_PAGE, DIST / TITLE_PAGE)
    print(f"wrote {DIST / 'manuscript_source.zip'} ({len(files)} files) and {DIST / TITLE_PAGE}")

    engine = args.tectonic or shutil.which("tectonic") or shutil.which("latexmk")
    if engine:
        documents = ["saga_manuscript.tex", "supplementary.tex", TITLE_PAGE]
        for tex in [d for d in documents if (SRC / d).exists()]:
            pdf = build(tex, engine)
            print(f"built {pdf}")
            if tex != TITLE_PAGE:
                import pymupdf  # only needed to check compiled PDFs
                with pymupdf.open(pdf) as doc:
                    text = "".join(page.get_text() for page in doc) + " ".join(str(v) for v in doc.metadata.values())
                problems += leaks(pdf.name, text)
    else:
        print("no tectonic or latexmk found: PDFs not built")

    if problems:
        print("identifying information in the anonymized manuscript:", *problems, sep="\n  ")
        sys.exit(1)
    print("anonymity check passed")


if __name__ == "__main__":
    main()
