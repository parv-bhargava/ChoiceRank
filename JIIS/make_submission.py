"""Build the Journal of Intelligent Information Systems (Springer) upload files from JIIS/ChoiceRank.

python JIIS/make_submission.py [--tectonic PATH] -> JIIS/dist/
  jiis_manuscript.tex   one flattened source file: sections and tables inlined, bibliography pasted in from the .bbl
  sn-jnl.cls, sn-basic.bst, refs.bib, Fig1.pdf   the other source files, all at the top level (no subfolders)
  latex_source.zip      the files above in one archive
  jiis_manuscript.pdf   compiled from the flattened source
  ESM_1.pdf             Online Resource 1 (supplementary tables and analyses)

Springer asks for one .tex file without \\input, no subfolders, and figures named Fig1, Fig2, ...
Exits with status 1 if the flattened source does not compile cleanly or the manuscript exceeds the 25-page limit.
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
MAIN = "jiis_manuscript.tex"
FIGURES = {"figures/reliability.pdf": "Fig1.pdf"}
PAGE_LIMIT = 25  # JIIS: including references, tables, and figures
INPUT = re.compile(r"\\input\{([^}]+)\}")


def inline(path):
    text = path.read_text(encoding="utf-8")
    return INPUT.sub(lambda m: inline(SRC / (m.group(1) if m.group(1).endswith(".tex") else m.group(1) + ".tex"))
                     .rstrip("\n"), text)


def compile_tex(tex, cwd, engine, keep=False):
    out = cwd / "build"
    out.mkdir(exist_ok=True)
    if Path(engine).name.lower().startswith("tectonic"):
        cmd = [engine, "--keep-logs", *(["--keep-intermediates"] if keep else []), "-o", str(out), tex]
    else:
        cmd = [engine, "-pdf", "-interaction=nonstopmode", f"-outdir={out}", tex]
    subprocess.run(cmd, cwd=cwd, check=True, capture_output=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tectonic", help="path to tectonic (default: tectonic or latexmk on PATH)")
    args = parser.parse_args()
    engine = args.tectonic or shutil.which("tectonic") or shutil.which("latexmk")
    if not engine:
        sys.exit("needs tectonic or latexmk to resolve the bibliography")
    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir()

    # the .bbl from a build of the unflattened source, pasted in place of \bibliography (Springer's recommendation)
    bbl = (compile_tex(MAIN, SRC, engine, keep=True) / MAIN.replace(".tex", ".bbl")).read_text(encoding="utf-8")
    shutil.rmtree(SRC / "build")
    text = inline(SRC / MAIN)
    text = text.replace("\\bibliography{refs}", bbl.strip())
    for old, new in FIGURES.items():
        text = text.replace(old, new)
    if INPUT.search(text) or "\\bibliography{" in text or re.search(r"\\includegraphics(\[[^]]*\])?\{[^}]*/", text):
        sys.exit("flattening left an \\input, \\bibliography or subfolder path behind")
    (DIST / MAIN).write_text(text, encoding="utf-8")
    for name in ("sn-jnl.cls", "sn-basic.bst", "refs.bib"):
        shutil.copy2(SRC / name, DIST / name)
    for old, new in FIGURES.items():
        shutil.copy2(SRC / old, DIST / new)
    sources = [MAIN, "sn-jnl.cls", "sn-basic.bst", "refs.bib", *FIGURES.values()]
    with zipfile.ZipFile(DIST / "latex_source.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for name in sources:
            z.write(DIST / name, name)
    print(f"wrote {DIST / 'latex_source.zip'} ({len(sources)} files, no subfolders)")

    problems = []
    build = compile_tex(MAIN, DIST, engine)
    shutil.copy2(build / MAIN.replace(".tex", ".pdf"), DIST / MAIN.replace(".tex", ".pdf"))
    log = (build / MAIN.replace(".tex", ".log")).read_text(encoding="utf-8", errors="replace")
    problems += [f"{MAIN}: {line.strip()}" for line in log.splitlines()
                 if "undefined" in line.lower() and ("Citation" in line or "Reference" in line)]
    supp = compile_tex("supplementary.tex", SRC, engine)
    shutil.copy2(supp / "supplementary.pdf", DIST / "ESM_1.pdf")
    shutil.rmtree(supp)
    shutil.rmtree(build)

    import pymupdf  # only needed to check compiled PDFs
    for pdf in (MAIN.replace(".tex", ".pdf"), "ESM_1.pdf"):
        with pymupdf.open(DIST / pdf) as doc:
            text = "".join(page.get_text() for page in doc)
            print(f"built {DIST / pdf}: {len(doc)} pages")
            if "??" in text:
                problems.append(f"{pdf}: unresolved reference (??)")
            if pdf == MAIN.replace(".tex", ".pdf") and len(doc) > PAGE_LIMIT:
                problems.append(f"{pdf}: {len(doc)} pages, over the {PAGE_LIMIT}-page JIIS limit")
    if problems:
        print("\n".join(problems), file=sys.stderr)
        sys.exit(1)
    print("submission checks passed")


if __name__ == "__main__":
    main()
