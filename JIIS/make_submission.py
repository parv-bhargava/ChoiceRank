"""Build the Journal of Intelligent Information Systems (Springer) upload files from JIIS/ChoiceRank and check that the
manuscript is anonymous.

python JIIS/make_submission.py [--tectonic PATH] -> JIIS/dist/
  jiis_manuscript.tex   one flattened source file: sections and tables inlined, bibliography pasted in from the .bbl
  sn-jnl.cls, sn-basic.bst, refs.bib   the other source files, all at the top level (no subfolders)
  latex_source.zip      the files above in one archive
  jiis_manuscript.pdf   compiled from the flattened source
  ESM_1.pdf             Online Resource 1 (supplementary tables and analyses)
  title_page.tex/.pdf   identifying details (not sent to reviewers), uploaded on their own and kept out of the zip

Springer asks for one .tex file without \\input, no subfolders, and figures named Fig1, Fig2, ...
Exits with status 1 if the flattened source does not compile cleanly, the manuscript exceeds the 25-page limit,
the zip has a subfolder, an author, affiliation or the repository appears in a file sent to reviewers,
or the title page still has a [TODO: ...] placeholder.
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
FIGURES = {}  # main-text figures as {"figures/x.pdf": "Fig1.pdf"}; the reliability figure is in Online Resource 1
PAGE_LIMIT = 25  # JIIS: including references, tables, and figures
INPUT = re.compile(r"\\input\{([^}]+)\}")
TITLE_PAGE = "title_page.tex"
# authors, affiliations and the repository; "Liu" alone would also match Liu et al. in the references
IDENTIFYING = re.compile(r"Bhargava|Jafari|Aihan|\bParv\b|parv-bhargava|George Washington|\bGWU\b|gwu\.edu|Northwestern",
                         re.I)


def identifying(text, where):
    return [f"{where}: names {m.group(0)!r}" for m in IDENTIFYING.finditer(text)]


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
    shutil.copy2(SRC / TITLE_PAGE, DIST / TITLE_PAGE)

    problems = []
    with zipfile.ZipFile(DIST / "latex_source.zip") as z:
        problems += [f"latex_source.zip: subfolder in {n}" for n in z.namelist() if "/" in n]
    problems += identifying(text, MAIN)
    problems += [f"{TITLE_PAGE}: placeholder {m}"
                 for m in re.findall(r"\[TODO:[^]]*\]", (SRC / TITLE_PAGE).read_text(encoding="utf-8"))]
    build = compile_tex(MAIN, DIST, engine)
    shutil.copy2(build / MAIN.replace(".tex", ".pdf"), DIST / MAIN.replace(".tex", ".pdf"))
    log = (build / MAIN.replace(".tex", ".log")).read_text(encoding="utf-8", errors="replace")
    problems += [f"{MAIN}: {line.strip()}" for line in log.splitlines()
                 if "undefined" in line.lower() and ("Citation" in line or "Reference" in line)]
    supp = compile_tex("supplementary.tex", SRC, engine)
    shutil.copy2(supp / "supplementary.pdf", DIST / "ESM_1.pdf")
    shutil.rmtree(supp)
    title = compile_tex(TITLE_PAGE, SRC, engine)
    shutil.copy2(title / TITLE_PAGE.replace(".tex", ".pdf"), DIST / TITLE_PAGE.replace(".tex", ".pdf"))
    shutil.rmtree(title)
    shutil.rmtree(build)

    import pymupdf  # only needed to check compiled PDFs
    for pdf in (MAIN.replace(".tex", ".pdf"), "ESM_1.pdf"):
        with pymupdf.open(DIST / pdf) as doc:
            text = "".join(page.get_text() for page in doc)
            print(f"built {DIST / pdf}: {len(doc)} pages")
            problems += identifying(text + " ".join(v for v in doc.metadata.values() if v), pdf)
            if "??" in text:
                problems.append(f"{pdf}: unresolved reference (??)")
            if pdf == MAIN.replace(".tex", ".pdf") and len(doc) > PAGE_LIMIT:
                problems.append(f"{pdf}: {len(doc)} pages, over the {PAGE_LIMIT}-page JIIS limit")
    if problems:
        print("\n".join(problems), file=sys.stderr)
        sys.exit(1)
    print("submission checks passed (anonymity, no subfolders, references, page limit, title page)")


if __name__ == "__main__":
    main()
