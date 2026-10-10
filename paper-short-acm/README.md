# Four-page ACM submission draft

This directory is independent of the original `paper/` manuscript. The compiled draft has four pages including references, using the same ACM `sigconf,nonacm` class as the original, with standard body fonts and margins. The original manuscript and PDF have not been edited.

Files:

- `main.tex`: complete short-paper source, including three chart panels and one detailed RAG table.
- `refs.bib`: independent copy of the original bibliography; only cited entries appear in the PDF.
- `main.pdf`: compiled submission draft.
- `original-preservation.json`: SHA-256 record of all 27 original paper files before drafting.
- `make_charts.py`: regenerates the three panels from existing result CSVs without model inference.
- `figures/results-overview.pdf` and `.png`: vector publication figure and preview.
- `figures/chart-data.json`: plotted values, confidence intervals, and source-file hashes.
- `validation.json`: page-count and original-file preservation checks.

Build from this directory with `tectonic main.tex`. On this machine:

```powershell
& 'C:\Users\parvb\tools\tectonic\tectonic.exe' main.tex
```

The figure PDFs are bundled, so LaTeX compilation needs no access to the original paper or cached results. To regenerate the charts with the project Python environment, run `python make_charts.py` while the existing `../results/` CSVs are available, then rebuild LaTeX. The panels show equal-weight ranking means, raw versus numeric-group transfer ECE, and top-3 paired Jev--bge RAG differences with ordinary 95% confidence intervals. Their caption distinguishes these intervals from Holm-adjusted significance tests.

The draft preserves all three research questions: reranking effectiveness, judged-pair calibration and source-label transfer, and downstream RAG context selection. It includes model/readout definitions, dataset and policy details, strong ranking baselines, same-base and equal-context controls, small-sample statistical sensitivity, multi-generator RAG results, operational cost, and the original study's key limitations. Detailed per-collection sweeps remain in the longer original.

Numbers were checked against the original manuscript and cached data. The short version clarifies that 308 DL plus 1,525 BEIR topics make 1,833 total; 15 NFCorpus topics have no BM25 candidate passages and score zero. No inference experiments were rerun.

This uses the existing named-author ACM draft format. Before submitting to a particular venue, apply its required class version, anonymous-review settings, and front-matter instructions. A public link to reproduction materials has not been invented; add the actual released artifact location when available. No additional RAG result against Qwen8B is claimed.
