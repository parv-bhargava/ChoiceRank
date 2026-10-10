# Four-page ACM submission draft

This directory is independent of the longer `paper/` manuscript. The compiled draft has four pages including references, using the same ACM `sigconf,nonacm` class, with standard body fonts and margins. The original was preserved when the short draft was created. On 2026-10-10, the user requested the MuSiQue overlap correction and clean-subset result; that correction was applied to both versions. Copies from immediately before this correction are retained in `C:\Users\parvb\AppData\Local\Temp\choicerank-before-overlap-correction-20261010`.

Files:

- `main.tex`: complete short-paper source, including three chart panels and one detailed RAG table.
- `refs.bib`: independent copy of the original bibliography; only cited entries appear in the PDF.
- `main.pdf`: compiled submission draft.
- `original-preservation.json`: SHA-256 record of all 27 original paper files before drafting.
- `make_charts.py`: regenerates the three panels from existing result CSVs without model inference.
- `figures/results-overview.pdf` and `.png`: vector publication figure and preview.
- `figures/chart-data.json`: plotted values, confidence intervals, and source-file hashes.
- `validation.json`: current page count, source hashes, and correction provenance; the initial preservation record remains separate.

Build from this directory with `tectonic main.tex`. On this machine:

```powershell
& 'C:\Users\parvb\tools\tectonic\tectonic.exe' main.tex
```

The figure PDFs are bundled, so LaTeX compilation needs no access to the original paper or cached results. To regenerate the charts with the project Python environment, run `python make_charts.py` while the existing `../results/` CSVs are available, then rebuild LaTeX. The panels show equal-weight ranking means, raw versus numeric-group transfer ECE, and top-3 paired Jev--bge RAG differences on the original 1,000-question splits with ordinary 95% confidence intervals. The caption states that the original question-level tests ignore MuSiQue family dependence. The table separately adds the 329-question subset; its intervals are reported in the text.

The draft preserves all three research questions: reranking effectiveness, judged-pair calibration and source-label transfer, and downstream RAG context selection. It includes model/readout definitions, dataset and policy details, strong ranking baselines, same-base and equal-context controls, small-sample statistical sensitivity, multi-generator RAG results, operational cost, and the original study's key limitations. Detailed per-collection sweeps remain in the longer original.

Numbers were checked against the longer manuscript and cached data. The short version clarifies that 308 DL plus 1,525 BEIR topics make 1,833 total; 15 NFCorpus topics have no BM25 candidate passages and score zero. No inference experiments were rerun.

The decomposition audit found 494/349 final questions sharing their terminal sub-question ID with pilot/calibration (578 either), and 671 sharing any decomposition ID with either. The retrospective sensitivity uses the other 329 questions, retaining all calibration fits, readouts, policies, and cached generated answers. Top-3 Jev/bge F1 is 43.6/41.3, 41.1/35.7, and 48.1/42.4 for Qwen2.5-7B, Llama-3.1-8B, and Qwen3-30B-A3B. None passes the original nine-policy question-level Holm correction. Family intervals resample 212 subset-bearing connected components defined over all 2,417 source questions (20,000 paired replicates, seed 13), preserve question weights, and are unadjusted for multiplicity. The subset excludes direct component reuse but 23 questions remain indirectly linked to design families; it is not a new confirmatory or fully family-disjoint evaluation.

From the repository root, run `python scripts/audit_musique_split_overlap.py` followed by `python scripts/musique_clean_subset.py`. The latter writes membership, all-policy summaries, paired tests, family intervals, and input hashes under `results/musique_clean_subset_*`, plus the longer paper's generated sensitivity table. The short source contains the compact rounded table rows for portability.

This uses the existing named-author ACM draft format. Before submitting to a particular venue, apply its required class version, anonymous-review settings, and front-matter instructions. A public link to reproduction materials has not been invented; add the actual released artifact location when available. No additional RAG result against Qwen8B is claimed.
