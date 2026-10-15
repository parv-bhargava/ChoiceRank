# ChoiceRank: Typed Decision Models as Passage Rerankers

Code for *Typed Decision Models as Passage Rerankers: Effectiveness, Calibration, and Context Selection* (Parv Bhargava, 2026).

The paper asks three questions:

- **RQ1:** Can typed decision models (Jev 1.13, Clef 27B, Jeff 0.8B, Laya 421M), used as pointwise relevance judges, match cross-encoders and label-probability LLM rerankers?
  - Tested on the BM25 top-100 candidates of TREC DL 2019–2023 and eight BEIR collections.
- **RQ2:** Are their relevance probabilities calibrated?
- **RQ3:** Are those probabilities useful for choosing which passages a RAG generator sees?
  - Tested on MuSiQue and NQ with three generators.

## What is in this repository

| Path | Contents |
|---|---|
| `choicerank/` | Data loading, question formats and readouts, metrics, and one judge class per system (`judges/`). |
| `scripts/` | Run, evaluation, and table/figure scripts (see below). |
| `results/` | Aggregate evaluation CSV/JSON files behind every table and figure; see `results/README.md`. |
| `paper/`, `paper-short-acm/` | Manuscript sources (full and four-page versions). |
| `tests/` | Unit tests for the review revisions. |

Inputs and raw run outputs are not version-controlled:
- `data/`: downloaded automatically from public sources on first use.
- `runs/`: per-pair scores, TREC runs, API caches, and generator answers. Written by the run scripts.

The Colab notebook used to time Jeff on a T4 is kept local and is not part of the release.

## Setup

Tested with Python 3.12.6 on Windows 11 with an RTX 4060 (8 GB). Any CUDA GPU works for the local models; the CPU works too, but slowly.

```bash
python -m venv .venv && . .venv/Scripts/activate        # Linux/macOS: . .venv/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128   # or the CPU wheel
pip install -r requirements-lock.txt                     # exact versions; requirements.txt has loose pins
cp .env.example .env                                     # add OPENROUTER_API_KEY for the paid systems
```

### Data, fetched on first use
- **Ranking:** BM25 top-100 runs with passage text come from [`castorini/rank_llm_data`](https://huggingface.co/datasets/castorini/rank_llm_data), the same candidates as RankGPT and RankLLM. Qrels come from [`castorini/eval`](https://github.com/castorini/eval).
- **RAG, MuSiQue-Ans dev:** from [`dgslibisey/MuSiQue`](https://huggingface.co/datasets/dgslibisey/MuSiQue).
- **RAG, NQ-open test with DPR top-20:** from the [DPR release](https://github.com/facebookresearch/DPR) (`nq-test.json.gz`).
- **Splits:** RAG splits are seeded (seed 13): calib 200 / pilot 300 / final 1,000 questions, and the final split is disjoint from the other two.

## Reproducing the paper

Run the steps in order: steps 1 and 2 write the per-pair scores to `runs/`, and step 3 turns them into every table and figure in the paper.

### 1. Local models (free, GPU)

```bash
python scripts/run_rerank.py --model bm25   --dataset all
python scripts/run_rerank.py --model minilm --dataset all
python scripts/run_rerank.py --model bge-m3 --dataset all
python scripts/run_rerank.py --model bge-m3-2048 --dataset touche news robust04
python scripts/run_rerank.py --model qwen3-reranker-0.6B --dataset all
python scripts/run_rerank.py --model qwen3-reranker-4B --dataset all
python scripts/run_rerank.py --model qwen3-reranker-8B --dataset all
python scripts/run_rerank.py --model qwen08base-yesno --dataset all
python scripts/run_rerank.py --model qwen08base-graded --dataset all
python scripts/run_rerank.py --model laya   --qtype all --dataset all
```

The 4B/8B Qwen rerankers use 4-bit NF4 weights on CUDA. The same-base controls use 448 passage tokens; decision models use 4,000 characters, so their differences are configuration comparisons, not isolated training effects. Jev's training cutoff is unknown; the two newer-content datasets below are not established post-cutoff tests.

```bash
python scripts/run_rerank.py --model minilm --dataset touche news robust04 --passage-tokens 448
python scripts/run_rerank.py --model bge-m3 --dataset touche news robust04 --passage-tokens 448
python scripts/run_rerank.py --model bm25 --dataset novel
python scripts/run_rerank.py --model bge-m3 --dataset novel
python scripts/run_rerank.py --model qwen3-reranker-8B --dataset novel
python scripts/run_listwise.py --dataset dl19
```

The published RankZephyr comparison uses published full-precision results; the local 4-bit DL19 run is a sanity check. Extra optional configurations may be evaluated with the same dataset groups.

**Jeff** runs through its reference server ([firelex/jeff](https://github.com/firelex/jeff), commit `d0173b4`):

1. Start the server with `JEFF_CHECKPOINT=Jeff-Qwen3.5-0.8B-v1.2 JEFF_DEVICE=cuda PORT=8765 jeff-serve`.
2. Then run `python scripts/run_rerank.py --model jeff-0.8b --qtype all --dataset all`.

The server handles one request at a time.

### 2. API models (paid, through OpenRouter)

Paid scripts require `--i-approve-spend`; `--max-usd` is checked after completed scoring batches or generator calls, so in-flight requests can exceed the limit. Run with `--dry-run` to estimate cost. The costs below describe the original runs, not guaranteed current endpoint prices.

| System | Command | Scope | Approx. cost |
|---|---|---|---|
| Jev 1.13 (`typesafe/jev-1.13-20260917`) | `run_rerank.py --model jev --qtype all --dataset all` | 13 collections, 3 formats per call | $6.2 |
| Jev, equal 448-token budget | `run_rerank.py --model jev --qtype all --dataset touche news robust04 --passage-tokens 448` | 3 collections | $1.4 |
| Clef 27B (`cloudflare/clef`) | `run_rerank.py --model clef --qtype all --dataset dl trec-covid robust04` | 7 collections | $11.7 |
| Qwen3-235B P(yes) | `run_rerank.py --model qwen235-yesno --dataset dl trec-covid nfcorpus robust04` | 8 collections | $2.8 |
| RQ3: Jev scoring and 3 generators | Explicit commands below | 2 × 1,000 questions | $6.9 incl. pilots |

The same-base Qwen3.8-27B P(yes) run (`--model qwen27base-yesno`) goes through Cloudflare, which limits new OpenRouter accounts to 20 requests per minute. Set `CHOICERANK_RPM=18` for it.

All judgments are cached, so interrupted runs resume where they stopped. API models can change behind a fixed name; we pin the Jev snapshot, and `runs/manifest.jsonl` records the date of every run.

Additional paid ranking configurations in the manuscript:

```bash
python scripts/run_rerank.py --model qwen235-graded --dataset dl trec-covid nfcorpus robust04 --dry-run
python scripts/run_rerank.py --model qwen27base-yesno --dataset dl19 dl20 trec-covid --dry-run
python scripts/run_rerank.py --model jev --qtype all --dataset novel --dry-run
```

Replace `--dry-run` with `--i-approve-spend --max-usd N` using your chosen dollar limit to execute paid commands. This also applies to the Jev equal-budget and Clef commands in the table. Set `CHOICERANK_RPM=18` for the Clef-base endpoint (PowerShell: `$env:CHOICERANK_RPM = '18'`).

RQ3 scoring and generation, repeated for both datasets:

```bash
python scripts/rag_score.py --model bge-m3 --dataset musique --split calib pilot final
python scripts/rag_score.py --model bge-m3 --dataset nq --split calib pilot final
python scripts/rag_score.py --model jev --dataset musique --split calib pilot final --dry-run
python scripts/rag_score.py --model jev --dataset nq --split calib pilot final --dry-run
python scripts/rag_generate.py --dataset musique --split pilot --gen qwen/qwen-2.5-7b-instruct --dry-run
python scripts/rag_generate.py --dataset nq --split pilot --gen qwen/qwen-2.5-7b-instruct --dry-run
```

Execute the Jev scoring commands with a chosen spending limit before generating answers. The pilot commands document the original design-stage evaluations; the current code includes later exploratory policies, so replaying it does not recreate the historical lock process. For the final protocol, `python scripts/rag_run_final.py --i-approve-spend --budget 6` runs both datasets and all three generators, using cached calls and allocating the remaining budget to each step. The raw-threshold additions are post-hoc. Fixed-policy RAG quality claims compare Jev with bge; Qwen8B has not been evaluated downstream.

The answer-mask perturbation uses existing gold labels and inserts an answer-location marker; it does not identify a reasoning mechanism:

```bash
python scripts/rag_score.py --model bge-m3 --dataset musique --split final --mask-answers
python scripts/rag_score.py --model jev --dataset musique --split final --mask-answers --dry-run
```

The Qwen0.6B wording diagnostic can be recomputed with `python scripts/review3_analyses.py musique_mechanism`. This command loads the local model and reruns scoring; the existing `results/r3_musique_mechanism.csv` can be inspected without recomputation. The optional future command `python scripts/rag_score.py --model qwen3-reranker-8B --dataset musique --split final` records selection scores only; it is not part of the reported RAG comparison.

### 3. Evaluate and rebuild tables and figures (free, minutes)

```bash
python scripts/eval_all.py          # results/ranking.csv, results/calibration.csv
python scripts/significance.py      # paired t-tests, bootstrap CIs, Holm per system pair
python scripts/review_analyses.py   # readouts, transfer calibration, pool bias, RAG selection, CIs
python scripts/review3_analyses.py transfer robust meta equiv sensitivity eqctx yesno_fallback incoll calib_diff
python scripts/depth_analysis.py    # reranking-depth simulation
python scripts/rag_summary.py       # RQ3 tests -> paper/tables/rag.tex
python scripts/make_tables.py && python scripts/make_appendix.py && python scripts/make_figures.py   # -> paper/tables, paper/figures
python scripts/audit_musique_split_overlap.py  # direct component reuse across existing MuSiQue splits
python scripts/musique_clean_subset.py  # cached 329-question sensitivity, family intervals, table
python scripts/reproduction_manifest.py  # full input hashes, split IDs, package and analysis provenance
python -m unittest discover -s tests -v
```

`sensitivity` writes `results/r3_meta_sensitivity.csv` and the normal-versus-mKH appendix table. `make_tables.py` writes a compact main ranking table and the full appendix table. The reliability figure includes bge, Qwen8B, and Jev/Laya Noul. `rag_summary.py` verifies the complete final question-policy grid and invalidates scores when the generation SHA-256 changes; partial runs are rejected unless `rag_eval.py --allow-partial` is explicitly used for exploratory analysis.

Build the multi-file manuscript from the `paper/` directory with `tectonic main.tex`, or upload the paper directory to Overleaf. Historical cached outputs are the basis for exact numerical verification. Earlier run manifests did not record Hugging Face weight commits, so the exact historical model revisions cannot be recovered from architecture strings. The reproduction manifest records complete hashes of the downloaded inputs; preserve it and the original cached runs, and pin model/dataset revisions for new evaluations.

## Design notes

These are disclosed in the paper:
- **Readouts:** the per-collection probability readout (strict vs lenient) was chosen after observing TREC-COVID. Both readouts are reported in the appendix.
- **RQ3 protocol:** primary ranker comparisons and comparisons against top-3 and adaptive selection were fixed after the pilot split. Top-2, top-5, raw-probability thresholds, and the ±2 F1 equivalence tests are exploratory additions.
- **RQ3 comparisons:** top-3 and Adaptive-k were the locked threshold-rule comparators. Top-2/top-5 comparisons were added later and are exploratory; equivalence margins and raw-threshold additions are post-hoc.
- **Label use:** raw RAG thresholds still use a readout selected on labelled calibration questions. They are not label-free. RQ2 raw Noul evaluation requires no target-label fitting.
- **Transfer calibration:** Group means within-DL/within-dense-BEIR transfer, grouped by numeric relevance threshold and weighted equally per source collection; it does not guarantee matching relevance semantics.
- **Splits:** Question IDs are disjoint, but 494/349 of 1,000 final MuSiQue questions reuse the terminal sub-question ID from pilot/calibration (578 either); 671 share any component with either. The component audit records this separately from the one normalized whole-question duplicate. A retrospective 329-question analysis excludes directly shared components; it does not ensure transitive family isolation.

## Licensing

- **Code:** MIT (`LICENSE`).
- **Underlying collections:** these keep their own licenses. No passage text is redistributed. Robust04, TREC News (Washington Post) and Signal-1M require their own data agreements.

## Releasing and archiving

To make a citable release:

1. Make the repository public (Settings → General → Danger zone → Change visibility).
2. Sign in to [Zenodo](https://zenodo.org) with GitHub and switch on this repository under *GitHub*.
3. Create a GitHub release (for example `v1.0`). Zenodo archives it and mints a DOI, using `.zenodo.json` for the metadata.
4. Put the repository URL and the DOI in `paper/sections/declarations.tex` and in the citation below.

`CITATION.cff` gives GitHub's "Cite this repository" entry.

## Citation

```bibtex
@misc{bhargava2026choicerank,
  title  = {Typed Decision Models as Passage Rerankers: Effectiveness, Calibration, and Context Selection},
  author = {Bhargava, Parv},
  year   = {2026},
  note   = {arXiv preprint, forthcoming}
}
```
