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
| `notebooks/` | Colab notebook for Jeff latency on a T4. |

Inputs, run outputs, and results are not version-controlled:
- `data/`: downloaded automatically from public sources on first use.
- `runs/`: per-pair scores, TREC runs, API caches, and generator answers. Written by the run scripts.
- `results/`: evaluation CSVs. Written by the analysis scripts.

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
python scripts/run_rerank.py --model laya   --qtype all --dataset all
```

**Jeff** runs through its reference server ([firelex/jeff](https://github.com/firelex/jeff), commit `d0173b4`):

1. Start the server with `JEFF_CHECKPOINT=Jeff-Qwen3.5-0.8B-v1.2 JEFF_DEVICE=cuda PORT=8765 jeff-serve`.
2. Then run `python scripts/run_rerank.py --model jeff-0.8b --qtype all --dataset all`.

The server handles one request at a time.

### 2. API models (paid, through OpenRouter)

Every paid script refuses to call an API without `--i-approve-spend` and stops at `--max-usd`. Run it with `--dry-run` first to get a cost estimate.

| System | Command | Scope | Approx. cost |
|---|---|---|---|
| Jev 1.13 (`typesafe/jev-1.13-20260917`) | `run_rerank.py --model jev --qtype all --dataset all` | 13 collections, 3 formats per call | $6.2 |
| Jev, equal 448-token budget | `run_rerank.py --model jev --qtype all --dataset touche news robust04 --passage-tokens 448` | 3 collections | $1.4 |
| Clef 27B (`cloudflare/clef`) | `run_rerank.py --model clef --qtype all --dataset dl trec-covid robust04` | 7 collections | $11.7 |
| Qwen3-235B P(yes) | `run_rerank.py --model qwen235-yesno --dataset dl trec-covid nfcorpus robust04` | 8 collections | $2.8 |
| RQ3: Jev scoring and 3 generators | `rag_score.py ...`, then `rag_run_final.py --budget 6` | 2 × 1,000 questions | $6.9 incl. pilots |

The same-base Qwen3.8-27B P(yes) run (`--model qwen27base-yesno`) goes through Cloudflare, which limits new OpenRouter accounts to 20 requests per minute. Set `CHOICERANK_RPM=18` for it.

All judgments are cached, so interrupted runs resume where they stopped. API models can change behind a fixed name; we pin the Jev snapshot, and `runs/manifest.jsonl` records the date of every run.

### 3. Evaluate and rebuild tables and figures (free, minutes)

```bash
python scripts/eval_all.py          # results/ranking.csv, results/calibration.csv
python scripts/significance.py      # paired t-tests, bootstrap CIs, Holm per system pair
python scripts/review_analyses.py   # readouts, transfer calibration, pool bias, RAG selection, CIs
python scripts/depth_analysis.py    # reranking-depth simulation
python scripts/rag_summary.py       # RQ3 tests -> paper/tables/rag.tex
python scripts/make_tables.py && python scripts/make_appendix.py && python scripts/make_figures.py   # -> paper/tables, paper/figures
```

## Design notes

These are disclosed in the paper:
- **Readouts:** the per-collection probability readout (strict vs lenient) was chosen after observing TREC-COVID. Both readouts are reported in the appendix.
- **RQ3 protocol:** policies, thresholds and tested comparisons were locked after the pilot split and before any final-split generation. The ±2 F1 equivalence tests are post-hoc.

## Licensing

- **Code:** MIT (`LICENSE`).
- **Underlying collections:** these keep their own licenses. No passage text is redistributed. Robust04, TREC News (Washington Post) and Signal-1M require their own data agreements.

## Citation

```bibtex
@misc{bhargava2026choicerank,
  title  = {Typed Decision Models as Passage Rerankers: Effectiveness, Calibration, and Context Selection},
  author = {Bhargava, Parv},
  year   = {2026},
  note   = {arXiv preprint, forthcoming}
}
```
