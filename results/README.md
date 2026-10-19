# Results

Aggregate results behind every table, figure and number in the paper *Typed Decision Models as Passage Rerankers: Effectiveness, Calibration, and Context Selection*.

**What these files are:**
- Small CSV/JSON summaries computed from the per-pair model scores in `runs/`.
- They contain no passage text and no API keys.
- Every file can be regenerated from `runs/` with the scripts listed below (section 5).
- `runs/` itself is not published.

## Conventions

| Item | Meaning |
|---|---|
| **Units** | nDCG@10, MRR@10, R@20, ECE and F1 are ×100 (e.g. `56.1` = 0.561). Brier score and probabilities are on a 0–1 scale. Latency is in ms per query–passage pair. Cost is USD per 1k queries at 100 passages. |
| **Candidates** | BM25 top-100 from RankLLM/Pyserini (`depth` = 100). |
| **Datasets** | `dl19`–`dl23` (TREC DL passage); `trec-covid`, `nfcorpus`, `touche`, `dbpedia`, `scifact`, `signal`, `news`, `robust04` (BEIR-8); `noveleval`, `futurequery` (newer-content checks, not in the 13-collection averages); `dl19-splade`, `dl20-splade` (same queries and qrels, SPLADE++ top-100 candidates). |
| **Calibration collections** | The 9 with judged negatives: `dl19`–`dl23`, `trec-covid`, `dbpedia`, `news`, `robust04`. |

**Systems:**
- **Baselines:** `bm25`, `minilm`, `bge-m3` (bge-reranker-v2-m3), `qwen3-reranker-{0.6B,4B,8B}` (4B/8B in 4-bit).
- **Typed decision models:** `jev-*`, `clef-*`, `jeff-0.8b-*`, `laya-*`. The suffix is the question format: `choice3`, `noul` (yes/no) or `score4`.
- **Label-probability LLM baselines:**
  - `qwen235-yesno` / `qwen235-graded`: Qwen3-235B, yes/no or 0–4 rating.
  - `qwen27base-yesno`: Clef's base model.
  - `qwen08base-yesno` / `qwen08base-graded`: Jeff's base model.
- **Modifiers and suffixes:**
  - `-p448`: every passage cut to its first 448 tokens.
  - `bge-m3-2048`: bge with a 2,048-token window.
  - `@strict`: Choice-3 or Score-4 under the strict readout on every collection (the main calibration rows); without the suffix, these systems use the post-hoc per-collection readout policy.
  - `mean9`: mean over the 9 calibration collections.

## 1. Ranking effectiveness (RQ1)

| File | Contents |
|---|---|
| `ranking.csv` | nDCG@10, MRR@10, R@20 for every system × collection (main ranking table). |
| `significance.csv` | Paired *t*-tests over queries between system pairs per collection: mean difference `delta`, bootstrap 95% CI, `p`, and `p_holm` (Holm correction across the collections of each pair). |
| `r3_meta.csv` | DerSimonian–Laird random-effects pooled difference over collections, with and without Robust04 (normal 95% CI, τ, I²). |
| `r3_meta_sensitivity.csv` | Same comparisons with normal vs. modified Knapp–Hartung (mKH) intervals, 90%/95% CIs and post-hoc TOST at ±2 nDCG (`equivalent`). |
| `r3_equiv.csv` | TOST equivalence (±2 nDCG, post-hoc margin) for the "on par" claims (normal inference). |
| `r3_eqctx.csv` | Equal-context ablation: Jev vs. bge at 448 tokens and vs. bge-2048 on Touche, News, Robust04; Jev (448) vs. Qwen3-Reranker-8B (448 tokens); Jev vs. `qwen3-reranker-8B-4k` (both read the first 4,000 characters). |
| `truncation.csv` | Share of candidate passages longer than each model's passage budget. |
| `judged.csv` | Judged@10 and condensed (judged-only) nDCG′@10, to show sensitivity to incomplete pools. |
| `depth.csv` | nDCG@10, condensed nDCG′@10 and Judged@10 when only the BM25 top-*k* is reranked (k = 10…100, simulated from the same scores). |
| `prompt_sensitivity.csv` | nDCG@10 and ECE on DL19, DL20 and SciFact under each system's original prompt and three paraphrases fixed in code before running (issue #12). Rows appear for the variants that have been run. |
| `first_stage.csv` | Neural first-stage check (issue #13): nDCG@10 on DL19/20 when reranking the BM25 vs. the SPLADE++ EnsembleDistil top-100, with paired differences to bge on the same candidates. |

## 2. Calibration (RQ2)

**Basic metrics:**

| File | Contents |
|---|---|
| `calibration.csv` | Per system × collection: ECE (15 bins), equal-mass ECE, Brier, base rate, mean prediction, number of judged pairs, temperature- and Platt-scaled ECE/Brier, latency, cost. |
| `calib_readouts.csv` | Decision models under both probability readouts (strict / lenient). `used` marks the readout applied in the main text. |
| `calib_ci.csv` | Query-cluster bootstrap 95% CIs of ECE, and of the mean ECE difference to bge (`mean9_minus_bge`). |
| `r3_calib_diff.csv` | Paired query-bootstrap ECE difference vs. bge per collection (2,000 replicates; Holm over the 9 collections) and the mean over 9. |
| `r3_calib_robust.csv` | ECE with 5–30 bins, equal-mass ECE, Brier decomposition (reliability, resolution, uncertainty). |

**Recalibration and pool checks:**

| File | Contents |
|---|---|
| `calib_transfer.csv` | Transfer (leave-one-collection-out) Platt scaling: no target labels used. |
| `r3_transfer.csv` | Transfer Platt three ways: `pooled` (all other collections, equal weight per pair), `balanced` (equal weight per collection), `matched` (only collections with the same numeric binarization threshold, DL→DL / BEIR→BEIR; this does not imply identical relevance semantics). |
| `calib_transfer_nodl.csv` | Transfer fit only on the other BEIR collections. |
| `r3_incoll.csv` | In-collection Platt scaling, 10× repeated 5-fold query cross-fitting (uses target labels; reference floor). |
| `calib_common7.csv` | Calibration on the 7 collections shared by all API models. |
| `calib_bounds.csv` | ECE on judged pairs vs. treating unjudged candidates as non-relevant; mean P(rel) for relevant / non-relevant pairs. Includes `@strict` rows for the Choice-3/Score-4 systems. |
| `r3_yesno_fallback.csv` | How often neither / only one of the Yes/No tokens appeared in the top-5 log-probabilities of the API yes/no baselines. |

## 3. Context selection for RAG (RQ3)

**Main RAG results:**

| File | Contents |
|---|---|
| `rag_{dataset}_{split}_{generator}.csv` | Per selection policy: EM, F1, accuracy, mean passages `n`, generator input `tokens`, share of empty selections, selection precision/recall, all-gold rate. `pilot` files are the design runs; `final` files are the 1,000 held-out questions; `final_fd` files are the 597 family-disjoint MuSiQue questions (primary MuSiQue result, issue #3). NQ files add EM/empty on questions without a gold passage. |
| `rag_final_summary.csv` | All final runs in one table (dataset × generator × policy). |
| `rag_{dataset}_final_{generator}_{ranker}.csv` | Issue #4: the nine locked policies for Qwen3-Reranker-8B (4-bit) and -0.6B on the final questions (Platt fit on the same calibration questions). |
| `rag_final_tests.csv` | Paired tests: (a) Jev vs. bge under each policy (Holm per dataset × generator), (b) threshold rule vs. fixed *k*, (c) Jev vs. Qwen3-Reranker-8B / -0.6B under each policy (Holm per dataset × generator × ranker; added after final results), with 95% CIs, TOST (±2 F1, post-hoc) and a test pooled over generators. |
| `rag_final_tests_em.csv` | Test (a), Jev vs. bge under each policy, repeated on Exact Match (`delta_EM`, Holm per dataset × generator). EM is a secondary metric. |
| `rag_final_fd_summary.csv`, `rag_final_fd_tests.csv` | The same summary and tests on the family-disjoint MuSiQue split, plus family-bootstrap 95% CIs (`fam_ci_lo`, `fam_ci_hi`; 20,000 replicates, seed 13). |
| `rag_thrraw.csv` | Post-hoc raw-probability thresholds vs. Platt threshold, top-3 and Adaptive-*k*. The raw readout was chosen on labelled calibration questions, so these are not label-free. |

**Generator-free diagnostics:**

| File | Contents |
|---|---|
| `rag_selection.csv` | Gold-passage recall@k and all-gold@k per ranker. |
| `rag_hit1.csv` | MuSiQue: top-1 is any supporting paragraph (hit@1) / the last hop (last_hop@1), with exact McNemar tests. |
| `rag_hops.csv` | MuSiQue: top-1 is the first hop / last hop / contains the answer string. |
| `r3_musique_mechanism.csv` | Same diagnostics for every Jev readout and for Qwen3-Reranker-0.6B under three instructions. |
| `r3_musique_masked.csv` | Same diagnostics with gold-answer strings replaced by `[MASKED]`. Descriptive only: the marker itself reveals where the answer occurred. |

**MuSiQue split-overlap audit and clean-subset sensitivity.** Question IDs are disjoint across the calibration, pilot and final splits, but MuSiQue questions share sub-questions.

| File | Contents |
|---|---|
| `musique_split_overlap_audit.csv` / `.json` | Per final question: whether its terminal sub-question or any sub-question also occurs in pilot or calibration. Includes MuSiQue question and answer text (MuSiQue is CC BY 4.0). |
| `musique_clean_subset_ids.csv` | The 329 final questions sharing no sub-question with pilot or calibration, with their source-graph family. |
| `musique_clean_subset_summary.csv` | F1/EM per policy and generator on the 329-question subset. |
| `musique_clean_subset_tests.csv` | Jev vs. bge and threshold-rule tests on the subset: question-level Holm, plus family-bootstrap 95% CIs (212 families, 20,000 replicates, seed 13). |
| `musique_clean_subset_metadata.json` | Subset definition, counts and settings. |
| `musique_fd_split_audit.json` | Family-disjoint MuSiQue splits (`calib_fd`, `final_fd`; issue #3): sizes, hop mix, family counts, and checks that no decomposition ID or family is shared with each other or the original pilot. |

## 4. Reproducibility

`reproduction_manifest.json` records:
- the Python and package versions;
- the split seed and all split IDs;
- SHA-256 hashes of every input file and analysis script;
- the git commit and a note on model revisions.

`determinism.csv` (issue #6): Jev run-to-run stability. DL19 and SciFact were re-scored with the same snapshot in all three formats (`run_rerank.py --repeat-tag rep2`). Per format: mean/max absolute difference in P(rel) and in the ranking score, share of identical pairs, per-query Kendall tau (mean/min), and nDCG@10 of both runs.

## 5. Regenerating these files

Run from the repository root, after the run scripts have filled `runs/` (see the main README):

```bash
python scripts/eval_all.py               # ranking.csv, calibration.csv
python scripts/significance.py           # significance.csv
python scripts/review_analyses.py        # calib_readouts/transfer/bounds/ci/common7/transfer_nodl, judged, rag_selection, rag_hops, rag_hit1
python scripts/review3_analyses.py       # all r3_*.csv, rag_thrraw.csv
python scripts/depth_analysis.py         # depth.csv
python scripts/truncation_stats.py       # truncation.csv
python scripts/prompt_sensitivity.py     # prompt_sensitivity.csv
python scripts/first_stage.py            # first_stage.csv
python scripts/rag_eval.py --dataset musique --split final --gen <generator>   # rag_{dataset}_{split}_{generator}.csv
python scripts/rag_score.py --model qwen3-reranker-8B --dataset musique --split calib final   # local GPU; also -0.6B, nq
python scripts/rag_generate.py --dataset musique --split final --gen <generator> --rankers qwen3-reranker-8B --i-approve-spend --max-usd 1
python scripts/rag_eval.py --dataset musique --split final --gen <generator> --rankers qwen3-reranker-8B   # rag_{dataset}_final_{generator}_{ranker}.csv
python scripts/rag_summary.py            # rag_final_summary.csv, rag_final_tests.csv, rag_final_tests_em.csv
python scripts/rag_summary.py --split final_fd   # rag_final_fd_summary.csv, rag_final_fd_tests.csv, paper/tables/rag-fd.tex
python scripts/audit_musique_split_overlap.py   # musique_split_overlap_audit.*
python scripts/audit_musique_split_overlap.py --family-disjoint   # musique_fd_split_audit.json
python scripts/musique_clean_subset.py          # musique_clean_subset_*
python scripts/determinism.py                  # determinism.csv
python scripts/reproduction_manifest.py         # reproduction_manifest.json
```

**Caveats when reusing these numbers:**
- **Bootstrap results** (CIs, family intervals) are seeded but can differ in the last digit across library versions.
- **Pooled equivalence:** the ±2 margins were chosen after seeing the results.
- **Calibration** is measured on judged BM25 candidates only.
