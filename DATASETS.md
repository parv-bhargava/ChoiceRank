# Dataset review (2026-10-03)

Sources: two web-research agents plus my own measurements on the castorini BM25 runs and Pyserini qrels (`github.com/castorini/eval/qrels`). Two of the measurements were cross-checked independently and agree: J@50 for DL19 (71%) and Robust04 (90%).

## Verdict

The current 5 collections (DL19/20, TREC-COVID, NFCorpus, SciFact) are fine for nDCG, but they are not the best choice for this paper:

1. **Calibration (RQ2) rests on only 3 collections.** NFCorpus and SciFact have positives-only qrels (SciFact: 2% of top-50 judged), so they cannot measure calibration. Six more collections have dense, graded pools with judged negatives.
2. **Top-50 breaks comparability.** RankGPT, RankZephyr and RankLLM all rerank **BM25 top-100** on **BEIR-8 + DL19–23**. At top-100 on the same castorini runs, our numbers are directly comparable to published GPT-3.5/GPT-4 listwise results.
3. **RQ3:** HotpotQA-distractor always has 2 gold passages out of 10, so a threshold has little room to beat top-k. MuSiQue (2–4 gold out of 20) and NQ+DPR (about 20% of questions with no answer-bearing passage) test selection properly.

## Reranking and calibration collections (BM25 runs: castorini/rank_llm_data; qrels: castorini/eval)

J@50 = share of BM25 top-50 pairs with a human judgment.

| Collection | q | Scale | Judged negatives | J@50 | Jev $ top-50 / top-100 | Role |
|---|---|---|---|---|---|---|
| DL19 / DL20 | 43 / 54 | 0–3 | yes | 71% / 72% | 0.13 / 0.25 | have; nDCG + calibration |
| **DL21 / DL22 / DL23** | 53 / 76 / 82 | 0–3 | yes | 68 / 64 / 55% | 0.28 / 0.54 | **add**: RankLLM standard, MS MARCO v2. DL23 has 31 synthetic queries; DL22 labels are copied to near-duplicates |
| TREC-COVID | 50 | 0–2 | yes | 81% | 0.10 / 0.20 | have |
| **Robust04** | 249 | 0–2 | yes (77%) | **90%** | 0.76 / 1.51 | **add**: best calibration set; long documents (truncated to 4k characters) |
| **TREC-News** | 57 | 0–4 | yes | 66% | 0.18 / 0.35 | **add**: long documents |
| **DBPedia-entity** | 400 | 0–2 | yes | 38% | 0.53 / 1.06 | **add** |
| Touche v2 | 49 | 0–2 | partial (Anserini drops zeros; use HF `BeIR/webis-touche2020-qrels`) | 16–20% | 0.13 / 0.25 | add for nDCG only (BM25 is known to be hard to beat) |
| Signal-1M | 97 | 1–2 | no | 12% | 0.11 / 0.23 | add for nDCG only |
| NFCorpus | 323 | 1–2 | no | 14% | 0.47 / 0.88 | keep for nDCG; **drop from calibration** |
| SciFact | 300 | 1 | no | 2% | 0.61 / 1.22 | keep for nDCG; **drop from calibration** |
| **All 13** | 1,833 | | | | **3.28 / 6.50** | Clef-flash costs about 2.1× Jev |

Dense calibration set if all are added: DL19–23, TREC-COVID, Robust04, News, DBPedia = **9 collections, about 1,064 queries**, up from 3 collections and 147 queries.

Licensing: the News, Robust04 and Signal corpora are licensed (TREC/NIST). The castorini runs embed the text, so we don't need the corpora, but we must not redistribute that text. Release only docids and scores.

## RQ3 (RAG context selection)

| Rank | Dataset | Why | Size used |
|---|---|---|---|
| 1 | **MuSiQue-Ans dev** (`dgslibisey/MuSiQue`, CC BY 4.0) | 20 paragraphs, 2–4 `is_supporting` gold, hard distractors. Used by DPS, BAR-RAG and Context-Picker | 1,000 q (stratified by hop) |
| 2 | **NQ-open test + DPR top-20** (`nq-test.json.gz`, `has_answer` flags) | Real retrieval; positives vary (median 2); **about 20% have none**, which tests whether a threshold can return 0 passages and the generator abstains | 1,000 q |
| 3 | HotpotQA distractor dev | Familiar anchor; secondary because of fixed 2-gold structure and single-hop shortcuts | 500 q |
| opt | RGB English (300 q) | Negative-rejection probe | — |

Generator: **Qwen2.5-7B-Instruct**, as used in DPS and BAR-RAG. Run it locally as a Q4 build on the 4060, or via OpenRouter at about $0.10 per million tokens ($1–3 for about 20k calls). Use temperature 0 and cap output at about 32 tokens. Spot-check with one larger model.

Metrics:
- **Answer quality:** EM/F1 with SQuAD normalization, using the official HotpotQA and MuSiQue scripts.
- **Selection quality:** precision, recall and all-gold recall against gold passages; number of passages and tokens.
- **No-gold questions:** abstention rate.
- **Plots:** Pareto curve of answer quality against tokens, with no-context and gold-only bounds.

## Baselines and analyses reviewers will expect

- A size-matched **pointwise LLM reranker**, e.g. Qwen3-Reranker-0.6B, run locally for free.
- The "Beyond Yes and No" graded-label approach (2310.14122) as the closest method.
- Post-hoc calibration of cross-encoders (done: Platt, 2-fold cross-fit).
- Judged@10 reported; calibration on judged pairs only (done).
- Agreement with human labels: **Cohen's κ against DL19–23 qrels**, compared with UMBRELA (GPT-4o: binary κ 0.42–0.50). Free from cached judgments.
- **Comparability:** at top-100 on castorini runs, we can cite RankGPT's GPT-4 numbers on identical candidates in place of running our own listwise baseline (or in addition to a cheap one).

## v2 / follow-up options

- **FollowIR** (Robust04/News21/Core17 with modified instructions; p-MRR). Decision models take explicit criteria text, so this is a distinctive angle.
- **BRIGHT** (3–4 small StackExchange domains, about 100 q each): a "System-1 limits" stress test. Positives-only qrels, so effectiveness only.
- **Lower-contamination checks:** NovelEval (21 q, in castorini), FutureQueryEval (148 q, post-April-2025). Caveat: both predate the Sept-2026 decision models, so they only reduce label-memorization risk.
- TREC RAG 2024 (27 fully manual topics) and TREC RAG 2025 (22 topics). These have no prebuilt BM25 runs, so passages would have to come from the 113M-segment corpus.
