# Revision to-do: experiments and code (for Parv)

This list comes from a reviewer-style read of the paper by Dr. Amir Jafari (October 2026). The verdict was **major revision**. The analysis is careful and honest, but a journal reviewer would push back on the items below. Each item has a matching GitHub issue.

The text-only fixes (A1–A11) are already applied on branch `review/text-revisions`. See the issue "Text revisions applied to the manuscript", and read that diff before you start, because the result paragraphs were reorganized.

**General rules for every item**
- Fix the analysis plan **before** you look at new results. Write the policies, thresholds and readouts into the issue first, then run. The paper already has many post-hoc choices, and new experiments should not add more.
- Run paid scripts with `--dry-run` first to get a cost estimate.
- Regenerate tables with the scripts (`eval_all.py → significance.py → make_tables.py / make_figures.py`). Never hand-edit `paper/tables/*.tex`.
- After each item:
  - update the numbers in `paper/sections/*.tex` **and** `paper-short-acm/main.tex`;
  - rebuild both PDFs and regenerate `paper-short-acm/validation.json`;
  - note what changed in the issue and close it.

---

## Required

### B1. Family-disjoint MuSiQue split for RQ3 (highest priority)
**Problem.** 671 of the 1,000 MuSiQue final questions share at least one decomposition sub-question with the calibration or pilot split. The headline RAG result therefore rests on a split with leakage. The 329-question clean subset is retrospective and small, and a reviewer will ask for a proper split.

**To do.**
1. Build families as connected components over all 2,417 MuSiQue dev questions, linking questions that share any decomposition ID. The graph code already exists in `scripts/audit_musique_split_overlap.py` and `scripts/musique_clean_subset.py`.
2. Add family-disjoint splits to `choicerank/rag.py`, for example `calib_fd`, `final_fd`, with seed 13, still stratified by hop count. Whole families go into one split. Aim for about 200 calibration and as many final questions as the families allow; report the count. A new pilot isn't needed: **reuse the nine policies and thresholds locked after the original pilot, with no re-tuning**.
3. Score the new questions with Jev and bge (`scripts/rag_score.py`). Refit Platt scaling and re-select Jev's readout on `calib_fd`.
4. Run the three generators on `final_fd` (`rag_generate.py`, then `rag_eval.py`) and run the paired tests plus a family bootstrap.
5. **Paper:** make the family-disjoint split the primary RQ3 result. Move the original split and the 329-question subset to a sensitivity paragraph or the appendix, and update the abstract, RQ3, the conclusion and Threats to Validity.

**Done when** the new `tab:rag` reports family-disjoint results and the families are verified disjoint by the audit script.

### B2. Add Qwen3-Reranker-8B (and 0.6B) to the RAG comparison
**Problem.** Qwen3-Reranker-8B ties Jev in RQ1 (56.0 vs 56.1) but is missing from RQ3. As it stands, "Jev helps RAG" could simply mean "any strong reranker beats bge". The paper currently has to say "no RAG advantage over Qwen8B is established".

**To do.** Score the RAG splits with `qwen3-reranker-8B` (and `-0.6B`); use the B1 split if it is done. Fit Platt scaling for them on calibration, run all nine policies with the three generators, and test Jev vs Qwen8B with the same family of Holm-corrected tests. Also report last-hop@1 and first-hop@1 for Qwen8B in `tab:ragsel`. That shows whether Jev's "answer-bearing hop first" behavior is unique.

**Paper:** add Qwen8B rows to `tab:rag` and `tab:ragsel`, and update RQ3, the abstract and the conclusion. If Qwen8B matches Jev, say so plainly; it is still a publishable finding.

### B4. Equal-context control against Qwen3-Reranker-8B
**Problem.** Jev reads 4,000 characters (about 1,000 tokens), while the label-probability baselines read 448 tokens. The equal-context ablation covers only Jev vs bge, so "Jev ≈ Qwen8B" may be confounded by context length.

**To do.** Qwen3-Reranker supports long inputs. Run `qwen3-reranker-8B` with about 1,000 passage tokens (`--passage-tokens 1000`, or whatever matches 4,000 characters) at least on Touche, News and Robust04, and ideally on all 13 collections. Optionally also run Jev at 448 tokens on the same three collections; the bge ablation already does this for Touche, News and Robust04.

**Paper:** extend `tab:longdoc` with Qwen8B rows. Rerun the pooled Jev–Qwen8B comparison, including the "without Robust04" variant (this was B10), and update RQ1 and Threats to Validity.

### B6. Jev run-to-run determinism
**Problem.** Jev is a closed API called once per pair, and the appendix says determinism was never tested. Reviewers will ask how stable the numbers are.

**To do.** Re-score DL19, plus one BEIR collection such as SciFact, with the same Jev snapshot, using all three formats. Report the per-pair absolute probability difference (mean and max), the change in nDCG@10, and the Kendall τ between runs. It's cheap; check with `--dry-run`.

**Paper:** add one sentence in Setup or the appendix, and remove "we did not test run-to-run determinism".

### B7. Report Exact Match alongside F1 in RAG
**Problem.** Only F1 is reported, and journals usually expect EM as well.

**To do.** `scripts/rag_eval.py` already computes `EM` (and `Acc`). Extend `rag_summary.py` and `make_tables.py` so the main RAG table reports EM, or add an appendix table with EM for every policy and generator. No new API calls are needed.

### B11. Make strict-everywhere the primary calibration row
**Problem.** The per-collection readout policy was chosen after looking at TREC-COVID. The text now leads with Noul and strict-everywhere, but `tab:calibration` still shows the policy row as "Jev C3" and the strict one as a daggered extra row.

**To do.** In `scripts/make_tables.py` (around lines 118–150), make the strict readout the default row for every C3/S4 system and add the per-collection policy as the marked sensitivity rows. Do this for Laya, Jeff, Clef and Jev alike, not only Jev. Regenerate the table, then check that every C3/S4 calibration number quoted in `paper/sections/results.tex` still refers to the right row.

### B12. Fill in the Declarations and release the code
The red `[TODO: …]` markers in `paper/sections/declarations.tex` must be filled in before any submission:
- **Data and code availability:** make the repository public, or create a public release, and archive it on Zenodo for a DOI.
- **Competing interests:** state any relationship with TypeSafe, Cloudflare, firelex, ConvAI Innovations or OpenRouter. Reviewers will ask, because the paper evaluates a commercial API.
- **Funding:** say who paid for the API credits (about $31).

---

## Optional (do these if time and compute allow)

### B3. Qwen3-Reranker-4B/8B in full precision
**Problem.** The strongest baselines ran in 4-bit NF4 because of the 8 GB laptop GPU. Since the headline is "Jev ≈ Qwen8B", a reviewer may say the baseline was handicapped.

**To do.** On a GPU with at least 24 GB (GWU HPC, or a Colab A100/L4), run `qwen3-reranker-8B` (and 4B) in bf16 on all 13 collections. `choicerank/judges/crossencoder.py:76` forces 4-bit on CUDA, so add a flag to turn that off. Report the 4-bit vs bf16 difference. If it's small, one sentence closes the issue; if it's large, update RQ1 and the pooled analysis.

### B5. Extend the Clef same-base control to all seven Clef collections
**Problem.** The "Clef ≈ its base" claim rests on 3 collections, and mKH can't establish equivalence with k = 3. The rate limit was 20 requests per minute, so this needs patience but no money.

**To do.** Run the Qwen3.8-27B yes/no baseline on DL21–23 and Robust04 as well, then redo the same-base paragraph and its row in `tab:meta-sensitivity`.

### B8. Prompt sensitivity
**Problem.** There is one prompt per format, and reviewers often ask whether conclusions survive a paraphrase.

**To do.** Write 2–3 paraphrases of the Noul question, and of the yes/no prompt for the Qwen baselines. Run them on DL19, DL20 and SciFact, and report the spread in nDCG@10 and ECE in the appendix.

### B9. Dense first stage
**Problem.** BM25 is the only first stage.

**To do.** On DL19/20, rerank the top-100 from a dense retriever (for example the Pyserini prebuilt `contriever`, `bge-base` or SPLADE run) with bge, Qwen8B and Jev, to check that the ordering of systems holds. Put it in the appendix.

---

## Order of work
B7 and B11 (no API cost) → B6 (cheap) → B1 → B2 (reuse B1's split) → B4 → B12. Then the optional items B3, B5, B8 and B9.
