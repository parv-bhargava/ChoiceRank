"""Rewrite runs/*.trec from runs/*.parquet with rank-derived scores and BM25-rank tie-breaking
(the same convention as run_rerank.write_trec). Needed for runs written before 2026-10-03,
when raw float scores let trec_eval break ties by docid."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from choicerank import RUNS_DIR
from scripts.run_rerank import write_trec

for pq in sorted(RUNS_DIR.glob("*.parquet")):
    df = pd.read_parquet(pq).sort_values(["qid", "bm25_rank"])
    scores: dict[str, dict[str, float]] = {}
    for r in df.itertuples():
        scores.setdefault(str(r.qid), {})[str(r.docid)] = r.score
    tag = pq.stem.split(".", 2)[2]
    write_trec(pq.with_suffix(".trec"), scores, tag)
    ties = df.groupby("qid").score.apply(lambda s: s.duplicated().sum()).sum()
    print(f"{pq.stem}: rewrote, {ties} tied scores")
