"""Jev run-to-run determinism (issue #6): compare the original Jev runs with a repeat scored under separate cache rows.

Run the repeat first (paid):
  python scripts/run_rerank.py --model jev --qtype all --dataset dl19 scifact --repeat-tag rep2 --dry-run
  python scripts/run_rerank.py --model jev --qtype all --dataset dl19 scifact --repeat-tag rep2 --i-approve-spend --max-usd 2

Then:  python scripts/determinism.py  -> results/determinism.csv

Per collection and question format: mean and max absolute difference in P(relevant) and in the ranking score per
query-passage pair, the share of pairs with identical scores, nDCG@10 of both runs and their difference, and the
mean Kendall tau-b between the two runs' scores within each query.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy import stats

from choicerank import RESULTS_DIR, RUNS_DIR
from choicerank.data import load_qrels
from choicerank.evaluate import per_query
from eval_all import read_trec

FORMATS = ["choice3", "noul", "score4"]


def compare(ds: str, fmt: str, tag: str) -> dict:
    a = pd.read_parquet(RUNS_DIR / f"{ds}.top100.jev-{fmt}.parquet")
    b = pd.read_parquet(RUNS_DIR / f"{ds}.top100.jev-{tag}-{fmt}.parquet")
    m = a.merge(b, on=["qid", "docid"], suffixes=("_a", "_b"))
    dp, ds_ = (m.p_rel_a - m.p_rel_b).abs(), (m.score_a - m.score_b).abs()
    taus = [stats.kendalltau(g.score_a, g.score_b).statistic for _, g in m.groupby("qid")]
    q = load_qrels(ds)
    na = per_query(read_trec(RUNS_DIR / f"{ds}.top100.jev-{fmt}.trec"), q)
    nb = per_query(read_trec(RUNS_DIR / f"{ds}.top100.jev-{tag}-{fmt}.trec"), q)
    ks = sorted(set(na) & set(nb))
    d = np.array([nb[k] - na[k] for k in ks])
    return dict(dataset=ds, format=fmt, n_pairs=len(m), n_queries=len(ks),
                p_absdiff_mean=dp.mean(), p_absdiff_max=dp.max(), score_absdiff_mean=ds_.mean(),
                score_absdiff_max=ds_.max(), pct_pairs_identical=100 * (ds_ == 0).mean(),
                kendall_tau_mean=np.nanmean(taus), kendall_tau_min=np.nanmin(taus),
                ndcg10_run1=100 * np.mean([na[k] for k in ks]), ndcg10_run2=100 * np.mean([nb[k] for k in ks]),
                ndcg10_diff=100 * d.mean(), ndcg10_query_absdiff_max=100 * np.abs(d).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="+", default=["dl19", "scifact"])
    ap.add_argument("--tag", default="rep2")
    args = ap.parse_args()
    rows = [compare(ds, fmt, args.tag) for ds in args.dataset for fmt in FORMATS]
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "determinism.csv", index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
