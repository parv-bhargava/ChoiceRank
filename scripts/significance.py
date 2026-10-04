"""Paired per-query significance tests on nDCG@10: two-sided paired t-test, bootstrap 95% CI
of the mean difference, and Holm correction across datasets for each system pair."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

from choicerank import RESULTS_DIR, RUNS_DIR
from choicerank.data import ALL, load_qrels
from choicerank.evaluate import per_query
from scripts.eval_all import read_trec

DECISION = [f"{m}-{q}" for m in ("jev", "clef", "laya", "jeff-0.8b") for q in ("choice3", "noul", "score4")]
LLM = ["qwen3-reranker-0.6B", "qwen235-yesno"]  # label-probability baselines
FORMATS = ("choice3", "noul", "score4")
PAIRS = ([(s, "bge-m3") for s in DECISION + LLM + ["minilm"]]
         + [(f"jev-{q}", f"clef-{q}") for q in FORMATS]          # closed vs open flagship decision model
         + [(f"jev-{q}", "qwen235-yesno") for q in FORMATS]      # typed vs label-probability readout
         + [(f"clef-{q}", "qwen235-yesno") for q in FORMATS]
         + [(f"clef-{q}", "qwen27base-yesno") for q in FORMATS]  # same base model: decision training vs P(yes)
         + [(s, "minilm") for s in DECISION]
         + [(s, "bm25") for s in DECISION + LLM + ["minilm", "bge-m3"]])


def holm(ps):
    order, m, adj, run = np.argsort(ps), len(ps), [0.0] * len(ps), 0.0
    for i, k in enumerate(order):
        run = max(run, (m - i) * ps[k])
        adj[k] = min(1.0, run)
    return adj


def main():
    rows = []
    for a, b in PAIRS:
        block = []
        for ds in ALL:
            fa, fb = RUNS_DIR / f"{ds}.top100.{a}.trec", RUNS_DIR / f"{ds}.top100.{b}.trec"
            if not (fa.exists() and fb.exists()):
                continue  # system not (yet) run on this collection
            q = load_qrels(ds)
            A, B = per_query(read_trec(fa), q), per_query(read_trec(fb), q)
            ks = sorted(set(A) & set(B))
            d = np.array([A[k] - B[k] for k in ks])
            p = stats.ttest_rel([A[k] for k in ks], [B[k] for k in ks]).pvalue
            rng = np.random.default_rng(0)
            boots = rng.choice(d, (2000, len(d))).mean(axis=1)
            block.append(dict(A=a, B=b, dataset=ds, n=len(ks), delta=100 * d.mean(),
                              ci_lo=100 * np.percentile(boots, 2.5), ci_hi=100 * np.percentile(boots, 97.5), p=p))
        if not block:
            continue
        for r, ph in zip(block, holm([r["p"] for r in block])):
            r["p_holm"] = ph
        rows += block
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "significance.csv", index=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}" if abs(x) < 1 else f"{x:.2f}"))


if __name__ == "__main__":
    main()
