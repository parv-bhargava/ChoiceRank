"""RQ3 evaluation: answer quality and selection quality per policy -> results/rag_{dataset}_{split}.csv

  python scripts/rag_eval.py --dataset musique --split pilot --gen qwen/qwen-2.5-7b-instruct
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from choicerank import RESULTS_DIR, RUNS_DIR
from choicerank.rag import contains, exact_match, f1, load_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["musique", "nq"])
    ap.add_argument("--split", required=True, choices=["pilot", "main", "final"])
    ap.add_argument("--gen", required=True)
    args = ap.parse_args()
    slug = args.gen.replace("/", "_")
    df = pd.read_parquet(RUNS_DIR / "rag" / f"{args.dataset}.{args.split}.gen.{slug}.parquet")
    items = {it.qid: it for it in load_split(args.dataset, args.split)}
    recs = []
    for r in df.itertuples():
        it = items[r.qid]
        gold = {p.pid for p in it.passages if p.gold}
        sel = set(json.loads(r.pids))
        tp = len(sel & gold)
        recs.append(dict(policy=r.policy, qid=r.qid, has_gold=bool(gold),
                         EM=exact_match(r.answer, it.answers), F1=f1(r.answer, it.answers), Acc=contains(r.answer, it.answers),
                         n=r.n_passages, tokens=r.in_tokens, empty=float(not sel),
                         sel_P=tp / len(sel) if sel else np.nan, sel_R=tp / len(gold) if gold else np.nan,
                         all_gold=float(gold <= sel) if gold else np.nan))
    per = pd.DataFrame(recs)
    agg = per.groupby("policy").agg(EM=("EM", "mean"), F1=("F1", "mean"), Acc=("Acc", "mean"), n=("n", "mean"),
                                    tokens=("tokens", "mean"), empty=("empty", "mean"), sel_P=("sel_P", "mean"),
                                    sel_R=("sel_R", "mean"), all_gold=("all_gold", "mean"), q=("qid", "nunique"))
    for c in ["EM", "F1", "Acc", "empty", "sel_P", "sel_R", "all_gold"]:
        agg[c] *= 100
    if args.dataset == "nq":  # answerable vs no-answer-passage questions
        sub = per[~per.has_gold].groupby("policy")
        agg["EM_nogold"], agg["empty_nogold"] = 100 * sub.EM.mean(), 100 * sub.empty.mean()
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"rag_{args.dataset}_{args.split}_{slug}.csv"
    agg.sort_values("F1", ascending=False).to_csv(out)
    per.to_parquet(RUNS_DIR / "rag" / f"{args.dataset}.{args.split}.per_q.{slug}.parquet", index=False)
    print(agg.sort_values("F1", ascending=False).round(1).to_string())


if __name__ == "__main__":
    main()
