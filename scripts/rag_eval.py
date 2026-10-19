"""RQ3 evaluation: answer quality and selection quality per policy -> results/rag_{dataset}_{split}.csv

  python scripts/rag_eval.py --dataset musique --split pilot --gen qwen/qwen-2.5-7b-instruct
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from choicerank import RESULTS_DIR, RUNS_DIR
from choicerank.rag import contains, exact_match, f1, load_split


def validate_grid(df, expected_qids, expected_policies=None):
    """Reject partial or duplicated evaluations before producing a final table."""
    required = {"qid", "policy", "answer", "pids", "n_passages", "in_tokens"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Generation file lacks columns: {sorted(missing)}")
    if df.empty or df.duplicated(["qid", "policy"]).any():
        raise ValueError("Generation grid is empty or contains duplicate question-policy rows")
    if df[list(required)].isna().any().any():
        raise ValueError("Generation grid contains missing answers or selection metadata")
    expected_qids = set(expected_qids)
    policies = set(df.policy)
    if expected_policies is not None and policies != set(expected_policies):
        raise ValueError(f"Policy grid mismatch: missing {sorted(set(expected_policies) - policies)}, "
                         f"unexpected {sorted(policies - set(expected_policies))}")
    for policy, rows in df.groupby("policy"):
        qids = set(rows.qid)
        if qids != expected_qids:
            raise ValueError(f"Incomplete question grid for {policy}: "
                             f"{len(expected_qids - qids)} missing, {len(qids - expected_qids)} unexpected")


def file_hash(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["musique", "nq"])
    ap.add_argument("--split", required=True, choices=["pilot", "main", "final", "final_fd"])
    ap.add_argument("--gen", required=True)
    ap.add_argument("--allow-partial", action="store_true", help="explicit exploratory evaluation of a partial run")
    ap.add_argument("--rankers", nargs="+", default=None, help="as passed to rag_generate.py (default jev bge-m3)")
    args = ap.parse_args()
    from scripts.rag_generate import RANKERS, final_policies, gen_path

    rankers = args.rankers or RANKERS
    slug = args.gen.replace("/", "_")
    source = gen_path(args.dataset, args.split, args.gen, rankers)
    extra = "" if rankers == RANKERS else "_" + "+".join(rankers)
    df = pd.read_parquet(source)
    items = {it.qid: it for it in load_split(args.dataset, args.split)}
    if not args.allow_partial:
        validate_grid(df, items, final_policies(rankers) if args.split.startswith("final") else None)
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
    out = RESULTS_DIR / f"rag_{args.dataset}_{args.split}_{slug}{extra}.csv"
    agg.sort_values("F1", ascending=False).to_csv(out)
    per_path = gen_path(args.dataset, args.split, args.gen, rankers, kind="per_q")
    per.to_parquet(per_path, index=False)
    per_path.with_suffix(".metadata.json").write_text(json.dumps(
        {"generation_sha256": file_hash(source), "complete_grid": not args.allow_partial,
         "questions": len(items), "policies": sorted(df.policy.unique())}, indent=2) + "\n", encoding="utf-8")
    print(agg.sort_values("F1", ascending=False).round(1).to_string())


if __name__ == "__main__":
    main()
