"""Score RQ3 candidate passages with a judge -> runs/rag/{dataset}.{split}.{model}.parquet

  python scripts/rag_score.py --model bge-m3 --dataset musique --split calib pilot
  python scripts/rag_score.py --model jev --dataset musique --split calib pilot --dry-run
  python scripts/rag_score.py --model jev --dataset musique --split calib pilot --i-approve-spend --max-usd 0.5

Decision models are asked all three formats in one call (qtype=all); the parquet keeps every readout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from choicerank import ROOT, RUNS_DIR
from choicerank.prompts import QUESTIONS, p_lenient, readout
from choicerank.rag import load_split

PAID = {"jev", "clef", "clef-flash"}
OUT = RUNS_DIR / "rag"


def main():
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True, choices=["musique", "nq"])
    ap.add_argument("--split", nargs="+", required=True, choices=["calib", "pilot", "main", "final"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--i-approve-spend", action="store_true")
    ap.add_argument("--max-usd", type=float, default=0.5)
    args = ap.parse_args()
    paid = args.model in PAID
    decision = args.model not in ("minilm",) and not args.model.startswith("bge-m3")
    qtype = "all" if decision else "choice3"

    if args.dry_run or (paid and not args.i_approve_spend):
        from choicerank.cost import estimate

        for split in args.split:
            print(split, estimate([it.as_query() for it in load_split(args.dataset, split)], args.model, qtype))
        if paid and not args.dry_run:
            print("Paid model: re-run with --i-approve-spend --max-usd N")
        return

    from choicerank.judges import JudgeCache, make_judge

    judge, cache = make_judge(args.model, qtype), JudgeCache()
    OUT.mkdir(parents=True, exist_ok=True)
    spent = 0.0
    for split in args.split:
        rows = []
        for it in tqdm(load_split(args.dataset, split), desc=f"{args.dataset}.{split}.{args.model}"):
            res, new = judge.judge_query(it.as_query(), cache)
            for rank_in, p in enumerate(it.passages):
                j = res[p.pid]
                row = dict(qid=it.qid, pid=p.pid, orig_rank=rank_in, gold=p.gold, score=j.score, p_rel=j.p_rel,
                           latency_ms=j.latency_ms, cost_usd=j.cost_usd)
                if decision:
                    for qt in QUESTIONS:
                        s, pr = readout(qt, j.raw[qt])
                        row[f"score_{qt}"], row[f"p_{qt}"], row[f"plen_{qt}"] = s, pr, p_lenient(qt, j.raw[qt])
                rows.append(row)
                if p.pid in new:
                    spent += j.cost_usd
            if paid and spent > args.max_usd:
                print(f"STOP: spend ${spent:.3f} exceeded --max-usd {args.max_usd}")
                break
        pd.DataFrame(rows).to_parquet(OUT / f"{args.dataset}.{split}.{args.model}.parquet", index=False)
        print(f"{args.dataset}.{split}.{args.model}: {len(rows)} passages, cumulative ${spent:.4f}")
        if paid and spent > args.max_usd:
            break


if __name__ == "__main__":
    main()
