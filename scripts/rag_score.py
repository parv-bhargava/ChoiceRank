"""Score RQ3 candidate passages with a judge -> runs/rag/{dataset}.{split}.{model}.parquet

  python scripts/rag_score.py --model bge-m3 --dataset musique --split calib pilot
  python scripts/rag_score.py --model jev --dataset musique --split calib pilot --dry-run
  python scripts/rag_score.py --model jev --dataset musique --split calib pilot --i-approve-spend --max-usd 0.5

Decision models are asked all three formats in one call (qtype=all); the parquet keeps every readout.

--mask-answers (review feedback): replace every gold answer string (and alias) in passage titles and texts with
"[MASKED]" before scoring. This is a descriptive perturbation: the marker reveals answer location and does not
isolate reasoning from memorization. Output: {dataset}.{split}.{model}-masked.parquet; separate cache id.
"""

from __future__ import annotations

import argparse
import copy
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from choicerank import ROOT, RUNS_DIR
from choicerank.prompts import QUESTIONS, p_lenient, readout
from choicerank.rag import load_split

PAID = {"jev", "clef", "clef-flash", "qwen235-yesno", "qwen27base-yesno", "qwen235-graded"}
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
    ap.add_argument("--mask-answers", action="store_true")
    args = ap.parse_args()
    items_of = (lambda d, sp: [mask_answers(it) for it in load_split(d, sp)]) if args.mask_answers else load_split
    tag = args.model + ("-masked" if args.mask_answers else "")
    paid = args.model in PAID
    decision = not (args.model == "minilm" or args.model.startswith(("bge-m3", "qwen3-reranker-"))
                    or args.model.endswith(("-yesno", "-graded")))
    qtype = "all" if decision else "choice3"

    if args.dry_run or (paid and not args.i_approve_spend):
        from choicerank.cost import estimate

        for split in args.split:
            print(split, estimate([it.as_query() for it in items_of(args.dataset, split)], args.model, qtype))
        if paid and not args.dry_run:
            print("Paid model: re-run with --i-approve-spend --max-usd N")
        return

    from choicerank.judges import JudgeCache, make_judge

    judge, cache = make_judge(args.model, qtype), JudgeCache()
    if args.mask_answers:
        judge.id_suffix = (getattr(judge, "id_suffix", "") or "") + "@mask"
    OUT.mkdir(parents=True, exist_ok=True)
    spent = 0.0
    for split in args.split:
        rows = []
        for it in tqdm(items_of(args.dataset, split), desc=f"{args.dataset}.{split}.{tag}"):
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
        pd.DataFrame(rows).to_parquet(OUT / f"{args.dataset}.{split}.{tag}.parquet", index=False)
        print(f"{args.dataset}.{split}.{tag}: {len(rows)} passages, cumulative ${spent:.4f}")
        if paid and spent > args.max_usd:
            break


def mask_answers(item):
    """Copy of a RagItem with every answer string (and alias) replaced by [MASKED] in titles and texts.
    Matching is case-insensitive and tolerant of punctuation/whitespace between the answer's words (as in SQuAD
    normalisation), longest answers first."""
    it = copy.deepcopy(item)
    toks = sorted({tuple(re.findall(r"\w+", a.lower())) for a in item.answers} - {()}, key=lambda t: -sum(map(len, t)))
    if toks:
        alt = "|".join(r"[^\w]*".join(re.escape(w) for w in t) for t in toks)
        rx = re.compile(r"(?<!\w)(?:" + alt + r")(?!\w)", re.IGNORECASE)
        for p in it.passages:
            p.title, p.text = rx.sub("[MASKED]", p.title), rx.sub("[MASKED]", p.text)
    return it


if __name__ == "__main__":
    main()
