"""Rerank BM25 candidates with one judge and write a TREC run + per-pair parquet.

Examples
  python scripts/run_rerank.py --model bm25 --dataset dl19
  python scripts/run_rerank.py --model minilm --dataset dl19 dl20 scifact
  python scripts/run_rerank.py --model jev --qtype choice3 --dataset core --dry-run
  python scripts/run_rerank.py --model jev --dataset dl19 --limit-queries 1 --i-approve-spend --max-usd 0.05

Paid judges refuse to run without --i-approve-spend, and stop once --max-usd is exceeded.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from choicerank import ROOT, RUNS_DIR
from choicerank.data import DATASETS, GROUPS, load_queries, truncate_passages

PAID = {"jev", "clef", "clef-flash", "qwen235-yesno", "qwen27base-yesno", "qwen235-graded"}


def run_name(model, qtype, ds, topk):
    """`model` is the run label (may carry a suffix such as -p448). Cross-encoders/BM25 have no qtype."""
    base = model.split("-p")[0] if "-p" in model else model
    tag = model if base in ("bm25", "minilm") or base.startswith(("bge-m3", "qwen3-reranker")) or base.endswith(("-yesno", "-graded")) else f"{model}-{qtype}"
    return f"{ds}.top{topk}.{tag}"


def write_trec(path: Path, scores: dict[str, dict[str, float]], tag: str):
    """`scores[qid]` must be in BM25 order: the stable sort then breaks score ties by BM25 rank.
    The written score is rank-derived, because trec_eval/ir_measures break ties by docid, and
    decision models (e.g. Jev rounds to 2 decimals) produce many exact ties. Raw scores live
    in the parquet."""
    with open(path, "w", encoding="utf-8") as f:
        for qid, docs in scores.items():
            ranked = sorted(docs.items(), key=lambda x: -x[1])
            for rank, (docid, _) in enumerate(ranked, 1):
                f.write(f"{qid} Q0 {docid} {rank} {len(ranked) - rank + 1} {tag}\n")


def main():
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", nargs="+", required=True, help=f"names or groups: {sorted(GROUPS)}")
    ap.add_argument("--qtype", default="choice3", choices=["choice3", "noul", "score4", "all"])
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--limit-queries", type=int, default=None, help="0 = all; default = dataset default")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--i-approve-spend", action="store_true")
    ap.add_argument("--max-usd", type=float, default=1.0)
    ap.add_argument("--passage-tokens", type=int, default=None,
                    help="equal-budget ablation: truncate every passage to its first N tokens for all models")
    args = ap.parse_args()

    names = [n for a in args.dataset for n in GROUPS.get(a, [a])]
    unknown = [n for n in names if n not in DATASETS]
    if unknown:
        ap.error(f"unknown datasets {unknown}; groups: {sorted(GROUPS)}")
    paid = args.model in PAID
    label = args.model + (f"-p{args.passage_tokens}" if args.passage_tokens else "")

    def load(n):
        qs = load_queries(n, args.topk, args.limit_queries)
        return truncate_passages(qs, args.passage_tokens) if args.passage_tokens else qs

    if args.dry_run or (paid and not args.i_approve_spend):
        from choicerank.cost import estimate

        rows = [{"dataset": n, **estimate(load(n), args.model, args.qtype)} for n in names]
        df = pd.DataFrame(rows)
        print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print(f"TOTAL ${df.usd.sum():.2f}  ({df.tokens_M.sum():.1f}M tokens, {df.pairs.sum()} pairs)")
        if paid and not args.dry_run:
            print("\nPaid model: re-run with --i-approve-spend --max-usd N to actually call the API.")
        return

    RUNS_DIR.mkdir(exist_ok=True)
    judge = None
    if args.model != "bm25":
        from choicerank.judges import JudgeCache, make_judge

        judge, cache = make_judge(args.model, args.qtype), JudgeCache()
        if args.passage_tokens:
            judge.id_suffix = f"@p{args.passage_tokens}"

    spent = 0.0
    for name in names:
        queries = load(name)
        tag = run_name(label, args.qtype, name, args.topk)
        scores, rows, t0 = {}, [], time.perf_counter()
        for q in tqdm(queries, desc=tag):
            if judge is None:
                scores[q.qid] = {c.docid: c.bm25 for c in q.cands}
                continue
            res, new = judge.judge_query(q, cache)
            scores[q.qid] = {d: j.score for d, j in res.items()}
            for rank, c in enumerate(q.cands, 1):
                j = res[c.docid]
                rows.append(dict(qid=q.qid, docid=c.docid, bm25_rank=rank, score=j.score, p_rel=j.p_rel,
                                 latency_ms=j.latency_ms, in_tokens=j.in_tokens, cost_usd=j.cost_usd))
                if c.docid in new:
                    spent += j.cost_usd
            if paid and spent > args.max_usd:
                print(f"STOP: spend ${spent:.3f} exceeded --max-usd {args.max_usd}")
                break
        if args.qtype == "all" and rows:
            # One request answered all three question types: emit one run per readout.
            write_readouts(name, label, args.topk, queries, rows, judge, cache)
        else:
            write_trec(RUNS_DIR / f"{tag}.trec", scores, tag)
            if rows:
                pd.DataFrame(rows).to_parquet(RUNS_DIR / f"{tag}.parquet", index=False)
        elapsed = time.perf_counter() - t0
        print(f"{tag}: {len(scores)} queries in {elapsed:.1f}s, cumulative ${spent:.4f}")
        log_manifest({"date": time.strftime("%Y-%m-%dT%H:%M:%S"), "dataset": name, "model": label,
                      "qtype": args.qtype, "topk": args.topk, "n_queries": len(scores), "n_pairs": len(rows),
                      "seconds": round(elapsed, 1), "usd": round(sum(r["cost_usd"] for r in rows), 6),
                      **environment(judge)})
        if paid and spent > args.max_usd:
            break


def environment(judge) -> dict:
    """Provenance for runs/manifest.jsonl: model version, hardware, library versions."""
    import platform

    env = {"host": platform.node(), "python": platform.python_version()}
    try:
        import torch

        env["torch"] = torch.__version__
        env["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor()
    except ImportError:
        pass
    if judge is not None:
        env["judge_id"] = judge.judge_id
        env["model_version"] = getattr(judge, "model", None) and str(getattr(judge, "model"))
        if getattr(judge, "backend", "").startswith("jeff"):
            import httpx

            try:
                env["model_version"] = httpx.get(judge.url.rsplit("/v1/", 1)[0] + "/health", timeout=5).json()["checkpoint"]
            except Exception:
                pass
            env["device"] = "jeff-serve (see server host)"
        elif getattr(judge, "paid", False):
            env["device"] = "remote API"
        if type(judge).__name__ == "LayaJudge":
            import laya

            env["model_version"] = f"convaiinnovations/laya ({judge.variant}), laya {laya.__version__}"
        if type(judge).__name__ == "CrossEncoderJudge":
            from choicerank.judges.crossencoder import ALIASES

            env["model_version"] = ALIASES.get(judge.name, judge.name)
    return env


def log_manifest(entry: dict) -> None:
    import json

    with open(RUNS_DIR / "manifest.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def write_readouts(name, model, topk, queries, rows, judge, cache):
    from choicerank.prompts import QUESTIONS, p_lenient, readout

    base = pd.DataFrame(rows)
    raws = {}
    for q in queries:
        for d, j in cache.get_many(judge.cache_id, q.qid).items():
            raws[(q.qid, d)] = j.raw
    for qt in QUESTIONS:
        tag = run_name(model, qt, name, topk)
        out = [readout(qt, raws[(r.qid, r.docid)][qt]) for r in base.itertuples()]
        df = base.assign(score=[s for s, _ in out], p_rel=[p for _, p in out],
                         p_rel_lenient=[p_lenient(qt, raws[(r.qid, r.docid)][qt]) for r in base.itertuples()])
        scores = {}
        for r in df.itertuples():
            scores.setdefault(r.qid, {})[r.docid] = r.score
        write_trec(RUNS_DIR / f"{tag}.trec", scores, tag)
        df.to_parquet(RUNS_DIR / f"{tag}.parquet", index=False)


if __name__ == "__main__":
    main()
