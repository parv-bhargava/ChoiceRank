"""RQ3 generation: apply context-selection policies and answer with a generator LLM (OpenRouter).

  python scripts/rag_generate.py --dataset musique --split pilot --gen qwen/qwen-2.5-7b-instruct --dry-run
  python scripts/rag_generate.py --dataset musique --split pilot --gen qwen/qwen-2.5-7b-instruct --i-approve-spend --max-usd 0.5

Policies (per ranker R in {jev, bge-m3}; P = R's relevance probability):
  none / all / gold                      closed-book, all candidates (ranker order), oracle gold passages
  R:top{1,2,3,5,10}                      fixed top-k by R's ranking score
  R:thrPlatt{0.3,0.5,0.7}                P > t after Platt scaling fit on the disjoint calib split
  jev:thrRaw{0.3,0.5,0.7}                P > t with Jev's raw (uncalibrated) probability
  R:adaptive                             Adaptive-k: cut at the largest score gap
Pre-registered readout rule: Jev's probability readout is the one with the lowest Brier score on the
calib split among {C3 strict, C3 lenient, Noul}; ranking always uses the Score-4 expected level.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from choicerank import ROOT, RUNS_DIR
from choicerank.evaluate import apply_platt, fit_platt
from choicerank.rag import adaptive_k, floor_threshold, load_split, threshold, topk

OUT = RUNS_DIR / "rag"
PROMPT_VERSION = "v1"
SYSTEM = "Answer the question. Give only the short answer (a few words), with no explanation."
CTX_TMPL = ("Answer the question based on the given passages. Give only the short answer (a few words), "
            "with no explanation.\n\n{docs}\n\nQuestion: {q}\nAnswer:")
CB_TMPL = "Question: {q}\nAnswer:"
RANKERS = ["jev", "bge-m3"]
KS, TS = [1, 2, 3, 5, 10], [0.3, 0.5, 0.7]
# Added after the MuSiQue pilot (declared in the paper): lower thresholds and top-k floors.
TS_POST, FLOORS, TS_FLOOR = [0.1, 0.2], [1, 2], [0.1, 0.3, 0.5]


# ------------------------------------------------------------------------- scores + calibration

def ranker_tables(dataset: str, split: str, stand_in: bool = False) -> dict[str, pd.DataFrame]:
    """Per ranker: qid, pid, rank_score, p_raw, p_platt (Platt fit on calib split, with gold labels)."""
    out = {}
    for r in RANKERS:
        src = r
        if not (OUT / f"{dataset}.{split}.{r}.parquet").exists():
            if not stand_in:
                raise FileNotFoundError(f"score {dataset}.{split} with {r} first (scripts/rag_score.py)")
            src = "bge-m3"  # dry-run cost estimate only: same policy count, stand-in scores
            print(f"[dry-run] {r} scores missing; using bge-m3 as a stand-in to size the plan")
        cal = pd.read_parquet(OUT / f"{dataset}.calib.{src}.parquet")
        ev = pd.read_parquet(OUT / f"{dataset}.{split}.{src}.parquet")
        if src == "jev":
            cands = {"C3 strict": "p_choice3", "C3 lenient": "plen_choice3", "Noul": "p_noul"}
            brier = {k: float(np.mean((cal[c] - cal.gold.astype(int)) ** 2)) for k, c in cands.items()}
            pick = min(brier, key=brier.get)
            print(f"[{dataset}] Jev readout by calib Brier: " + ", ".join(f"{k} {v:.3f}" for k, v in brier.items())
                  + f" -> {pick}")
            pcol, scol = cands[pick], "score_score4"
        else:
            pcol, scol = "p_rel", "score"
        a, b = fit_platt(cal[pcol].to_numpy(), cal.gold.astype(int).to_numpy())
        out[r] = pd.DataFrame({"qid": ev.qid, "pid": ev.pid, "orig_rank": ev.orig_rank, "rank_score": ev[scol],
                               "p_raw": ev[pcol], "p_platt": apply_platt(ev[pcol].to_numpy(), a, b)})
    return out


# Locked 2026-10-04 after both pilots, BEFORE any final-split generation (see PROGRESS.md).
FINAL_POLICIES = {"none", "all", "gold"} | {
    f"{r}:{p}" for r in RANKERS
    for p in ["top1", "top2", "top3", "top5", "adaptive", "floor2+thr0.3", "floor2+thr0.5", "thrPlatt0.2", "thrPlatt0.3"]}
# Post-hoc addendum (2026-10-04, final results already seen): raw-probability
# thresholds with a readout selected using labelled calibration questions.
FINAL_POLICIES |= {"jev:thrRaw0.3", "jev:thrRaw0.5", "jev:thrRaw0.7"}


def policies(item, tables) -> dict[str, list[str]]:
    pol = {"none": [], "gold": [p.pid for p in item.passages if p.gold]}
    for r, t in tables.items():
        d = t[t.qid == item.qid].sort_values(["rank_score", "orig_rank"], ascending=[False, True])
        ranked = list(d.pid)
        score, praw, pcal = (dict(zip(d.pid, d[c])) for c in ("rank_score", "p_raw", "p_platt"))
        if r == "jev":
            pol["all"] = ranked  # all candidates in Jev order (order matters little; one shared baseline)
        for k in KS:
            pol[f"{r}:top{k}"] = topk(ranked, k)
        for th in TS + TS_POST:
            pol[f"{r}:thrPlatt{th}"] = threshold(ranked, pcal, th)
            if r == "jev" and th in TS:
                pol[f"{r}:thrRaw{th}"] = threshold(ranked, praw, th)
        for kmin in FLOORS:
            for th in TS_FLOOR:
                pol[f"{r}:floor{kmin}+thr{th}"] = floor_threshold(ranked, pcal, th, kmin)
        pol[f"{r}:adaptive"] = adaptive_k(ranked, score)
    return pol


# ------------------------------------------------------------------------- generation

def build_prompt(item, pids: list[str]) -> str:
    by = {p.pid: p for p in item.passages}
    if not pids:
        return CB_TMPL.format(q=item.question)
    docs = "\n\n".join(f"Passage {i + 1} (Title: {by[p].title}): {by[p].text}" for i, p in enumerate(pids))
    return CTX_TMPL.format(docs=docs, q=item.question)


class GenCache:
    def __init__(self, path=OUT / "gen_cache.sqlite"):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        self.db.execute("CREATE TABLE IF NOT EXISTS g (key TEXT PRIMARY KEY, payload TEXT)")

    @staticmethod
    def key(model, prompt):
        return hashlib.sha256(f"{PROMPT_VERSION}|{model}|{prompt}".encode()).hexdigest()

    def get(self, k):
        with self.lock:
            row = self.db.execute("SELECT payload FROM g WHERE key=?", (k,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, k, v):
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO g VALUES (?,?)", (k, json.dumps(v)))
            self.db.commit()


def call_openrouter(client, model, prompt, key):
    body = {"model": model, "temperature": 0, "max_tokens": 32, "usage": {"include": True},
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]}
    for attempt in range(6):
        r = client.post("https://openrouter.ai/api/v1/chat/completions", json=body,
                        headers={"Authorization": f"Bearer {key}"})
        if r.status_code in (408, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524):
            time.sleep(min(2 ** attempt, 30))
            continue
        r.raise_for_status()
        d = r.json()
        u = d.get("usage", {})
        return {"answer": d["choices"][0]["message"]["content"].strip(), "in_tokens": u.get("prompt_tokens", 0),
                "out_tokens": u.get("completion_tokens", 0), "cost_usd": float(u.get("cost", 0.0)),
                "provider": d.get("provider")}
    r.raise_for_status()


def price(model):
    m = {x["id"]: x for x in httpx.get("https://openrouter.ai/api/v1/models", timeout=60).json()["data"]}[model]
    return float(m["pricing"]["prompt"]), float(m["pricing"]["completion"])


def main():
    load_dotenv(ROOT / ".env")
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["musique", "nq"])
    ap.add_argument("--split", required=True, choices=["pilot", "main", "final"])
    ap.add_argument("--gen", required=True, help="OpenRouter model id, e.g. qwen/qwen-2.5-7b-instruct")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--i-approve-spend", action="store_true")
    ap.add_argument("--max-usd", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    items = load_split(args.dataset, args.split)
    tables = ranker_tables(args.dataset, args.split, stand_in=args.dry_run)
    cache = GenCache()
    plan, prompts = [], {}
    for it in items:
        for name, pids in policies(it, tables).items():
            if args.split == "final" and name not in FINAL_POLICIES:
                continue
            pr = build_prompt(it, pids)
            k = GenCache.key(args.gen, pr)
            prompts[k] = pr
            plan.append(dict(qid=it.qid, policy=name, pids=json.dumps(pids), n_passages=len(pids), key=k))
    todo = [k for k in prompts if cache.get(k) is None]
    pin, pout = price(args.gen)
    est_in = sum(len(prompts[k]) / 4 + 30 for k in todo)
    est = est_in * pin + len(todo) * 8 * pout
    print(f"{len(plan)} (question, policy) pairs -> {len(prompts)} unique prompts, {len(todo)} uncached; "
          f"est. {est_in / 1e6:.2f}M input tokens, ${est:.2f} for {args.gen}")
    if args.dry_run or not args.i_approve_spend:
        if not args.dry_run:
            print("Re-run with --i-approve-spend --max-usd N to call the API.")
        return

    key = os.environ["OPENROUTER_API_KEY"]
    client = httpx.Client(timeout=60)
    spent, lock = [0.0], threading.Lock()

    def work(k):
        if spent[0] > args.max_usd:
            return
        res = call_openrouter(client, args.gen, prompts[k], key)
        cache.put(k, res)
        with lock:
            spent[0] += res["cost_usd"]

    with ThreadPoolExecutor(args.workers) as pool:
        list(tqdm(pool.map(work, todo), total=len(todo), desc=args.gen))
    if spent[0] > args.max_usd:
        print(f"STOP: spend ${spent[0]:.3f} exceeded --max-usd {args.max_usd}; partial results cached")
    rows = []
    for row in plan:
        res = cache.get(row["key"])
        if res:
            rows.append({**row, **res})
    slug = args.gen.replace("/", "_")
    pd.DataFrame(rows).to_parquet(OUT / f"{args.dataset}.{args.split}.gen.{slug}.parquet", index=False)
    print(f"wrote {len(rows)} rows; spend this run ${spent[0]:.4f}")


if __name__ == "__main__":
    main()
