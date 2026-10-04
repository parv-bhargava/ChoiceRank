"""Run the locked RQ3 final protocol: 2 datasets x 3 generators, resumable (all calls cached).

  python scripts/rag_run_final.py --i-approve-spend --budget 6.0

`--budget` is a GLOBAL cap on RQ3 final-split spend (Jev scoring of the final split + all final
generations); each step is given only the remaining budget. Progress: runs/rag/final.log
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
GENS = ["qwen/qwen-2.5-7b-instruct", "meta-llama/llama-3.1-8b-instruct", "qwen/qwen3-30b-a3b-instruct-2507"]
LOG = ROOT / "runs" / "rag" / "final.log"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def final_spend() -> float:
    """Jev scoring of final-split passages + final-split generations, from the caches."""
    import pandas as pd

    total = 0.0
    for ds in ("musique", "nq"):
        p = ROOT / "runs" / "rag" / f"{ds}.final.jev.parquet"
        if p.exists():
            total += float(pd.read_parquet(p).cost_usd.sum())
    for g in GENS:
        for ds in ("musique", "nq"):
            p = ROOT / "runs" / "rag" / f"{ds}.final.gen.{g.replace('/', '_')}.parquet"
            if p.exists():
                df = pd.read_parquet(p).drop_duplicates("key")
                total += float(df.cost_usd.sum())
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--i-approve-spend", action="store_true", required=True)
    ap.add_argument("--budget", type=float, default=6.0)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    for ds in ("musique", "nq"):
        if not (ROOT / "runs" / "rag" / f"{ds}.final.jev.parquet").exists():
            log(f"scoring {ds} final with Jev")
            subprocess.run([PY, "scripts/rag_score.py", "--model", "jev", "--dataset", ds, "--split", "final",
                            "--i-approve-spend", "--max-usd", "0.7"], cwd=ROOT, check=True)
    for g in GENS:
        for ds in ("musique", "nq"):
            spent = final_spend()
            left = args.budget - spent
            log(f"final spend so far ${spent:.3f}; remaining budget ${left:.3f}")
            if left <= 0.05:
                log("STOP: global budget reached")
                return
            log(f"generate {ds} x {g}")
            r = subprocess.run([PY, "scripts/rag_generate.py", "--dataset", ds, "--split", "final", "--gen", g,
                                "--i-approve-spend", "--max-usd", f"{left:.3f}", "--workers", str(args.workers)],
                               cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
            tail = [l for l in r.stdout.splitlines() if "wrote" in l or "STOP" in l or "unique prompts" in l]
            log(" | ".join(tail) if tail else f"exit {r.returncode}: {r.stderr[-400:]}")
            if r.returncode != 0:
                log("step failed; re-run this script to resume (cached calls are not repeated)")
                return
    log(f"DONE. final spend ${final_spend():.3f}")


if __name__ == "__main__":
    main()
