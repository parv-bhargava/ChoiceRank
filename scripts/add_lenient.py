"""Add a `p_rel_lenient` column (see prompts.p_lenient) to every decision-model parquet,
computed from the raw typed answers in runs/cache.sqlite. Idempotent; run_rerank.write_readouts
writes the column directly for new runs."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from choicerank import RUNS_DIR
from choicerank.judges.base import JudgeCache
from choicerank.prompts import QUESTIONS, p_lenient

JUDGES = {"jev": "dm:jev:all", "laya": "dm:laya-english:all", "jeff-0.8b": "dm:jeff-0.8b:all"}

cache = JudgeCache()
for model, judge_id in JUDGES.items():
    for qt in QUESTIONS:
        for pq in sorted(RUNS_DIR.glob(f"*.top50.{model}-{qt}.parquet")):
            df = pd.read_parquet(pq)
            raws = {}
            for qid in df.qid.astype(str).unique():
                for d, j in cache.get_many(judge_id, qid).items():
                    raws[(qid, d)] = j.raw
            df["p_rel_lenient"] = [p_lenient(qt, raws[(str(r.qid), str(r.docid))][qt]) for r in df.itertuples()]
            df.to_parquet(pq, index=False)
            print(f"{pq.stem}: added p_rel_lenient")
