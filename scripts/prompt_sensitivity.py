"""Prompt sensitivity (issue #12): nDCG@10 and ECE of each system under its original prompt and three paraphrases.

Paraphrases are fixed in code before any run: choicerank/prompts.py PARAPHRASES (Jev Noul question),
choicerank/judges/llm_yesno.py USER_PARAPHRASES (Qwen3-235B yes/no prompt), and
choicerank/judges/crossencoder.py Qwen3RerankerJudge.INSTRUCTIONS (Qwen3-Reranker instruction).

Runs (DL19, DL20, SciFact):
  python scripts/run_rerank.py --model qwen3-reranker-0.6B-instr1 --dataset dl19 dl20 scifact      # free, local
  python scripts/run_rerank.py --model jev --qtype noul_p1 --dataset dl19 dl20 scifact --dry-run    # paid
  python scripts/run_rerank.py --model qwen235-v1-yesno --dataset dl19 dl20 scifact --dry-run       # paid
Then:  python scripts/prompt_sensitivity.py -> results/prompt_sensitivity.csv
Variants whose runs are missing are skipped. ECE uses judged pairs where the collection has judged negatives,
otherwise unjudged candidates count as non-relevant (SciFact), as in the rest of the paper.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from choicerank import RESULTS_DIR, RUNS_DIR
from choicerank.data import load_qrels
from choicerank.evaluate import ece, per_query
from eval_all import read_trec
from review_analyses import load_pairs

DATASETS = ["dl19", "dl20", "scifact"]
FAMILIES = {
    "Jev Noul": ["jev-noul", "jev-noul_p1", "jev-noul_p2", "jev-noul_p3"],
    "Qwen3-Reranker-0.6B": ["qwen3-reranker-0.6B"] + [f"qwen3-reranker-0.6B-instr{i}" for i in (1, 2, 3)],
    "Qwen3-Reranker-8B": ["qwen3-reranker-8B"] + [f"qwen3-reranker-8B-instr{i}" for i in (1, 2, 3)],
    "Qwen3-235B P(yes)": ["qwen235-yesno"] + [f"qwen235-v{i}-yesno" for i in (1, 2, 3)],
}


def main():
    rows = []
    for fam, systems in FAMILIES.items():
        for ds in DATASETS:
            q = load_qrels(ds)
            for i, s in enumerate(systems):
                trec, pq = RUNS_DIR / f"{ds}.top100.{s}.trec", RUNS_DIR / f"{ds}.top100.{s}.parquet"
                if not (trec.exists() and pq.exists()):
                    continue
                pairs = load_pairs(s, ds)
                rows.append(dict(family=fam, dataset=ds, system=s, variant="original" if i == 0 else f"paraphrase{i}",
                                 ndcg10=100 * np.mean(list(per_query(read_trec(trec), q).values())),
                                 ECE=100 * ece(pairs.p.to_numpy(), pairs.y.to_numpy())))
    out = pd.DataFrame(rows)
    if out.empty:
        print("no runs yet")
        return
    spread = out.groupby(["family", "dataset"]).agg(variants=("system", "count"), ndcg_min=("ndcg10", "min"),
                                                   ndcg_max=("ndcg10", "max"), ndcg_sd=("ndcg10", "std"),
                                                   ECE_min=("ECE", "min"), ECE_max=("ECE", "max"), ECE_sd=("ECE", "std"))
    spread["ndcg_range"], spread["ECE_range"] = spread.ndcg_max - spread.ndcg_min, spread.ECE_max - spread.ECE_min
    out.to_csv(RESULTS_DIR / "prompt_sensitivity.csv", index=False)
    print(out.round(2).to_string(index=False))
    print()
    print(spread.round(2).to_string())


if __name__ == "__main__":
    main()
