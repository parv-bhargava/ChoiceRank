"""Neural first stage (issue #13): rerank the SPLADE++ EnsembleDistil top-100 (RankLLM retrieve_results) on DL19/20
and compare each system's nDCG@10 with the BM25 top-100 setting.

  python scripts/run_rerank.py --model bm25 --dataset splade            # first-stage order (SPLADE++ scores)
  python scripts/run_rerank.py --model bge-m3 --dataset splade
  python scripts/run_rerank.py --model qwen3-reranker-8B --dataset splade
  python scripts/run_rerank.py --model jev --qtype all --dataset splade --dry-run      # paid
  python scripts/first_stage.py  -> results/first_stage.csv

Paired t-tests over queries compare each system with bge on the same candidates; systems whose runs are missing are
skipped. The "bm25" run label on the SPLADE collections is the first-stage order itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy import stats

from choicerank import RESULTS_DIR, RUNS_DIR
from choicerank.data import load_qrels
from choicerank.evaluate import per_query
from eval_all import read_trec

SETTINGS = {"BM25": ["dl19", "dl20"], "SPLADE++": ["dl19-splade", "dl20-splade"]}
SYSTEMS = ["bm25", "minilm", "bge-m3", "qwen3-reranker-8B", "jev-choice3", "jev-noul", "jev-score4"]


def main():
    rows = []
    for stage, datasets in SETTINGS.items():
        for ds in datasets:
            q = load_qrels(ds)
            runs = {s: per_query(read_trec(RUNS_DIR / f"{ds}.top100.{s}.trec"), q)
                    for s in SYSTEMS if (RUNS_DIR / f"{ds}.top100.{s}.trec").exists()}
            for s, r in runs.items():
                row = dict(first_stage=stage, dataset=ds.split("-")[0], system="first stage" if s == "bm25" else s,
                           n_queries=len(r), ndcg10=100 * np.mean(list(r.values())))
                if s != "bge-m3" and "bge-m3" in runs:
                    ks = sorted(set(r) & set(runs["bge-m3"]))
                    a, b = [r[k] for k in ks], [runs["bge-m3"][k] for k in ks]
                    row["delta_vs_bge"], row["p_vs_bge"] = 100 * np.mean(np.subtract(a, b)), stats.ttest_rel(a, b).pvalue
                rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "first_stage.csv", index=False)
    print(out.pivot_table(index="system", columns=["first_stage", "dataset"], values="ndcg10").round(1).to_string())


if __name__ == "__main__":
    main()
