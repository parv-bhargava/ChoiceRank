"""Answer F1 by MuSiQue hop count (issue #26): is Jev's RQ3 gain over bge confined to 2-hop questions?

The family-disjoint split is 88% 2-hop, so this breaks down the original final split (52% 2-hop) by the number of
hops in the question ID (2hop / 3hop* / 4hop*). For each generator, hop count and locked policy: mean F1 of Jev, bge
and, where run, Qwen3-Reranker-8B, and the paired Jev-minus-other difference with a question-level 95% t interval
(unadjusted; exploratory, added after final results). Reads runs/rag/musique.final.per_q.*.parquet (rag_summary.py
writes them).
  python scripts/rag_hop_f1.py  -> results/rag_hop_f1.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

from choicerank import RESULTS_DIR, RUNS_DIR

GENS = {"Qwen2.5-7B": "qwen_qwen-2.5-7b-instruct", "Llama-3.1-8B": "meta-llama_llama-3.1-8b-instruct",
        "Qwen3-30B-A3B": "qwen_qwen3-30b-a3b-instruct-2507"}
POLICIES = ["top1", "top2", "top3", "top5", "adaptive", "floor2+thr0.3", "floor2+thr0.5", "thrPlatt0.2", "thrPlatt0.3"]
OTHERS = ["bge-m3", "qwen3-reranker-8B"]


def hops(qid: str) -> int:
    return int(qid.split(":", 1)[1][0])  # musique:2hop__..., musique:3hop1__..., musique:4hop2__...


def main(split="final"):
    rows = []
    for gen, slug in GENS.items():
        base = RUNS_DIR / "rag" / f"musique.{split}.per_q.{slug}"
        parts = [pd.read_parquet(f"{base}.parquet")]
        q8 = Path(f"{base}.qwen3-reranker-8B.parquet")
        if q8.exists():
            parts.append(pd.read_parquet(q8))
        f1 = pd.concat(parts).pivot_table(index="qid", columns="policy", values="F1")
        f1["hops"] = [hops(q) for q in f1.index]
        for h, g in f1.groupby("hops"):
            for pol in POLICIES:
                jev = g[f"jev:{pol}"]
                for other in OTHERS:
                    col = f"{other}:{pol}"
                    if col not in g:
                        continue
                    d = 100 * (jev - g[col])
                    half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
                    rows.append(dict(split=split, gen=gen, hops=h, n=len(d), policy=pol, other=other,
                                     F1_jev=100 * jev.mean(), F1_other=100 * g[col].mean(), delta_F1=d.mean(),
                                     ci_lo=d.mean() - half, ci_hi=d.mean() + half,
                                     p=stats.ttest_1samp(d, 0).pvalue))
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "rag_hop_f1.csv", index=False)
    print(out[out.other == "bge-m3"].pivot_table(index=["policy"], columns=["hops", "gen"], values="delta_F1")
          .round(1).to_string())
    print(out.groupby(["other", "hops"]).agg(n=("n", "first"), delta_min=("delta_F1", "min"),
                                             delta_max=("delta_F1", "max"), ci_excl0=("ci_lo", lambda x: (x > 0).sum()),
                                             cells=("ci_lo", "size")).round(1).to_string())


if __name__ == "__main__":
    main()
