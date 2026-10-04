"""Reranking-depth analysis from cached pointwise scores (no new model calls).

Pointwise scores are computed independently per (query, passage), so reranking only the BM25 top-k is exactly
simulated by re-ordering the first k candidates by model score and keeping BM25 order below rank k.
Cost of a pointwise reranker is linear in k. -> results/depth.csv, paper/tables/app_depth.tex
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from choicerank import RESULTS_DIR, ROOT, RUNS_DIR
from choicerank.data import ALL, BEIR8, DATASETS, DL, load_qrels
from choicerank.evaluate import rank_metrics

DEPTHS = [10, 20, 30, 50, 100]
SYSTEMS = {"minilm": "MiniLM", "bge-m3": "bge-v2-m3", "qwen3-reranker-0.6B": "Qwen3-Rer.-0.6B",
           "laya-choice3": "Laya C3", "jeff-0.8b-noul": "Jeff Noul", "clef-score4": "Clef S4", "jev-score4": "Jev S4"}


def run_at_depth(df: pd.DataFrame, k: int) -> dict:
    run = {}
    for q, g in df.groupby("qid"):
        g = g.sort_values("bm25_rank")
        head = g[g.bm25_rank <= k].sort_values(["score", "bm25_rank"], ascending=[False, True])
        tail = g[g.bm25_rank > k]
        order = list(head.docid) + list(tail.docid)
        run[str(q)] = {str(d): float(len(order) - i) for i, d in enumerate(order)}
    return run


def main():
    rows = []
    for s in SYSTEMS:
        for ds in ALL:
            pq = RUNS_DIR / f"{ds}.top100.{s}.parquet"
            if not pq.exists():
                continue
            df = pd.read_parquet(pq)
            qrels = load_qrels(ds)
            for k in DEPTHS:
                run = run_at_depth(df, k)
                cond = {q: {d: v for d, v in docs.items() if d in qrels.get(q, {})} for q, docs in run.items()}
                j10 = [np.mean([d in qrels.get(q, {}) for d in sorted(docs, key=lambda x: -docs[x])[:10]])
                       for q, docs in run.items() if q in qrels]
                rows.append(dict(system=s, dataset=ds, k=k,
                                 nDCG10=rank_metrics(run, qrels, DATASETS[ds].rel_threshold)["nDCG@10"],
                                 nDCG10_condensed=rank_metrics(cond, qrels, DATASETS[ds].rel_threshold)["nDCG@10"],
                                 judged_at_10=100 * np.mean(j10)))
    d = pd.DataFrame(rows)
    d.to_csv(RESULTS_DIR / "depth.csv", index=False)
    # BM25 reference
    bm = pd.read_csv(RESULTS_DIR / "ranking.csv").query("system == 'bm25'").set_index("dataset")["nDCG@10"]
    lines = [r"\begin{table*}[tp]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
             r"\caption{nDCG@10 when only the BM25 top-$k$ is reranked (simulated exactly from the pointwise scores; "
             r"passages below rank $k$ keep their BM25 order). Averages over TREC DL19--23 and BEIR-8. The cost of a pointwise "
             r"reranker grows linearly with $k$. BM25 alone: DL " + f"{bm[DL].mean():.1f}" + r", BEIR " + f"{bm[BEIR8].mean():.1f}" + r".}",
             r"\label{tab:depth}", r"\begin{tabular}{l" + "r" * len(DEPTHS) + "|" + "r" * len(DEPTHS) + "}", r"\toprule",
             r" & \multicolumn{" + str(len(DEPTHS)) + r"}{c|}{DL19--23} & \multicolumn{" + str(len(DEPTHS)) + r"}{c}{BEIR-8} \\",
             "System & " + " & ".join(f"$k$={k}" for k in DEPTHS) + " & " + " & ".join(f"{k}" for k in DEPTHS) + r" \\",
             r"\midrule"]
    for s, name in SYSTEMS.items():
        g = d[d.system == s]
        if g.dataset.nunique() < len(ALL):
            continue
        dl = [g[(g.k == k) & g.dataset.isin(DL)].nDCG10.mean() for k in DEPTHS]
        be = [g[(g.k == k) & g.dataset.isin(BEIR8)].nDCG10.mean() for k in DEPTHS]
        lines.append(f"{name} & " + " & ".join(f"{v:.1f}" for v in dl) + " & " + " & ".join(f"{v:.1f}" for v in be) + r" \\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{" + str(2 * len(DEPTHS) + 1) + r"}{l}{\textit{BEIR-8: condensed nDCG$'$@10 (left), Judged@10 in \% (right)}} \\")
    for s, name in SYSTEMS.items():
        g = d[d.system == s]
        if g.dataset.nunique() < len(ALL):
            continue
        cb = [g[(g.k == k) & g.dataset.isin(BEIR8)].nDCG10_condensed.mean() for k in DEPTHS]
        jb = [g[(g.k == k) & g.dataset.isin(BEIR8)].judged_at_10.mean() for k in DEPTHS]
        lines.append(f"{name} & " + " & ".join(f"{v:.1f}" for v in cb) + " & " + " & ".join(f"{v:.0f}" for v in jb) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (ROOT / "paper" / "tables" / "app_depth.tex").write_text("\n".join(lines), encoding="utf-8")
    piv = d.assign(grp=np.where(d.dataset.isin(DL), "DL", "BEIR")).pivot_table(index=["system", "grp"], columns="k", values="nDCG10")
    print(piv.round(1).to_string())


if __name__ == "__main__":
    main()
