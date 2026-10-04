"""Reliability diagrams for P(relevant) on the nine collections with judged negatives -> paper/figures/.

Palette: dataviz reference categorical slots 1-4 (validated: CVD separation PASS; contrast WARN
for aqua/yellow -> relieved by distinct markers, a legend, and Table 2's numbers).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from choicerank import ROOT, RUNS_DIR
from choicerank.data import DATASETS, load_qrels
from choicerank.evaluate import labels_for, reliability

OUT = ROOT / "paper" / "figures"
SYSTEMS = [  # (run system, label, color, marker) -- fixed categorical order, never cycled
    ("minilm", "MiniLM-L6", "#2a78d6", "o"),
    ("bge-m3", "bge-v2-m3", "#eb6834", "s"),
    ("laya-choice3", "Laya (C3)", "#1baf7a", "^"),
    ("jev-choice3", "Jev 1.13 (C3)", "#eda100", "D"),
]
PANELS = [("TREC DL19–23", ["dl19", "dl20", "dl21", "dl22", "dl23"]),
          ("BEIR (4 judged sets)", ["trec-covid", "dbpedia", "news", "robust04"])]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"


def pooled(system, datasets):
    ps, ys = [], []
    for ds in datasets:
        spec = DATASETS[ds]
        df = pd.read_parquet(RUNS_DIR / f"{ds}.top100.{system}.parquet")
        if spec.readout == "lenient" and "p_rel_lenient" in df:
            df["p_rel"] = df["p_rel_lenient"]  # same readout policy as eval_all.py
        nested = {}
        for q, d, p in zip(df.qid.astype(str), df.docid.astype(str), df.p_rel):
            nested.setdefault(q, {})[d] = float(p)
        p, y = labels_for(nested, load_qrels(ds), spec.rel_threshold, spec.dense_pool)
        ps.append(p), ys.append(y)
    return np.concatenate(ps), np.concatenate(ys)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 7.5, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.85), sharey=True, constrained_layout=True)
    for ax, (title, dss) in zip(axes, PANELS):
        ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
        for system, label, color, marker in SYSTEMS:
            p, y = pooled(system, dss)
            pts = [(mp, fy) for mp, fy, n in reliability(p, y, n_bins=10) if n >= 15]  # hide sparse bins
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=color, lw=1.5, marker=marker, ms=3.6, mec="white", mew=0.6,
                    label=label, zorder=3)
        ax.set_title(title, color=INK, fontsize=7.5, pad=3)
        ax.set_xlim(0, 1), ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.5, 1]), ax.set_yticks([0, 0.5, 1])
        ax.grid(color=GRID, lw=0.5), ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xlabel("Predicted P(relevant)")
    axes[0].set_ylabel("Fraction relevant")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False, fontsize=6.5,
               handlelength=1.6, columnspacing=0.9, handletextpad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"reliability.{ext}", dpi=300)
    print("wrote", OUT / "reliability.pdf")


if __name__ == "__main__":
    main()
