"""Rebuild the short paper's three-panel figure from existing analysis CSVs.

Run from any directory with the ChoiceRank project Python environment.
No inference, generation, or original-paper writes are performed.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "figures"
DATASETS = ["dl19", "dl20", "dl21", "dl22", "dl23", "trec-covid", "nfcorpus",
            "touche", "dbpedia", "scifact", "signal", "news", "robust04"]
CALIB = ["dl19", "dl20", "dl21", "dl22", "dl23", "trec-covid", "dbpedia", "news", "robust04"]
RANK_SYSTEMS = [
    ("jev-score4", "Jev S4"), ("jev-noul", "Jev Noul"),
    ("qwen3-reranker-8B", "Qwen8B"), ("qwen3-reranker-4B", "Qwen4B"),
    ("bge-m3", "bge"), ("qwen3-reranker-0.6B", "Qwen0.6B"),
    ("minilm", "MiniLM"), ("bm25", "BM25"),
    ("laya-noul", "Laya Noul"), ("jeff-0.8b-noul", "Jeff Noul"),
]
CAL_SYSTEMS = [("bge-m3", "bge"), ("qwen3-reranker-8B", "Qwen8B"),
               ("laya-noul", "Laya Noul"), ("jeff-0.8b-noul", "Jeff Noul"), ("jev-noul", "Jev Noul")]
GENERATORS = [("Qwen2.5-7B", "Qwen2.5-7B"), ("Llama-3.1-8B", "Llama-3.1"),
              ("Qwen3-30B-A3B", "Qwen3-30B")]
BLUE, ORANGE, GRAY = "#0072B2", "#D55E00", "#666666"


def main():
    paths = {name: ROOT / "results" / name for name in
             ["ranking.csv", "calibration.csv", "r3_transfer.csv", "rag_final_tests.csv"]}
    rank = pd.read_csv(paths["ranking.csv"])
    rank = rank[(rank.depth == "top100") & rank.dataset.isin(DATASETS)]
    cal = pd.read_csv(paths["calibration.csv"])
    cal = cal[cal.dataset.isin(CALIB) & (cal.labels == "judged")]
    transfer = pd.read_csv(paths["r3_transfer.csv"])
    transfer = transfer[transfer.dataset.isin(CALIB)]
    tests = pd.read_csv(paths["rag_final_tests.csv"])
    effects = tests[(tests.comparison == "a: jev vs bge") &
                    (tests.A == "jev:top3") & (tests.B == "bge-m3:top3")]
    assert len(effects) == 6

    ranking = []
    for system, label in RANK_SYSTEMS:
        rows = rank[rank.system == system]
        assert len(rows) == 13 and rows.dataset.nunique() == 13, system
        ranking.append({"system": system, "label": label, "mean": float(rows["nDCG@10"].mean())})
    calibration = []
    for system, label in CAL_SYSTEMS:
        raw = cal[cal.system == system]
        group = transfer[transfer.system == system]
        assert len(raw) == 9 and raw.dataset.nunique() == 9, system
        assert len(group) == 9 and group.dataset.nunique() == 9, system
        calibration.append({"system": system, "label": label, "raw": float(raw.ECE.mean()),
                            "group": float(group.matched.mean())})
    plot_effects = []
    for dataset in ["musique", "nq"]:
        for generator, label in GENERATORS:
            rows = effects[(effects.dataset == dataset) & (effects.gen == generator)]
            assert len(rows) == 1, (dataset, generator, list(effects.gen.unique()))
            row = rows.iloc[0]
            plot_effects.append({"dataset": dataset, "generator": generator, "label": label,
                                 "delta": float(row.delta_F1), "lo": float(row.ci_lo),
                                 "hi": float(row.ci_hi), "p_holm": float(row.p_holm)})

    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.2, "axes.titlesize": 8,
                         "axes.labelsize": 7.2, "xtick.labelsize": 6.6, "ytick.labelsize": 7,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.25), gridspec_kw={"width_ratios": [1.04, 1, 1.1]})
    fig.subplots_adjust(left=0.095, right=0.99, bottom=0.21, top=0.78, wspace=0.84)
    for ax in axes:
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#888888")
        ax.tick_params(axis="both", length=0, pad=3)
        ax.xaxis.grid(True, color="#e6e6e6", linewidth=0.5)
        ax.set_axisbelow(True)

    ax = axes[0]
    for i, row in enumerate(ranking):
        color = BLUE if row["system"].startswith("jev") else ORANGE if row["system"] == "bge-m3" else GRAY
        ax.plot(row["mean"], i, "o", ms=3.6, color=color)
        ax.text(row["mean"] + 2.1, i, f'{row["mean"]:.1f}', va="center", fontsize=6.3, color=color)
    ax.set_yticks(range(len(ranking)), [r["label"] for r in ranking])
    ax.set_ylim(len(ranking) - 0.4, -0.6)
    ax.set_xlim(0, 70)
    ax.set_xticks([0, 20, 40, 60])
    ax.set_title("(a) Ranking", loc="left", pad=20, weight="bold")
    ax.set_xlabel("Mean nDCG@10")

    ax = axes[1]
    for i, row in enumerate(calibration):
        ax.plot([row["group"], row["raw"]], [i, i], color="#aaaaaa", lw=0.8)
        ax.plot(row["raw"], i, "o", color=ORANGE, ms=4, label="Raw" if i == 0 else None)
        ax.plot(row["group"], i, "s", color=BLUE, ms=3.7, label="Group" if i == 0 else None)
        ax.text(row["raw"] + 1.4, i, f'{row["raw"]:.1f}', va="center", fontsize=6.3, color=ORANGE)
        ax.text(row["group"] - 1.4, i, f'{row["group"]:.1f}', ha="right", va="center", fontsize=6.3, color=BLUE)
    ax.set_yticks(range(len(calibration)), [r["label"] for r in calibration])
    ax.set_ylim(len(calibration) - 0.35, -0.65)
    ax.set_xlim(0, 30)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_title("(b) Calibration", loc="left", pad=20, weight="bold")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.01), ncol=2, frameon=False,
              fontsize=6.5, handletextpad=0.3, columnspacing=0.9, borderaxespad=0)
    ax.set_xlabel("Mean ECE (lower better)")

    ax = axes[2]
    y = [0, 1, 2, 3.8, 4.8, 5.8]
    for i, row in enumerate(plot_effects):
        color = BLUE if row["dataset"] == "musique" else ORANGE
        ax.errorbar(row["delta"], y[i], xerr=[[row["delta"] - row["lo"]], [row["hi"] - row["delta"]]],
                    fmt="o" if row["dataset"] == "musique" else "s", ms=3.5,
                    color=color, lw=1.0, capsize=2,
                    label="MuSiQue" if i == 0 else "NQ" if i == 3 else None)
    ax.set_yticks(y, [r["label"] for r in plot_effects])
    ax.set_ylim(6.3, -0.6)
    ax.axvline(0, color="#777777", ls="--", lw=0.7)
    ax.set_xlim(-3, 12)
    ax.set_xticks([0, 5, 10])
    ax.set_title("(c) RAG: top-3", loc="left", pad=20, weight="bold")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.01), ncol=2, frameon=False,
              fontsize=6.5, handletextpad=0.3, columnspacing=0.8, borderaxespad=0)
    ax.set_xlabel(r"$\Delta$ F1 (Jev - bge)")
    for extension in ["pdf", "png"]:
        fig.savefig(OUT / f"results-overview.{extension}", dpi=300)
    payload = {"ranking": ranking, "calibration": calibration, "rag_effects": plot_effects,
               "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                                 for name, path in paths.items()}}
    (OUT / "chart-data.json").write_text(json.dumps(payload, indent=2))
    print("Wrote results-overview.pdf/png and audited chart-data.json")


if __name__ == "__main__":
    main()
