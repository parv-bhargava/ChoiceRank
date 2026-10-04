"""Appendix tables from the review re-analyses (results/*.csv) -> paper/tables/app_*.tex"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from choicerank import RESULTS_DIR, ROOT
from choicerank.data import BEIR8, CALIB, DATASETS, DL

OUT = ROOT / "paper" / "tables"
SHORT = {"dl19": "DL19", "dl20": "DL20", "dl21": "DL21", "dl22": "DL22", "dl23": "DL23", "trec-covid": "Covid",
         "dbpedia": "DBP", "news": "News", "robust04": "Rob04"}
NAMES = {"laya-choice3": "Laya C3", "jeff-0.8b-choice3": "Jeff C3", "jev-choice3": "Jev C3", "jev-score4": "Jev S4",
         "bm25": "BM25", "minilm": "MiniLM", "bge-m3": "bge-v2-m3", "qwen3-reranker-0.6B": "Qwen3-Rer.-0.6B",
         "qwen235-yesno": "Qwen3-235B P(yes)$^\\ast$", "jeff-0.8b-noul": "Jeff Noul", "laya-noul": "Laya Noul", "jev-noul": "Jev Noul",
         "clef-choice3": "Clef C3$^\\ast$", "clef-score4": "Clef S4$^\\ast$", "clef-noul": "Clef Noul$^\\ast$"}


def readouts():
    r = pd.read_csv(RESULTS_DIR / "calib_readouts.csv")
    r = r[r.system.isin(["laya-choice3", "jeff-0.8b-choice3", "jev-choice3", "jev-score4"])]
    lines = [r"\begin{table*}[tp]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{2.2pt}",
             r"\caption{ECE$\times$100 of Choice-3/Score-4 probabilities under the strict (s) and lenient (l) readouts. "
             r"The readout used in the main text is underlined. Noul has a single readout.}",
             r"\label{tab:readouts}", r"\begin{tabular}{ll" + "r" * len(CALIB) + "}", r"\toprule",
             "System & & " + " & ".join(SHORT[d] for d in CALIB) + r" \\", r"\midrule"]
    for s in ["laya-choice3", "jeff-0.8b-choice3", "jev-choice3", "jev-score4"]:
        for ro in ("strict", "lenient"):
            cells = []
            for d in CALIB:
                v = r[(r.system == s) & (r.dataset == d) & (r.readout == ro)].ECE.iloc[0]
                t = f"{v:.1f}"
                cells.append(rf"\underline{{{t}}}" if DATASETS[d].readout == ro else t)
            lines.append(f"{NAMES[s] if ro == 'strict' else ''} & {ro[0]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (OUT / "app_readouts.tex").write_text("\n".join(lines), encoding="utf-8")


def judged():
    j = pd.read_csv(RESULTS_DIR / "judged.csv")
    b = pd.read_csv(RESULTS_DIR / "calib_bounds.csv")
    order = ["bm25", "minilm", "bge-m3", "qwen3-reranker-0.6B", "qwen235-yesno", "laya-choice3", "jeff-0.8b-choice3",
             "clef-choice3", "clef-score4", "jev-choice3", "jev-noul", "jev-score4"]
    lines = [r"\begin{table*}[tp]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{2.5pt}",
             r"\caption{Pool coverage and pool-bias checks. Judged@10 (\%) and condensed (judged-only) nDCG$'$@10, "
             r"averaged over the collections each system was run on; ECE$\times$100 over the nine dense collections on judged "
             r"pairs only vs.\ treating unjudged candidates as non-relevant; mean $P(\text{rel})$ on judged relevant / non-relevant pairs. $^\ast$Run on a subset of collections (Clef: 7; Qwen3-235B: 8), so its averages are not directly comparable.}",
             r"\label{tab:judged}", r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r" & \multicolumn{2}{c}{Judged@10} & nDCG$'$ & \multicolumn{2}{c}{ECE} & \multicolumn{2}{c}{mean $P$} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
             r"System & DL & BEIR & All & judged & unj.=0 & rel. & non-rel. \\", r"\midrule"]
    for s in order:
        g = j[j.system == s]
        if g.empty:
            continue
        dl, be = g[g.dataset.isin(DL)].judged_at_10.mean(), g[g.dataset.isin(BEIR8)].judged_at_10.mean()
        bb = b[b.system == s]
        ece = (f"{bb.ECE_judged.mean():.1f} & {bb.ECE_unjudged0.mean():.1f} & {bb.mean_p_pos.mean():.2f} & "
               f"{bb.mean_p_neg.mean():.2f}") if s != "bm25" and len(bb) else "-- & -- & -- & --"
        lines.append(f"{NAMES[s]} & {dl:.1f} & {be:.1f} & {g.nDCG10_condensed.mean():.1f} & {ece} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (OUT / "app_judged.tex").write_text("\n".join(lines), encoding="utf-8")


def rag_selection():
    s = pd.read_csv(RESULTS_DIR / "rag_selection.csv")
    h = pd.read_csv(RESULTS_DIR / "rag_hops.csv").set_index("ranker")
    lines = [r"\begin{table*}[tp]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
             r"\caption{Generator-free selection quality on the RQ3 final sets: recall of gold passages within the top-$k$ "
             r"(all-gold: every gold passage selected), and, on MuSiQue, whether the top-1 passage is the first-hop or the "
             r"last (answer-bearing) supporting paragraph and whether it contains the answer string.}",
             r"\label{tab:ragsel}", r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Data & Ranker & R@1 & R@3 & R@5 & all-gold@5 \\", r"\midrule"]
    for ds in ("musique", "nq"):
        for rk, nm in (("jev", "Jev"), ("bge-m3", "bge-v2-m3")):
            g = s[(s.dataset == ds) & (s.ranker == rk)].set_index("k")
            lines.append(f"{'MuSiQue' if ds == 'musique' else 'NQ'} & {nm} & {g.loc[1].gold_recall:.1f} & "
                         f"{g.loc[3].gold_recall:.1f} & {g.loc[5].gold_recall:.1f} & {g.loc[5].all_gold:.1f} \\\\")
    lines += [r"\midrule", r"\multicolumn{6}{l}{MuSiQue top-1: \quad first hop / last hop / contains answer (\%)} \\"]
    for rk, nm in (("jev", "Jev"), ("bge-m3", "bge-v2-m3")):
        lines.append(rf" & {nm} & \multicolumn{{4}}{{l}}{{{h.loc[rk].top1_first_hop:.1f} / {h.loc[rk].top1_last_hop:.1f} / "
                     rf"{h.loc[rk].top1_contains_answer:.1f}}} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (OUT / "app_ragsel.tex").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    readouts(); judged(); rag_selection()
    print("wrote", *sorted(p.name for p in OUT.glob("app_*.tex")))
