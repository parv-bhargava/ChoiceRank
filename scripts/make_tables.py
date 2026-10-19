"""Generate the paper's LaTeX tables from results/*.csv -> paper/tables/*.tex.

Never hand-edit numbers in the paper: re-run eval_all.py, significance.py, truncation_stats.py,
then this script.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from choicerank import RESULTS_DIR, ROOT, RUNS_DIR
from choicerank.data import BEIR8, CALIB, DL

OUT = ROOT / "paper" / "tables"
HEAD = {"dl19": "DL19", "dl20": "DL20", "dl21": "DL21", "dl22": "DL22", "dl23": "DL23", "trec-covid": "Covid",
        "nfcorpus": "NFC", "touche": "Touche", "dbpedia": "DBP", "scifact": "SciF", "signal": "Signal",
        "news": "News", "robust04": "Rob04"}
LONG = ["touche", "news", "robust04"]

GROUPS = [
    ("", [("bm25", "BM25 (first stage)")]),
    ("Cross-encoders", [("minilm", "MiniLM-L6 (22M)"), ("bge-m3", "bge-reranker-v2-m3 (568M)")]),
    ("Label-probability LLM rerankers", [("qwen3-reranker-0.6B", "Qwen3-Reranker-0.6B"),
                                         ("qwen3-reranker-4B", "Qwen3-Reranker-4B (4-bit)"),
                                         ("qwen3-reranker-8B", "Qwen3-Reranker-8B (4-bit)"),
                                         ("qwen3-reranker-4B-bf16", "Qwen3-Reranker-4B (bf16)"),
                                         ("qwen3-reranker-8B-bf16", "Qwen3-Reranker-8B (bf16)"),
                                         ("qwen235-yesno", "Qwen3-235B, P(yes)"),
                                         ("qwen235-graded", "Qwen3-235B, graded 0--4"),
                                         ("qwen27base-yesno", "Qwen3.8-27B (Clef's base), P(yes)"),
                                         ("qwen08base-yesno", "Qwen3.5-0.8B (Jeff's base), P(yes)"),
                                         ("qwen08base-graded", "Qwen3.5-0.8B (Jeff's base), graded 0--4")]),
    ("Laya (421M)", [("laya-choice3", "\\quad Choice-3"), ("laya-noul", "\\quad Noul"), ("laya-score4", "\\quad Score-4")]),
    ("Jeff (0.8B)", [("jeff-0.8b-choice3", "\\quad Choice-3"), ("jeff-0.8b-noul", "\\quad Noul"),
                     ("jeff-0.8b-score4", "\\quad Score-4")]),
    ("Clef 27B (open weights, API; 7 collections)", [("clef-choice3", "\\quad Choice-3"), ("clef-noul", "\\quad Noul"),
                                                     ("clef-score4", "\\quad Score-4")]),
    ("Jev 1.13 (API)", [("jev-choice3", "\\quad Choice-3"), ("jev-noul", "\\quad Noul"), ("jev-score4", "\\quad Score-4")]),
]


MAIN_RANK_SYSTEMS = {"bm25", "minilm", "bge-m3", "qwen3-reranker-0.6B", "qwen3-reranker-4B",
                     "qwen3-reranker-8B", "qwen235-yesno", "qwen27base-yesno", "qwen08base-yesno",
                     "laya-noul", "jeff-0.8b-noul", "clef-score4", "jev-noul", "jev-score4"}


def ranking_table(rank: pd.DataFrame, sig: pd.DataFrame, compact=False) -> str:
    p = rank.pivot(index="system", columns="dataset", values="nDCG@10")
    # Averages only over complete collection sets: subset systems (Clef, Qwen3-235B) get "--" (no subset means)
    p["dlavg"] = p[DL].mean(axis=1, skipna=False)
    p["beiravg"] = p[BEIR8].mean(axis=1, skipna=False)
    p["all"] = p[DL + BEIR8].mean(axis=1, skipna=False)
    cols = DL + ["dlavg"] + BEIR8 + ["beiravg", "all"]
    groups = [(g, [(s, l) for s, l in rows if s in p.index]) for g, rows in GROUPS]
    if compact:
        groups = [(g, [(s, l) for s, l in rows if s in MAIN_RANK_SYSTEMS]) for g, rows in groups]
    groups = [(g, rows) for g, rows in groups if rows]  # optional baselines appear once their runs exist
    systems = [s for _, rows in groups for s, _ in rows]
    best = {c: p.loc[systems, c].max() for c in cols}
    s_bge = sig[sig.B == "bge-m3"].set_index(["A", "dataset"])
    head = [HEAD[c] if c in HEAD else {"dlavg": r"\textit{Avg}", "beiravg": r"\textit{Avg}", "all": r"\textbf{All}"}[c]
            for c in cols]
    lines = [r"\begin{table*}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{2.6pt}",
             r"\caption{nDCG@10 for reranking the BM25 top-100 (Pyserini runs and qrels, as in RankGPT/RankLLM). "
             r"$^\uparrow$/$^\downarrow$: significantly better/worse than bge-reranker-v2-m3 (paired $t$-test over queries, "
             r"Holm-corrected over the collections each pair was evaluated on, $p<0.05$). Best per column in bold. "
             r"Averages are reported only for systems run on every collection of the group (dash otherwise); "
             r"matched comparisons for Clef and the P(yes) LLMs are given in the text. "
             r"Touche, News and Robust04 contain long documents that every model truncates (Table~\ref{tab:longdoc}).}",
             r"\label{tab:ranking}",
             r"\begin{tabular}{l" + "r" * 5 + "|r|" + "r" * 8 + "|r|r}", r"\toprule",
             r" & \multicolumn{6}{c|}{TREC Deep Learning} & \multicolumn{9}{c|}{BEIR} & \\",
             "System & " + " & ".join(head) + r" \\", r"\midrule"]
    for gi, (gname, rows) in enumerate(groups):
        if gname:
            lines.append(rf"\multicolumn{{{len(cols) + 1}}}{{l}}{{\textit{{{gname}}}}} \\")
        for sysname, label in rows:
            cells = []
            for c in cols:
                v = p.loc[sysname, c]
                if np.isnan(v):  # system not run on this collection (or average undefined)
                    cells.append("--")
                    continue
                txt = f"{v:.1f}"
                if abs(v - best[c]) < 1e-9:
                    txt = rf"\textbf{{{txt}}}"
                if (sysname, c) in s_bge.index:
                    r = s_bge.loc[(sysname, c)]
                    if r.p_holm < 0.05:
                        txt += r"$^\uparrow$" if r.delta > 0 else r"$^\downarrow$"
                cells.append(txt)
            lines.append(f"{label} & " + " & ".join(cells) + r" \\")
        if gi < len(groups) - 1:
            lines.append(r"\addlinespace[1pt]")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    table = "\n".join(lines)
    if compact:
        table = table.replace("Best per column in bold.",
                              r"Best displayed configuration per column in bold; all formats and controls appear in Appendix Table~\ref{tab:ranking-full}.")
    else:
        table = table.replace(r"\label{tab:ranking}", r"\label{tab:ranking-full}")
    return table


def calibration_table(cal: pd.DataFrame, transfer: pd.DataFrame | None = None) -> str:
    """Raw ECE; transfer-calibrated ECE (Platt fit on the OTHER collections, no target labels), pooled over all of them
    and restricted to those graded like the target; in-collection Platt (10x repeated 5-fold query cross-fit; needs
    target labels); Brier; mean prediction; latency; cost. Choice-3/Score-4 rows use the strict readout on every
    collection; the post-hoc per-collection readout policy appears as marked sensitivity rows (issue #8)."""
    rows = [("minilm", "MiniLM-L6"), ("bge-m3", "bge-v2-m3"), ("qwen3-reranker-0.6B", "Qwen3-Rer.-0.6B"),
            ("qwen3-reranker-4B", "Qwen3-Rer.-4B"), ("qwen3-reranker-8B", "Qwen3-Rer.-8B"),
            ("qwen235-yesno", "Qwen3-235B P(yes)$^\\ast$"),
            ("laya-choice3@strict", "Laya C3"), ("laya-choice3", "\\quad policy$^\\dagger$"), ("laya-noul", "Laya Noul"),
            ("jeff-0.8b-choice3@strict", "Jeff C3"), ("jeff-0.8b-choice3", "\\quad policy$^\\dagger$"),
            ("jeff-0.8b-noul", "Jeff Noul"),
            ("clef-choice3@strict", "Clef C3$^\\ast$"), ("clef-choice3", "\\quad policy$^{\\ast\\dagger}$"),
            ("clef-noul", "Clef Noul$^\\ast$"),
            ("clef-score4@strict", "Clef S4$^\\ast$"), ("clef-score4", "\\quad policy$^{\\ast\\dagger}$"),
            ("jev-choice3@strict", "Jev C3"), ("jev-choice3", "\\quad policy$^\\dagger$"), ("jev-noul", "Jev Noul"),
            ("jev-score4@strict", "Jev S4"), ("jev-score4", "\\quad policy$^\\dagger$")]
    d = cal[cal.dataset.isin(CALIB)].groupby("system")
    agg = pd.DataFrame({"ECE": d.ECE.mean(), "Brier": d.Brier.mean(), "ECE_Platt": d.ECE_Platt.mean(),
                        "mean_p": d.mean_p.mean(), "base": d.base_rate.mean()})
    if transfer is not None:
        agg["ECE_LOCO"] = transfer.groupby("system").ECE_LOCO.mean()
    r3 = RESULTS_DIR
    if (r3 / "r3_transfer.csv").exists():
        t3 = pd.read_csv(r3 / "r3_transfer.csv").groupby("system")
        rob = pd.read_csv(r3 / "r3_calib_robust.csv").groupby("system")
        inc = pd.read_csv(r3 / "r3_incoll.csv").groupby("system").ECE_incoll_mean.mean()
        for sys_ in t3.groups:
            if sys_ not in agg.index:
                agg.loc[sys_, ["ECE", "Brier", "mean_p"]] = [rob.ECE_15.mean()[sys_], rob.Brier.mean()[sys_], rob.mean_p.mean()[sys_]]
            agg.loc[sys_, "ECE_LOCO"] = t3.pooled.mean()[sys_]
            agg.loc[sys_, "ECE_match"] = t3.matched.mean()[sys_]
            agg.loc[sys_, "ECE_Platt"] = inc[sys_]
    allc = cal[cal.dataset.isin(DL + BEIR8)].groupby("system")
    agg["lat"], agg["lat95"], agg["usd"] = allc.lat_p50_ms.median(), allc.lat_p95_ms.median(), allc.usd_per_1k_q.mean()
    rows = [(k, l) for k, l in rows if k in agg.index]
    n_calib = cal[cal.dataset.isin(CALIB)].groupby("system").dataset.nunique()
    if (r3 / "r3_calib_robust.csv").exists():  # readout variants such as 'jev-choice3@strict' are not in calibration.csv
        n_calib = n_calib.combine_first(rob.dataset.nunique())
    full = [r for r, lab in rows if n_calib.get(r, 0) == len(CALIB) and "policy" not in lab]
    best = {c: agg.loc[full, c].min() for c in ["ECE", "ECE_LOCO", "ECE_match", "Brier", "ECE_Platt"] if c in agg}
    lines = [r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{1.6pt}",
             r"\caption{Calibration of $P(\text{relevant})$ on the nine collections with judged negatives "
             r"(DL19--23, TREC-COVID, DBPedia, News, Robust04; unweighted mean over collections, ECE$\times$100, 15 bins, judged pairs only). "
             r"Raw: no recalibration. Transfer (no target labels): Platt scaling fit on the other eight collections pooled (All) or only on "
             r"the same numeric-threshold group, balancing source collections (Group: DL$\to$DL, BEIR$\to$BEIR); "
             r"the group need not share identical relevance semantics. All weights pairs equally. "
             r"In-coll.: Platt with 10$\times$ repeated 5-fold query cross-fitting on the target collection (needs target labels; SD $\le$0.6). "
             r"C3/S4: Choice-3/Score-4, strict readout on every collection. $^\dagger$Sensitivity row: post-hoc per-collection readout policy "
             r"(lenient on TREC-COVID, DBPedia, News; Section~\ref{sec:setup}); excluded from bold. "
             r"$^\ast$Seven of the nine collections (no DBPedia, News). Bold: best among systems on all nine.}",
             r"\label{tab:calibration}",
             r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r" & \multicolumn{4}{c}{ECE} & & \\ \cmidrule(lr){2-5}",
             r" & & \multicolumn{2}{c}{Transfer} & & & \\ \cmidrule(lr){3-4}",
             r"System & Raw & All & Group & In-coll. & Brier & $\bar{p}$ \\", r"\midrule"]
    lat = lambda v: "--" if pd.isna(v) else (f"{v:.1f}" if v < 10 else f"{v:.0f}")
    for key, label in rows:
        r = agg.loc[key]

        def fmt(c, f):
            v = r.get(c, np.nan)
            if pd.isna(v):
                return "--"
            t = f.format(v)
            return rf"\textbf{{{t}}}" if key in full and c in best and abs(v - best[c]) < 1e-9 else t
        usd = "--" if pd.isna(r.usd) else ("0" if not r.usd else f"{r.usd:.2f}")
        lines.append(f"{label} & {fmt('ECE', '{:.1f}')} & {fmt('ECE_LOCO', '{:.1f}')} & {fmt('ECE_match', '{:.1f}')} & "
                     f"{fmt('ECE_Platt', '{:.1f}')} & {fmt('Brier', '{:.3f}')} & {r.mean_p:.2f} \\\\")
    base = agg.loc["bge-m3", "base"]
    lines += [r"\midrule", rf"\multicolumn{{7}}{{l}}{{\footnotesize Base rate of relevance (mean over the nine collections): {base:.2f}}} \\",
              r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def longdoc_table(rank: pd.DataFrame, trunc: pd.DataFrame) -> str:
    """Truncation rates and the equal-budget ablation on the long-document collections."""
    from scipy import stats

    from choicerank.data import load_qrels
    from choicerank.evaluate import per_query

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eval_all import read_trec

    p = rank.pivot(index="system", columns="dataset", values="nDCG@10")
    t = trunc.set_index("dataset")
    get = lambda s, ds: f"{p.loc[s, ds]:.1f}" if s in p.index and ds in p.columns and not np.isnan(p.loc[s, ds]) else "--"
    lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}",
             r"\caption{Long-document collections. Top: share of candidate passages longer than each model's passage budget "
             r"(XLM-R tokens; Laya English $\approx$320, cross-encoders 448 at a 512 window, bge at 2{,}048 tokens, Jev/Jeff 4{,}000 characters). "
             r"Bottom: nDCG@10 with each model's own window and with every passage cut to the same first 448 tokens (``=448''). "
             r"Qwen8B (4k chars) reads the same first 4{,}000 characters as Jev. "
             r"$\Delta$: Jev Score-4 minus bge or Qwen8B at equal budget (paired $t$-test; $^{*}$: $p<0.05$).}",
             r"\label{tab:longdoc}",
             r"\begin{tabular}{lrrr}", r"\toprule", r" & Touche & News & Robust04 \\", r"\midrule",
             r"\multicolumn{4}{l}{\textit{Passages truncated (\%)}} \\"]
    for col, lab in [("pct_over_320", r"\quad Laya ($>$320 tok)"), ("pct_over_448", r"\quad cross-enc.\ ($>$448 tok)"),
                     ("pct_over_2000", r"\quad bge-2048 ($>$2{,}000 tok)"), ("pct_over_4000ch", r"\quad Jev/Jeff ($>$4k chars)")]:
        lines.append(lab + " & " + " & ".join(f"{t.loc[d, col]:.0f}" for d in LONG) + r" \\")
    lines += [r"\midrule", r"\multicolumn{4}{l}{\textit{nDCG@10}} \\"]
    for s, lab in [("bm25", r"\quad BM25"), ("minilm", r"\quad MiniLM (512)"), ("minilm-p448", r"\quad MiniLM (=448)"),
                   ("bge-m3", r"\quad bge (512)"), ("bge-m3-p448", r"\quad bge (=448)"), ("bge-m3-2048", r"\quad bge (2{,}048)"),
                   ("jev-score4", r"\quad Jev S4 (4k chars)"), ("jev-p448-score4", r"\quad Jev S4 (=448)"),
                   ("qwen3-reranker-8B", r"\quad Qwen8B (448)"), ("qwen3-reranker-8B-4k", r"\quad Qwen8B (4k chars)")]:
        if s.startswith("qwen3-reranker-8B") and all(get(s, d) == "--" for d in LONG):
            continue  # issue #5: Qwen8B rows appear once its runs exist on these collections
        lines.append(lab + " & " + " & ".join(get(s, d) for d in LONG) + r" \\")
    def deltas(a, b):
        out = []
        for d in LONG:
            q = load_qrels(d)
            J = per_query(read_trec(RUNS_DIR / f"{d}.top100.{a}.trec"), q)
            B = per_query(read_trec(RUNS_DIR / f"{d}.top100.{b}.trec"), q)
            ks = sorted(set(J) & set(B))
            diff = 100 * np.mean([J[k] - B[k] for k in ks])
            pv = stats.ttest_rel([J[k] for k in ks], [B[k] for k in ks]).pvalue
            out.append(f"{diff:+.1f}" + ("$^{*}$" if pv < 0.05 else ""))
        return " & ".join(out)

    lines += [r"\midrule", r"$\Delta$ Jev$-$bge (=448) & " + deltas("jev-p448-score4", "bge-m3-p448") + r" \\"]
    if all((RUNS_DIR / f"{d}.top100.qwen3-reranker-8B-4k.trec").exists() for d in LONG):  # issue #5
        lines.append(r"$\Delta$ Jev$-$Qwen8B (4k chars) & " + deltas("jev-score4", "qwen3-reranker-8B-4k") + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rank = pd.read_csv(RESULTS_DIR / "ranking.csv")
    cal = pd.read_csv(RESULTS_DIR / "calibration.csv")
    sig = pd.read_csv(RESULTS_DIR / "significance.csv")
    trunc = pd.read_csv(RESULTS_DIR / "truncation.csv")
    (OUT / "ranking.tex").write_text(ranking_table(rank, sig, compact=True), encoding="utf-8")
    (OUT / "app_ranking_full.tex").write_text(ranking_table(rank, sig), encoding="utf-8")
    tr = RESULTS_DIR / "calib_transfer.csv"
    (OUT / "calibration.tex").write_text(calibration_table(cal, pd.read_csv(tr) if tr.exists() else None), encoding="utf-8")
    (OUT / "longdoc.tex").write_text(longdoc_table(rank, trunc), encoding="utf-8")
    print("wrote", *sorted(p.name for p in OUT.glob("*.tex")))


if __name__ == "__main__":
    main()
