"""RQ3 final analysis (locked comparisons) -> results/rag_final_summary.csv, results/rag_final_tests.csv,
paper/tables/rag.tex

Pre-registered comparisons (PROGRESS.md, locked before final-split generation):
  (a) ranker effect: Jev vs bge under the same policy (top1/2/3/5, adaptive, floor2+thr0.3/0.5, thrPlatt0.2/0.3)
  (b) calibrated rule: jev:floor2+thr0.3 vs jev:top3 and vs jev:adaptive
  (c) generator input tokens
Paired two-sided t-tests over questions on F1. Holm correction is applied separately to family (a) and family (b)
within each (dataset, generator); the four-comparison family also includes exploratory top-2/top-5 comparisons
not in the lock record. TOST uses a post-hoc +/-2 F1 margin without correction; pooling over generators is exploratory.
(Revised 2026-10-04 after internal review: families were previously mixed.)
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

from choicerank import RESULTS_DIR, ROOT, RUNS_DIR

GENS = {"qwen/qwen-2.5-7b-instruct": "Qwen2.5-7B", "meta-llama/llama-3.1-8b-instruct": "Llama-3.1-8B",
        "qwen/qwen3-30b-a3b-instruct-2507": "Qwen3-30B-A3B"}
DS = {"musique": "MuSiQue", "nq": "NQ"}
SHARED = ["top1", "top2", "top3", "top5", "adaptive", "floor2+thr0.3", "floor2+thr0.5", "thrPlatt0.2", "thrPlatt0.3"]


def holm(ps):
    order, m, adj, run = np.argsort(ps), len(ps), [0.0] * len(ps), 0.0
    for i, k in enumerate(order):
        run = max(run, (m - i) * ps[k])
        adj[k] = min(1.0, run)
    return adj


B_REFS = ["jev:top2", "jev:top3", "jev:top5", "jev:adaptive"]
PRIMARY_B_REFS = {"jev:top3", "jev:adaptive"}
MARGIN = 2.0  # F1 points, equivalence margin for TOST


def paired(x, y, meta):
    """Paired t-test, 95% CI of the mean difference, and TOST equivalence p-value at +/-MARGIN (F1 points)."""
    d = 100 * (x - y).to_numpy()
    n, m, se = len(d), d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    t = stats.t.ppf(0.975, n - 1)
    p_lo = 1 - stats.t.cdf((m + MARGIN) / se, n - 1)  # H0: diff <= -margin
    p_hi = stats.t.cdf((m - MARGIN) / se, n - 1)      # H0: diff >= +margin
    return {**meta, "delta_F1": m, "ci_lo": m - t * se, "ci_hi": m + t * se,
            "p": stats.ttest_rel(x, y).pvalue, "p_tost": max(p_lo, p_hi)}


def per_question(ds, gen, rankers=None):
    """Validate source completeness and rebuild stale or unverified per-question scores.
    rankers: None for the original jev/bge grid, or e.g. ['qwen3-reranker-8B'] for an additional ranker (issue #4)."""
    from scripts.rag_eval import file_hash, validate_grid
    from scripts.rag_generate import RANKERS, final_policies, gen_path
    from choicerank.rag import load_split

    rankers = rankers or RANKERS
    p = gen_path(ds, "final", gen, rankers, kind="per_q")
    source = gen_path(ds, "final", gen, rankers)
    validate_grid(pd.read_parquet(source), [it.qid for it in load_split(ds, "final")], final_policies(rankers))
    metadata = p.with_suffix(".metadata.json")
    provenance = json.loads(metadata.read_text(encoding="utf-8")) if metadata.exists() else {}
    if not p.exists() or not provenance.get("complete_grid") or provenance.get("generation_sha256") != file_hash(source):
        import subprocess

        extra = [] if rankers == RANKERS else ["--rankers", *rankers]
        subprocess.run([sys.executable, "scripts/rag_eval.py", "--dataset", ds, "--split", "final", "--gen", gen, *extra],
                       cwd=ROOT, check=True, capture_output=True)
    return pd.read_parquet(p)


EXTRA_RANKERS = ["qwen3-reranker-8B", "qwen3-reranker-0.6B"]  # issue #4: downstream comparison with Qwen rerankers


def extra_ranker_tests(ds, gen, wide_jev):
    """Summary rows and family (c) tests, Jev vs each additional ranker under the nine locked policies (Holm per
    dataset x generator x ranker), for the rankers whose final generations exist."""
    from scripts.rag_generate import gen_path

    summary, tests = [], []
    for rk in EXTRA_RANKERS:
        if not gen_path(ds, "final", gen, [rk]).exists():
            continue
        per = per_question(ds, gen, [rk])
        agg = per.groupby("policy").agg(F1=("F1", "mean"), EM=("EM", "mean"), n=("n", "mean"),
                                        tokens=("tokens", "mean"), empty=("empty", "mean"))
        for pol, r in agg.iterrows():
            summary.append(dict(dataset=ds, gen=GENS[gen], policy=pol, F1=100 * r["F1"], EM=100 * r["EM"],
                                n=r["n"], tokens=r["tokens"], empty=100 * r["empty"]))
        wide = per.pivot(index="qid", columns="policy", values="F1")
        rows = [paired(wide_jev[f"jev:{p}"], wide[f"{rk}:{p}"],
                       dict(dataset=ds, gen=GENS[gen], comparison=f"c: jev vs {rk}", A=f"jev:{p}", B=f"{rk}:{p}",
                            design="added_after_final_results", equivalence_design="posthoc_margin_uncorrected"))
                for p in SHARED]
        for r, ph in zip(rows, holm([r["p"] for r in rows])):
            r["p_holm"] = ph
        tests += rows
    return summary, tests


def main():
    summary, tests, wide_by = [], [], {}
    for ds in DS:
        for gen in GENS:
            if not (RUNS_DIR / "rag" / f"{ds}.final.gen.{gen.replace('/', '_')}.parquet").exists():
                print(f"skip {ds} x {gen}: not generated yet")
                continue
            per = per_question(ds, gen)
            agg = per.groupby("policy").agg(F1=("F1", "mean"), EM=("EM", "mean"), n=("n", "mean"),
                                            tokens=("tokens", "mean"), empty=("empty", "mean"))
            for pol, r in agg.iterrows():
                summary.append(dict(dataset=ds, gen=GENS[gen], policy=pol, F1=100 * r["F1"], EM=100 * r["EM"],
                                    n=r["n"], tokens=r["tokens"], empty=100 * r["empty"]))  # r["empty"], not Series.empty
            wide = per.pivot(index="qid", columns="policy", values="F1")
            wide_by[(ds, gen)] = wide
            fams = {"a: jev vs bge": [(f"jev:{p}", f"bge-m3:{p}") for p in SHARED],
                    "b: calibrated rule": [("jev:floor2+thr0.3", p) for p in B_REFS]}
            for kind, pairs in fams.items():
                rows = [paired(wide[a], wide[b], dict(dataset=ds, gen=GENS[gen], comparison=kind, A=a, B=b,
                                                    design="primary_after_pilot" if kind.startswith("a") or b in PRIMARY_B_REFS
                                                    else "exploratory_not_in_lock_record",
                                                    equivalence_design="posthoc_margin_uncorrected"))
                        for a, b in pairs]
                for r, ph in zip(rows, holm([r["p"] for r in rows])):
                    r["p_holm"] = ph
                tests += rows
            s_x, t_x = extra_ranker_tests(ds, gen, wide)
            summary += s_x
            tests += t_x
    # (b) pooled over generators: per-question mean F1 across the 3 generators
    for ds in DS:
        ws = [w for (d, g), w in wide_by.items() if d == ds]
        if len(ws) == len(GENS):
            mean = sum(ws) / len(ws)
            rows = [paired(mean["jev:floor2+thr0.3"], mean[b], dict(dataset=ds, gen="pooled", comparison="b: calibrated rule",
                                                                    A="jev:floor2+thr0.3", B=b, design="exploratory_pooled",
                                                                    equivalence_design="posthoc_margin_uncorrected")) for b in B_REFS]
            for r, ph in zip(rows, holm([r["p"] for r in rows])):
                r["p_holm"] = ph
            tests += rows
    S, T = pd.DataFrame(summary), pd.DataFrame(tests)
    S.to_csv(RESULTS_DIR / "rag_final_summary.csv", index=False)
    T.to_csv(RESULTS_DIR / "rag_final_tests.csv", index=False)
    show = ["none", "all", "gold", "jev:top1", "jev:top3", "jev:top5", "jev:adaptive", "jev:floor2+thr0.3",
            "jev:floor2+thr0.5", "jev:thrPlatt0.3", "bge-m3:top3", "bge-m3:adaptive", "bge-m3:floor2+thr0.3"]
    piv = S[S.policy.isin(show)].pivot_table(index="policy", columns=["dataset", "gen"], values="F1").reindex(show)
    print(piv.round(1).to_string())
    print()
    print(T.round(4).to_string(index=False))
    write_latex(S, T)


def write_latex(S, T):
    rows = [("none", "No context"), ("all", "All 20 passages"), ("gold", "Gold passages (oracle)"),
            (None, "Jev 1.13"), ("jev:top1", r"\quad top-1"), ("jev:top3", r"\quad top-3"), ("jev:top5", r"\quad top-5"),
            ("jev:adaptive", r"\quad adaptive-$k$"), ("jev:thrPlatt0.3", r"\quad $P>0.3$"),
            ("jev:floor2+thr0.3", r"\quad top-2 $+$ $P>0.3$"),
            (None, "bge-reranker-v2-m3"), ("bge-m3:top1", r"\quad top-1"), ("bge-m3:top3", r"\quad top-3"),
            ("bge-m3:adaptive", r"\quad adaptive-$k$"), ("bge-m3:floor2+thr0.3", r"\quad top-2 $+$ $P>0.3$")]
    for rk, name in [("qwen3-reranker-8B", "Qwen3-Reranker-8B (4-bit)"), ("qwen3-reranker-0.6B", "Qwen3-Reranker-0.6B")]:
        if (S.policy == f"{rk}:top1").any():  # issue #4: shown once its final generations exist
            rows += [(None, name), (f"{rk}:top1", r"\quad top-1"), (f"{rk}:top3", r"\quad top-3"),
                     (f"{rk}:adaptive", r"\quad adaptive-$k$"), (f"{rk}:floor2+thr0.3", r"\quad top-2 $+$ $P>0.3$")]
    cols = [(d, g) for d in DS for g in GENS.values() if ((S.dataset == d) & (S.gen == g)).any()]
    sig = T[T.comparison.str.startswith("a")].set_index(["dataset", "gen", "A"])
    out = [r"\begin{table*}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
           r"\caption{RQ3: answer F1 on the original 1{,}000 ID-disjoint final questions per dataset; MuSiQue component families overlap design splits. "
           r"Avg.\ $n$: mean number of selected passages (independent of generator). $^\dagger$: Jev significantly better than "
           r"bge-reranker-v2-m3 under the same policy in the original question-level paired $t$-test (Holm-corrected, $p<0.05$); these tests do not account for family dependence. "
           r"Displayed policies and ranker-effect comparisons were fixed after separate pilot questions. "
           r"Thresholds use a readout selected with labelled calibration questions; the $P>0.3$ rows are Platt-scaled.}",
           r"\label{tab:rag}",
           r"\begin{tabular}{l" + "r" * len(cols) + "rr}", r"\toprule",
           " & " + " & ".join(rf"\multicolumn{{{sum(1 for c in cols if c[0] == d)}}}{{c}}{{{DS[d]}}}" for d in DS
                                if any(c[0] == d for c in cols)) + r" & \multicolumn{2}{c}{Avg.\ $n$} \\",
           "Policy & " + " & ".join(g.split("-")[0] + "-" + g.split("-")[1] if "-" in g else g for _, g in cols)
           + r" & MuSiQue & NQ \\", r"\midrule"]
    for key, label in rows:
        if key is None:
            out.append(rf"\multicolumn{{{len(cols) + 3}}}{{l}}{{\textit{{{label}}}}} \\")
            continue
        cells = []
        for d, g in cols:
            v = S[(S.dataset == d) & (S.gen == g) & (S.policy == key)].F1
            txt = f"{v.iloc[0]:.1f}" if len(v) else "--"
            if (d, g, key) in sig.index and sig.loc[(d, g, key)].p_holm < 0.05 and sig.loc[(d, g, key)].delta_F1 > 0:
                txt += r"$^\dagger$"
            cells.append(txt)
        ns = []
        for d in DS:
            v = S[(S.dataset == d) & (S.gen == "Qwen2.5-7B") & (S.policy == key)].n
            ns.append(f"{v.iloc[0]:.1f}" if len(v) else "--")
        out.append(f"{label} & " + " & ".join(cells) + " & " + " & ".join(ns) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (ROOT / "paper" / "tables" / "rag.tex").write_text("\n".join(out), encoding="utf-8")
    print("wrote paper/tables/rag.tex")


if __name__ == "__main__":
    main()
