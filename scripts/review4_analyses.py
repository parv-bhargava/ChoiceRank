"""Analyses for the journal referee report, from cached runs (no model calls).

Outputs (results/):
  r4_rag_family_tests.csv     MuSiQue Jev vs bge / Qwen8B / Qwen0.6B per policy and generator (F1, EM): question-level t,
                              family cluster-robust t (CR1, G-1 df), family sign-flip and bootstrap p; Holm over 9 policies
  r4_dl19_completion_range.csv  DL19 SPLADE++ Jev S4 - bge nDCG@10: standard, condensed, uniform and extreme completions
                              of the unjudged labels in the two top-10s
  r4_rag_calibration.csv      held-out calibration of the RQ3 probabilities on the final RAG sets (raw and Platt fit on the
                              calibration split; question-cluster bootstrap CI of the Platt ECE)
  r4_rag_threshold.csv        Jev rank/threshold mismatch and threshold F1 versus fixed top-k interpolated at equal size
  r4_prediction_intervals.csv 95%/90% prediction intervals for a new collection (Higgins-Thompson-Spiegelhalter)
  r4_musique_fd_mechanism.csv first-ranked passage on the family-disjoint split: hit, last hop, first hop, answer containment
  r4_rag_fd_hops.csv          family-disjoint F1/EM differences by hop count with family cluster-robust CIs
  r4_calib_extra.csv          per-collection raw ECE of Laya/Jeff/Clef Score-4 (and Clef Choice-3) under both readouts
  r4_judged_counts.csv        judged relevant / judged non-relevant / unjudged BM25 top-100 candidates per collection
  r4_qwen8b_4k.csv            Qwen3-Reranker-8B at 4,000 characters versus 448 tokens, and Jev S4 at equal context
  r4_qwen8b_4k_pooled.csv     pooled Jev S4 - Qwen8B with the 4,000-character run substituted where it exists
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy import stats

from choicerank import DATA_DIR, RESULTS_DIR, RUNS_DIR
from choicerank.data import ALL, CALIB, DATASETS, load_qrels
from choicerank.evaluate import apply_platt, ece, fit_platt, rank_metrics
from choicerank.rag import _load_musique, _source, load_split, musique_families, normalize
from eval_all import read_trec
from review3_analyses import _deltas
from review_analyses import load_pairs
from significance import holm

RAG = RUNS_DIR / "rag"
GENS = {"qwen_qwen-2.5-7b-instruct": "Qwen2.5-7B", "meta-llama_llama-3.1-8b-instruct": "Llama-3.1-8B",
        "qwen_qwen3-30b-a3b-instruct-2507": "Qwen3-30B"}
POL = ["top1", "top2", "top3", "top5", "adaptive", "floor2+thr0.3", "floor2+thr0.5", "thrPlatt0.2", "thrPlatt0.3"]
REPS, SEED = 20000, 13


def _cluster(d, f):
    """Question-weighted mean difference with a family cluster-robust (CR1) SE, t with G-1 df."""
    d = np.asarray(d, float); n = len(d); m = d.mean()
    r = pd.Series(d - m).groupby(np.asarray(f)).sum().to_numpy(); G = len(r)
    if G < 2:
        return dict(delta=m, lo=np.nan, hi=np.nan, p=np.nan, G=G, n=n, se=np.nan)
    se = np.sqrt((r ** 2).sum() * G / (G - 1)) / n; t = stats.t.ppf(0.975, G - 1)
    return dict(delta=m, lo=m - t * se, hi=m + t * se, p=2 * stats.t.sf(abs(m / se), G - 1), G=G, n=n, se=se)


def _per_q(split, gen, ranker):
    suffix = "" if ranker in ("jev", "bge-m3") else f".{ranker}"
    return pd.read_parquet(RAG / f"musique.{split}.per_q.{gen}{suffix}.parquet")


def rag_family_tests():
    fam = musique_families(_load_musique())
    rows = []
    for split in ("final_fd", "final"):
        for g, gl in GENS.items():
            base = _per_q(split, g, "jev")
            for other in ("bge-m3", "qwen3-reranker-8B", "qwen3-reranker-0.6B"):
                oth = _per_q(split, g, other)
                for met in ("F1", "EM"):
                    A = base.pivot(index="qid", columns="policy", values=met)
                    B = oth.pivot(index="qid", columns="policy", values=met)
                    for p in POL:
                        a, b = A[f"jev:{p}"], B[f"{other}:{p}"]; ks = a.index.intersection(b.index)
                        d = 100 * (a[ks] - b[ks]).to_numpy(); f = np.array([fam[k] for k in ks])
                        c = _cluster(d, f)
                        S = pd.Series(d).groupby(f).sum().to_numpy(); C = pd.Series(d).groupby(f).size().to_numpy()
                        flips = np.random.default_rng(SEED).choice([-1, 1], size=(REPS, len(S)))
                        p_perm = (1 + (np.abs((flips * S).sum(1) / len(d)) >= abs(d.mean()) - 1e-12).sum()) / (REPS + 1)
                        pick = np.random.default_rng(SEED).integers(0, len(S), size=(REPS, len(S)))
                        est = S[pick].sum(1) / C[pick].sum(1)
                        rows.append(dict(split=split, gen=gl, comp=other, metric=met, policy=p, n=c["n"], families=c["G"],
                                         delta=c["delta"], p_question=stats.ttest_1samp(d, 0).pvalue, p_cluster=c["p"],
                                         ci_cluster_lo=c["lo"], ci_cluster_hi=c["hi"], p_signflip=p_perm,
                                         p_boot=min(1.0, 2 * min((est <= 0).mean(), (est >= 0).mean())),
                                         boot_lo=np.percentile(est, 2.5), boot_hi=np.percentile(est, 97.5)))
    T = pd.DataFrame(rows)
    for col in ("p_question", "p_cluster", "p_signflip", "p_boot"):
        T[col + "_holm"] = T.groupby(["split", "gen", "comp", "metric"])[col].transform(lambda s: holm(list(s)))
    T.to_csv(RESULTS_DIR / "r4_rag_family_tests.csv", index=False)


def dl19_completion_range():
    def ranked(path):
        run = read_trec(path)
        return {q: sorted(d, key=lambda k: -d[k]) for q, d in run.items()}

    qrels = load_qrels("dl19")
    A, B = ranked(RUNS_DIR / "dl19-splade.top100.jev-score4.trec"), ranked(RUNS_DIR / "dl19-splade.top100.bge-m3.trec")
    disc = 1 / np.log2(np.arange(2, 12))

    def ndcg(lst, lab):
        g = np.array([lab.get(d, 0) for d in lst[:10]], float)
        ideal = np.sort(np.array([v for v in lab.values() if v > 0], float))[::-1][:10]
        idcg = (ideal * disc[:len(ideal)]).sum()
        return (g * disc[:len(g)]).sum() / idcg if idcg > 0 else 0.0

    qs = sorted(set(A) & set(B) & {q for q in qrels if any(v > 0 for v in qrels[q].values())})
    res = {k: [] for k in ("standard", "condensed", "all_1", "all_2", "all_3", "min", "max")}
    for q in qs:
        lab, a, b = qrels[q], A[q], B[q]
        res["standard"].append(ndcg(a, lab) - ndcg(b, lab))
        res["condensed"].append(ndcg([d for d in a if d in lab], lab) - ndcg([d for d in b if d in lab], lab))
        U = sorted({d for d in a[:10] + b[:10] if d not in lab})
        for g in (1, 2, 3):
            l2 = {**lab, **{d: g for d in U}}; res[f"all_{g}"].append(ndcg(a, l2) - ndcg(b, l2))
        grades = (0, 1, 2, 3) if len(U) <= 9 else (0, 3)  # linear gain: per-query extremes sit at grades 0 and 3
        vals = [ndcg(a, {**lab, **dict(zip(U, c))}) - ndcg(b, {**lab, **dict(zip(U, c))})
                for c in itertools.product(grades, repeat=len(U))]
        res["min"].append(min(vals)); res["max"].append(max(vals))
    pd.DataFrame([dict(scoring=k, jev_s4_minus_bge=100 * np.mean(v), queries=len(qs)) for k, v in res.items()]).to_csv(
        RESULTS_DIR / "r4_dl19_completion_range.csv", index=False)


SETS = [("musique", "final_fd", "calib_fd"), ("musique", "final", "calib"), ("nq", "final", "calib")]


def rag_calibration():
    rng = np.random.default_rng(0)
    out = []
    for ds, split, cal in SETS:
        for rk in ("jev", "bge-m3", "qwen3-reranker-8B", "qwen3-reranker-0.6B"):
            C = pd.read_parquet(RAG / f"{ds}.{cal}.{rk}.parquet"); E = pd.read_parquet(RAG / f"{ds}.{split}.{rk}.parquet")
            for c in (["p_noul", "p_choice3", "p_score4"] if rk == "jev" else ["p_rel"]):
                a, b = fit_platt(C[c].to_numpy(), C.gold.astype(int).to_numpy())
                y = E.gold.astype(int).to_numpy(); pr = E[c].to_numpy(); pp = apply_platt(pr, a, b)
                qids = E.qid.to_numpy(); uq = np.unique(qids); idx = {q: np.where(qids == q)[0] for q in uq}
                bs = [100 * ece(pp[s], y[s]) for s in
                      (np.concatenate([idx[q] for q in rng.choice(uq, len(uq))]) for _ in range(500))]
                out.append(dict(dataset=ds, split=split, ranker=rk, readout=c, pairs=len(y), base_rate=y.mean(),
                                ece_raw=100 * ece(pr, y), brier_raw=np.mean((pr - y) ** 2), ece_platt=100 * ece(pp, y),
                                ece_platt_lo=np.percentile(bs, 2.5), ece_platt_hi=np.percentile(bs, 97.5),
                                brier_platt=np.mean((pp - y) ** 2), mean_p_platt=pp.mean(), platt_a=a, platt_b=b))
    pd.DataFrame(out).to_csv(RESULTS_DIR / "r4_rag_calibration.csv", index=False)


def rag_threshold():
    rows = []
    for (ds, split, cal), pc in zip(SETS, ("p_noul", "p_noul", "p_choice3")):
        C = pd.read_parquet(RAG / f"{ds}.{cal}.jev.parquet"); E = pd.read_parquet(RAG / f"{ds}.{split}.jev.parquet")
        a, b = fit_platt(C[pc].to_numpy(), C.gold.astype(int).to_numpy())
        E = E.assign(pp=apply_platt(E[pc].to_numpy(), a, b))
        nonprefix = nonprefix_floor = 0; nq = 0
        for _, g in E.groupby("qid"):
            sel = (g.sort_values(["score_score4", "orig_rank"], ascending=[False, True]).pp > 0.3).to_numpy().copy()
            k = sel.sum(); nq += 1; nonprefix += k > 0 and not sel[:k].all()
            sel[:2] = True; nonprefix_floor += not sel[:sel.sum()].all()
        for g, gl in GENS.items():
            P = pd.read_parquet(RAG / f"{ds}.{split}.per_q.{g}.parquet")
            agg = P.groupby("policy").agg(F1=("F1", "mean"), n=("n", "mean"))
            ks = agg.loc[["jev:top1", "jev:top2", "jev:top3", "jev:top5"]]
            for pol in ("jev:thrPlatt0.3", "jev:thrPlatt0.2", "jev:floor2+thr0.3", "jev:adaptive"):
                n, f = agg.loc[pol, "n"], agg.loc[pol, "F1"]
                fk = np.interp(n, ks.n.to_numpy(), ks.F1.to_numpy())
                rows.append(dict(dataset=ds, split=split, readout=pc, pct_threshold_set_not_prefix=100 * nonprefix / nq,
                                 pct_floor_set_not_prefix=100 * nonprefix_floor / nq, gen=gl, policy=pol, mean_n=n,
                                 F1=100 * f, F1_topk_same_n=100 * fk, diff=100 * (f - fk)))
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "r4_rag_threshold.csv", index=False)


def prediction_intervals():
    S = pd.read_csv(RESULTS_DIR / "r3_meta_sensitivity.csv")
    rows = []
    for (a, b, coll), g in S.groupby(["A", "B", "collections"], sort=False):
        n = g[g.method == "normal"].iloc[0]; k = int(n.k)
        for lvl in (0.95, 0.90):
            half = stats.t.ppf(1 - (1 - lvl) / 2, k - 2) * np.sqrt(n.tau2 + n.se ** 2)
            rows.append(dict(A=a, B=b, collections=coll, k=k, level=lvl, mean=n["diff"], tau=np.sqrt(n.tau2),
                             pi_lo=n["diff"] - half, pi_hi=n["diff"] + half))
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "r4_prediction_intervals.csv", index=False)


def musique_fd_mechanism():
    fam = musique_families(_load_musique())
    items = {it.qid: it for it in load_split("musique", "final_fd")}
    dec = {}
    for line in open(_source("musique_ans_v1.0_dev.jsonl"), encoding="utf-8"):
        r = json.loads(line); dec["musique:" + r["id"]] = [str(h["paragraph_support_idx"]) for h in r["question_decomposition"]]
    rows = []
    for name, f, col in [("jev-score4", "jev", "score_score4"), ("jev-noul", "jev", "p_noul"), ("bge-m3", "bge-m3", "score"),
                         ("qwen3-reranker-8B", "qwen3-reranker-8B", "score"),
                         ("qwen3-reranker-0.6B", "qwen3-reranker-0.6B", "score")]:
        t = pd.read_parquet(RAG / f"musique.final_fd.{f}.parquet")
        for q, g in t.groupby("qid"):
            p = g.sort_values([col, "orig_rank"], ascending=[False, True]).pid.iloc[0]; it = items[q]
            txt = normalize(next(x.title + " " + x.text for x in it.passages if x.pid == p))
            rows.append(dict(ranker=name, qid=q, hops=it.hops, hit=any(x.pid == p and x.gold for x in it.passages),
                             last=p == dec[q][-1], first=p == dec[q][0],
                             contains=any(normalize(a) and normalize(a) in txt for a in it.answers)))
    M = pd.DataFrame(rows)
    out = [dict(ranker=r, hops="all", **{k: 100 * g[k].mean() for k in ("hit", "last", "first", "contains")}, n=len(g))
           for r, g in M.groupby("ranker")]
    out += [dict(ranker=r, hops=h, **{k: 100 * g[k].mean() for k in ("hit", "last", "first", "contains")}, n=len(g))
            for (r, h), g in M.groupby(["ranker", "hops"])]
    W = M.pivot(index="qid", columns="ranker", values=["last", "contains", "hit", "first"]).astype(float)
    for met in ("last", "contains", "hit", "first"):
        for b in ("bge-m3", "qwen3-reranker-8B"):
            c = _cluster(100 * (W[(met, "jev-score4")] - W[(met, b)]), [fam[q] for q in W.index])
            out.append(dict(ranker=f"jev-score4 minus {b}", hops="all", **{met: c["delta"]}, ci_lo=c["lo"], ci_hi=c["hi"],
                            p=c["p"], n=c["n"]))
    pd.DataFrame(out).to_csv(RESULTS_DIR / "r4_musique_fd_mechanism.csv", index=False)

    hop = pd.Series({q: it.hops for q, it in items.items()})
    rows = []
    for g, gl in GENS.items():
        base = _per_q("final_fd", g, "jev")
        for other in ("bge-m3", "qwen3-reranker-8B"):
            oth = _per_q("final_fd", g, other)
            for met in ("F1", "EM"):
                A = base.pivot(index="qid", columns="policy", values=met)
                B = oth.pivot(index="qid", columns="policy", values=met)
                for p in POL:
                    d = 100 * (A[f"jev:{p}"] - B[f"{other}:{p}"])
                    for h in ("all", 2, 3, 4):
                        dd = d if h == "all" else d[hop[d.index] == h]
                        c = _cluster(dd.to_numpy(), [fam[q] for q in dd.index])
                        rows.append(dict(gen=gl, comp=other, metric=met, policy=p, hops=h, n=c["n"], families=c["G"],
                                         delta=c["delta"], ci_lo=c["lo"], ci_hi=c["hi"], p=c["p"]))
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "r4_rag_fd_hops.csv", index=False)


def calib_extra():
    rows = []
    for s in ("laya-score4", "jeff-0.8b-score4", "clef-choice3", "clef-score4"):
        for ro in ("strict", "lenient"):
            for ds in CALIB:
                if not (RUNS_DIR / f"{ds}.top100.{s}.parquet").exists():
                    continue
                t = load_pairs(s, ds, readout=ro)
                rows.append(dict(system=s, readout=ro, dataset=ds, ece=100 * ece(t.p.to_numpy(), t.y.to_numpy()),
                                 brier=np.mean((t.p - t.y) ** 2), policy_readout=DATASETS[ds].readout))
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "r4_calib_extra.csv", index=False)


def judged_counts():
    rows = []
    for ds in ALL:
        q = load_qrels(ds); spec = DATASETS[ds]; cand = pos = neg = 0
        for line in open(DATA_DIR / "bm25" / f"{spec.run_key}_top100.jsonl", encoding="utf-8"):
            r = json.loads(line); qid = str(r.get("query", {}).get("qid", r.get("qid")))
            for c in r.get("candidates", []):
                d = str(c.get("docid")); cand += 1
                if qid in q and d in q[qid]:
                    pos += q[qid][d] >= spec.rel_threshold; neg += q[qid][d] < spec.rel_threshold
        rows.append(dict(dataset=ds, rel_threshold=spec.rel_threshold, judged_negatives=spec.dense_pool,
                         candidates=cand, judged_relevant=pos, judged_nonrelevant=neg, unjudged=cand - pos - neg))
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "r4_judged_counts.csv", index=False)


Q8, Q4K = "qwen3-reranker-8B", "qwen3-reranker-8B-4k"


def _dl(t):
    w = 1 / t["var"]; fe = np.sum(w * t.delta) / w.sum(); k = len(t)
    Q = np.sum(w * (t.delta - fe) ** 2)
    tau2 = max(0.0, (Q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    wr = 1 / (t["var"] + tau2); re = np.sum(wr * t.delta) / wr.sum(); se = np.sqrt(1 / wr.sum())
    return dict(k=k, unweighted_mean=t.delta.mean(), re_mean=re, ci95_lo=re - 1.96 * se, ci95_hi=re + 1.96 * se,
                ci90_lo=re - 1.645 * se, ci90_hi=re + 1.645 * se, p=2 * stats.norm.sf(abs(re / se)), tau=np.sqrt(tau2))


def qwen8b_4k():
    have = [d for d in ALL if (RUNS_DIR / f"{d}.top100.{Q4K}.trec").exists()]
    rows = []
    for ds in have:
        q = load_qrels(ds); thr = DATASETS[ds].rel_threshold
        nd = {s: rank_metrics(read_trec(RUNS_DIR / f"{ds}.top100.{s}.trec"), q, thr)["nDCG@10"] for s in (Q8, Q4K)}
        for a, b in ((Q4K, Q8), ("jev-score4", Q4K), ("jev-score4", Q8)):
            t = _deltas(a, b, [ds]).iloc[0]
            rows.append(dict(dataset=ds, ndcg_448=nd[Q8], ndcg_4k=nd[Q4K], A=a, B=b, n=t.n, delta=t.delta, p=t.p))
    T = pd.DataFrame(rows)
    T["p_holm"] = T.groupby(["A", "B"]).p.transform(lambda s: holm(list(s)))
    T.to_csv(RESULTS_DIR / "r4_qwen8b_4k.csv", index=False)

    pooled = []
    for label, dsets in (("all", ALL), ("no_robust04", [d for d in ALL if d != "robust04"])):
        sub = pd.concat([_deltas("jev-score4", Q4K if d in have else Q8, [d]) for d in dsets], ignore_index=True)
        pooled.append(dict(comparison="jev-score4 - qwen8B (4k where run)", collections=label, **_dl(sub)))
        pooled.append(dict(comparison="jev-score4 - qwen8B (448)", collections=label,
                           **_dl(_deltas("jev-score4", Q8, dsets))))
    pooled.append(dict(comparison="qwen8B 4k - 448", collections=f"{len(have)} with 4k", **_dl(_deltas(Q4K, Q8, have))))
    pd.DataFrame(pooled).to_csv(RESULTS_DIR / "r4_qwen8b_4k_pooled.csv", index=False)


if __name__ == "__main__":
    for name in sys.argv[1:] or ["rag_family_tests", "dl19_completion_range", "rag_calibration", "rag_threshold",
                                 "prediction_intervals", "musique_fd_mechanism", "calib_extra", "judged_counts", "qwen8b_4k"]:
        globals()[name]()
        print("done", name, flush=True)
