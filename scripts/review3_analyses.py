"""Free re-analyses requested by internal review round 3 (REVIEWS.md). All computed from cached runs.

Outputs (results/):
  r3_calib_diff.csv       paired query-bootstrap ECE difference vs bge per collection (+Holm) and mean over 9 (2,000 reps)
  r3_transfer.csv         transfer Platt: pair-pooled vs collection-balanced sources, and grading-matched sources
  r3_calib_robust.csv     ECE over 5-30 bins, equal-mass ECE, Brier reliability/resolution/uncertainty
  r3_incoll.csv           in-collection Platt with 10x repeated 5-fold query cross-fitting (mean, sd)
  r3_meta.csv             random-effects (DerSimonian-Laird) pooled nDCG@10 difference over collections
  r3_eqctx.csv            paired tests for the equal-context (448-token) condition
  r3_yesno_fallback.csv   share of yes/no judgments where neither Yes nor No was in the top-5 tokens
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from choicerank import RESULTS_DIR, RUNS_DIR
from choicerank.data import ALL, CALIB, DATASETS, load_qrels
from choicerank.evaluate import _logit, ece, per_query
from eval_all import read_trec
from review_analyses import has_run, load_pairs
from significance import holm

RNG = np.random.default_rng(0)
REPS = 2000
CAL_SYS = ["bge-m3", "minilm", "qwen3-reranker-0.6B", "laya-choice3", "laya-noul", "jeff-0.8b-choice3",
           "jev-choice3", "jev-noul", "jev-score4", "jev-choice3@strict", "jeff-0.8b-noul", "clef-choice3", "clef-noul",
           "clef-score4", "qwen235-yesno", "qwen3-reranker-4B", "qwen3-reranker-8B"]


def lp(s, d):
    """load_pairs with an optional '@strict' / '@lenient' readout suffix on the system name."""
    name, _, ro = s.partition("@")
    return load_pairs(name, d, ro or None)


def avail_of(s):
    return [d for d in CALIB if has_run(s.partition("@")[0], d)]


def fit_platt_w(p, y, w):
    z = _logit(p)

    def nll(c):
        q = np.clip(1 / (1 + np.exp(-(c[0] * z + c[1]))), 1e-9, 1 - 1e-9)
        return -np.sum(w * (y * np.log(q) + (1 - y) * np.log(1 - q))) / np.sum(w)

    return minimize(nll, x0=[1.0, 0.0], method="L-BFGS-B").x


def platt(p, c):
    return 1 / (1 + np.exp(-(c[0] * _logit(p) + c[1])))


def _groups(t):
    return {q: (g.p.to_numpy(), g.y.to_numpy()) for q, g in t.groupby("qid")}


def _ece_of(groups, qids):
    p = np.concatenate([groups[q][0] for q in qids]); y = np.concatenate([groups[q][1] for q in qids])
    return 100 * ece(p, y)


def calib_diff():
    """Paired query-cluster bootstrap of ECE(system) - ECE(bge) per collection, and of the mean over 9."""
    out = []
    variants = [("jev-choice3", None), ("jev-choice3", "strict"), ("jev-noul", None), ("laya-choice3", None),
                ("laya-noul", None), ("jev-score4", "strict")]
    for s, ro in variants:
        per_boot, rows = [], []
        for ds in CALIB:
            a, b = _groups(load_pairs(s, ds, ro)), _groups(load_pairs("bge-m3", ds))
            qids = np.array(sorted(set(a) & set(b)))
            obs = _ece_of(a, qids) - _ece_of(b, qids)
            boots = []
            for _ in range(REPS):
                pick = RNG.choice(qids, len(qids))
                boots.append(_ece_of(a, pick) - _ece_of(b, pick))
            boots = np.array(boots)
            p = min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean()))
            rows.append(dict(system=s, readout=ro or "policy", dataset=ds, diff=obs, ci_lo=np.percentile(boots, 2.5),
                             ci_hi=np.percentile(boots, 97.5), p=max(p, 1 / REPS)))
            per_boot.append(boots)
        for r, ph in zip(rows, holm([r["p"] for r in rows])):
            r["p_holm"] = ph
        m = np.mean(per_boot, axis=0)
        rows.append(dict(system=s, readout=ro or "policy", dataset="mean9", diff=np.mean([r["diff"] for r in rows]),
                         ci_lo=np.percentile(m, 2.5), ci_hi=np.percentile(m, 97.5)))
        out += rows
    pd.DataFrame(out).to_csv(RESULTS_DIR / "r3_calib_diff.csv", index=False)


def transfer():
    """Leave-one-collection-out Platt with three source definitions:
    pooled (every pair weighted equally), balanced (every source collection weighted equally),
    matched (balanced, only sources whose relevance grading threshold equals the target's)."""
    out = []
    for s in CAL_SYS:
        avail = avail_of(s)
        data = {d: lp(s, d) for d in avail}
        for d in avail:
            tgt = data[d]
            row = dict(system=s, dataset=d, raw=100 * ece(tgt.p.to_numpy(), tgt.y.to_numpy()))
            for name, srcs, bal in [("pooled", [o for o in avail if o != d], False),
                                    ("balanced", [o for o in avail if o != d], True),
                                    ("matched", [o for o in avail if o != d and
                                                 DATASETS[o].rel_threshold == DATASETS[d].rel_threshold], True)]:
                if not srcs:
                    continue
                src = pd.concat([data[o].assign(w=1 / len(data[o]) if bal else 1.0) for o in srcs])
                c = fit_platt_w(src.p.to_numpy(), src.y.to_numpy(), src.w.to_numpy())
                row[name] = 100 * ece(platt(tgt.p.to_numpy(), c), tgt.y.to_numpy())
            out.append(row)
    pd.DataFrame(out).to_csv(RESULTS_DIR / "r3_transfer.csv", index=False)


def brier_decomp(p, y, n_bins=15):
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, n_bins - 1)
    base = y.mean(); rel = res = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            rel += m.mean() * (p[m].mean() - y[m].mean()) ** 2
            res += m.mean() * (y[m].mean() - base) ** 2
    return rel, res, base * (1 - base)


def robust():
    out = []
    for s in CAL_SYS:
        for d in avail_of(s):
            t = lp(s, d); p, y = t.p.to_numpy(), t.y.to_numpy()
            rel, res, unc = brier_decomp(p, y)
            row = dict(system=s, dataset=d, base_rate=y.mean(), mean_p=p.mean(), n_pairs=len(y), ECE_adapt=100 * ece(p, y, adaptive=True),
                       Brier=np.mean((p - y) ** 2), reliability=rel, resolution=res, uncertainty=unc)
            for nb in (5, 10, 15, 20, 30):
                row[f"ECE_{nb}"] = 100 * ece(p, y, n_bins=nb)
            out.append(row)
    pd.DataFrame(out).to_csv(RESULTS_DIR / "r3_calib_robust.csv", index=False)


def incoll(reps=10, folds=5):
    out = []
    for s in CAL_SYS:
        for d in avail_of(s):
            t = lp(s, d); qids = np.array(sorted(t.qid.unique())); vals = []
            for r in range(reps):
                rng = np.random.default_rng(r); fold = dict(zip(rng.permutation(qids), np.arange(len(qids)) % folds))
                f = t.qid.map(fold).to_numpy(); q = np.empty(len(t))
                for k in range(folds):
                    tr, te = f != k, f == k
                    c = fit_platt_w(t.p.to_numpy()[tr], t.y.to_numpy()[tr], np.ones(tr.sum()))
                    q[te] = platt(t.p.to_numpy()[te], c)
                vals.append(100 * ece(q, t.y.to_numpy()))
            out.append(dict(system=s, dataset=d, ECE_incoll_mean=np.mean(vals), ECE_incoll_sd=np.std(vals)))
    pd.DataFrame(out).to_csv(RESULTS_DIR / "r3_incoll.csv", index=False)


def _deltas(a, b, datasets):
    rows = []
    for ds in datasets:
        fa, fb = RUNS_DIR / f"{ds}.top100.{a}.trec", RUNS_DIR / f"{ds}.top100.{b}.trec"
        if not (fa.exists() and fb.exists()):
            continue
        q = load_qrels(ds)
        A, B = per_query(read_trec(fa), q), per_query(read_trec(fb), q)
        ks = sorted(set(A) & set(B)); d = 100 * np.array([A[k] - B[k] for k in ks])
        rows.append(dict(dataset=ds, n=len(d), delta=d.mean(), var=d.var(ddof=1) / len(d),
                         p=stats.ttest_rel([A[k] for k in ks], [B[k] for k in ks]).pvalue))
    return pd.DataFrame(rows)


def meta():
    """DerSimonian-Laird random-effects pooling of per-collection mean nDCG@10 differences."""
    out = []
    pairs = [("jev-score4", "bge-m3"), ("jev-score4", "qwen3-reranker-0.6B"), ("jev-score4", "minilm"),
             ("jev-noul", "bge-m3"), ("jev-choice3", "bge-m3"), ("clef-score4", "bge-m3"), ("jev-score4", "clef-score4"),
             ("jev-score4", "qwen3-reranker-8B"), ("jev-score4", "qwen3-reranker-4B"), ("qwen3-reranker-8B", "bge-m3"),
             ("qwen235-graded", "qwen235-yesno")]
    for a, b in pairs:
        for label, dsets in [("all", ALL), ("no_robust04", [d for d in ALL if d != "robust04"])]:
            t = _deltas(a, b, dsets)
            if len(t) < 3:
                continue
            w = 1 / t["var"]; fe = np.sum(w * t.delta) / w.sum()
            Q = np.sum(w * (t.delta - fe) ** 2); k = len(t)
            tau2 = max(0.0, (Q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
            wr = 1 / (t["var"] + tau2); re = np.sum(wr * t.delta) / wr.sum(); se = np.sqrt(1 / wr.sum())
            out.append(dict(A=a, B=b, collections=label, k=k, unweighted_mean=t.delta.mean(), re_mean=re,
                            ci_lo=re - 1.96 * se, ci_hi=re + 1.96 * se, p=2 * stats.norm.sf(abs(re / se)),
                            tau=np.sqrt(tau2), I2=max(0.0, (Q - (k - 1)) / Q) if Q > 0 else 0.0))
    pd.DataFrame(out).to_csv(RESULTS_DIR / "r3_meta.csv", index=False)


def eqctx():
    out = []
    for fmt in ("choice3", "noul", "score4"):
        for a, b in [(f"jev-p448-{fmt}", "bge-m3-p448"), (f"jev-{fmt}", "bge-m3"), (f"jev-p448-{fmt}", "bge-m3-2048")]:
            t = _deltas(a, b, ["touche", "news", "robust04"])
            if t.empty:
                continue
            t["p_holm"] = holm(list(t.p))
            out += [dict(A=a, B=b, **r) for r in t.drop(columns="var").to_dict("records")]
    pd.DataFrame(out).to_csv(RESULTS_DIR / "r3_eqctx.csv", index=False)


def yesno_fallback():
    con = sqlite3.connect(f"file:{RUNS_DIR / 'cache.sqlite'}?mode=ro", uri=True, timeout=60)
    out = []
    for judge, in con.execute("select distinct judge from j where judge like 'llm:%'"):
        n = miss = only = 0
        for (payload,) in con.execute("select payload from j where judge = ?", (judge,)):
            raw = json.loads(payload).get("raw", {}); n += 1
            py, pn = raw.get("p_yes", 0), raw.get("p_no", 0)
            miss += (py + pn) == 0; only += (py == 0) != (pn == 0)
        out.append(dict(judge=judge, n=n, neither_pct=100 * miss / max(n, 1), only_one_pct=100 * only / max(n, 1)))
    pd.DataFrame(out).to_csv(RESULTS_DIR / "r3_yesno_fallback.csv", index=False)


QWEN_INSTRUCTIONS = {  # same model, three task framings (round-3 mechanism check)
    "default": "Given a web search query, retrieve relevant passages that answer the query",
    "relevant": "Given a question, retrieve passages that are relevant to the question",
    "answers": "Given a question, judge whether the passage contains the answer to the question",
}


def musique_mechanism():
    """MuSiQue final: top-1 hop position for Jev (every readout), bge, and Qwen3-Reranker-0.6B under three
    instructions. Separates the effect of an 'answers?' framing from the effect of the model."""
    import json as _json

    from choicerank.judges.crossencoder import Qwen3RerankerJudge
    from choicerank.rag import load_split, normalize
    items = {it.qid: it for it in load_split("musique", "final")}
    dec = {}
    for line in open(Path(__file__).resolve().parent.parent / "data" / "rag" / "musique_ans_v1.0_dev.jsonl", encoding="utf-8"):
        r = _json.loads(line)
        dec["musique:" + r["id"]] = [str(h["paragraph_support_idx"]) for h in r["question_decomposition"]]
    scores = {}
    jev = pd.read_parquet(RUNS_DIR / "rag" / "musique.final.jev.parquet")
    for col in ("score_score4", "p_choice3", "p_noul"):
        scores[f"jev:{col}"] = jev[["qid", "pid", "orig_rank"]].assign(s=jev[col])
    bge = pd.read_parquet(RUNS_DIR / "rag" / "musique.final.bge-m3.parquet")
    scores["bge-m3"] = bge[["qid", "pid", "orig_rank"]].assign(s=bge.score)
    judge = Qwen3RerankerJudge("0.6B")
    order = {it.qid: {p.pid: i for i, p in enumerate(it.passages)} for it in items.values()}
    for name, ins in QWEN_INSTRUCTIONS.items():
        judge.INSTRUCTION = ins
        rows = []
        for q, it in items.items():
            res = judge._judge(it.question, [f"Title: {p.title} Content: {p.text}" for p in it.passages])
            rows += [(q, p.pid, order[q][p.pid], j.score) for p, j in zip(it.passages, res)]
        scores[f"qwen3rr:{name}"] = pd.DataFrame(rows, columns=["qid", "pid", "orig_rank", "s"])
    out = []
    for name, t in scores.items():
        hit = last = first = ans = 0
        for q, g in t.groupby("qid"):
            top1 = g.sort_values(["s", "orig_rank"], ascending=[False, True]).pid.iloc[0]
            it = items[q]
            hit += any(p.pid == top1 and p.gold for p in it.passages)
            last += top1 == dec[q][-1]; first += top1 == dec[q][0]
            txt = normalize(next(p.text for p in it.passages if p.pid == top1))
            ans += any(normalize(a) and normalize(a) in txt for a in it.answers)
        n = t.qid.nunique()
        out.append(dict(ranker=name, hit1=100 * hit / n, last_hop1=100 * last / n, first_hop1=100 * first / n,
                        contains_answer1=100 * ans / n))
    pd.DataFrame(out).to_csv(RESULTS_DIR / "r3_musique_mechanism.csv", index=False)


def _re_pool(t):
    """DerSimonian-Laird random-effects pooled mean difference and its SE from per-collection deltas/variances."""
    w = 1 / t["var"]; fe = np.sum(w * t.delta) / w.sum()
    Q = np.sum(w * (t.delta - fe) ** 2); k = len(t)
    tau2 = max(0.0, (Q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    wr = 1 / (t["var"] + tau2)
    return np.sum(wr * t.delta) / wr.sum(), np.sqrt(1 / wr.sum())


EQ_MARGIN = 2.0  # nDCG@10 points


def equiv():
    """Equivalence for the paper's "on par" claims: random-effects pooled difference with 95% CI, 90% CI and
    TOST at +/-2 nDCG@10 points (equivalent if the 90% CI lies inside the margin)."""
    from choicerank.data import DL
    out = []
    claims = [(f"jev-{f}", b, "DL19-23", DL) for f in ("score4", "noul", "choice3") for b in ("bge-m3", "minilm")]
    claims += [(f"clef-{f}", "qwen27base-yesno", "DL19/20+COVID", ["dl19", "dl20", "trec-covid"]) for f in ("choice3", "noul", "score4")]
    claims += [(f"jev-{f}", "qwen3-reranker-8B", "All-13", list(__import__("choicerank.data", fromlist=["ALL"]).ALL))
               for f in ("score4", "noul", "choice3")]
    claims += [(f"jev-{f}", f"clef-{f}", "Clef-7", ["dl19", "dl20", "dl21", "dl22", "dl23", "trec-covid", "robust04"])
               for f in ("choice3", "noul", "score4")]
    for a, b, label, dsets in claims:
        t = _deltas(a, b, dsets)
        m, se = _re_pool(t)
        z = (m + EQ_MARGIN) / se, (EQ_MARGIN - m) / se
        out.append(dict(A=a, B=b, collections=label, k=len(t), diff=m, ci95_lo=m - 1.96 * se, ci95_hi=m + 1.96 * se,
                        ci90_lo=m - 1.645 * se, ci90_hi=m + 1.645 * se,
                        p_tost=max(stats.norm.sf(z[0]), stats.norm.sf(z[1])),
                        equivalent=bool(m - 1.645 * se > -EQ_MARGIN and m + 1.645 * se < EQ_MARGIN)))
    pd.DataFrame(out).to_csv(RESULTS_DIR / "r3_equiv.csv", index=False)


if __name__ == "__main__":
    for name in sys.argv[1:] or ["transfer", "robust", "meta", "eqctx", "yesno_fallback", "incoll", "calib_diff", "musique_mechanism"]:
        globals()[name]()
        print("done", name, flush=True)
