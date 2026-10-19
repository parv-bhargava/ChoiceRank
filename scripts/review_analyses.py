"""Free re-analyses requested by internal review round 1 (REVIEWS.md). All computed from cached runs.

Outputs (results/):
  calib_readouts.csv   ECE/Brier of decision models under BOTH strict and lenient readouts, every dense collection (A1)
  calib_transfer.csv   label-free-for-target calibration: Platt fit on the OTHER collections (leave-one-collection-out)
                       and on DL19+20 only, applied to each target collection (B1)
  calib_bounds.csv     judged-only vs unjudged=0 ECE, and class-conditional mean P(rel) (B2)
  calib_ci.csv         query-cluster bootstrap 95% CIs for ECE per collection and for the mean ECE difference vs bge (A5)
  judged.csv           Judged@10 and condensed (judged-only) nDCG'@10 per system and collection (C1)
  rag_selection.csv    generator-free selection quality: gold recall@k / all-gold@k per ranker (C5)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from choicerank import RESULTS_DIR, RUNS_DIR
from choicerank.data import ALL, CALIB, DATASETS, DL, load_qrels
from choicerank.evaluate import apply_platt, calib_metrics, ece, fit_platt, rank_metrics
from eval_all import read_trec

SYSTEMS = ["minilm", "bge-m3", "qwen3-reranker-0.6B", "qwen235-yesno", "qwen27base-yesno", "laya-choice3", "laya-noul", "jeff-0.8b-choice3", "jeff-0.8b-noul",
           "clef-choice3", "clef-noul", "clef-score4", "jev-choice3", "jev-noul", "jev-score4"]
RNG = np.random.default_rng(0)


def has_run(system, ds):
    return (RUNS_DIR / f"{ds}.top100.{system}.parquet").exists()


def load_pairs(system, ds, readout=None, dense=None):
    """Judged pairs -> DataFrame(qid, p, y). readout: None (dataset default) | 'strict' | 'lenient'."""
    spec = DATASETS[ds]
    df = pd.read_parquet(RUNS_DIR / f"{ds}.top100.{system}.parquet")
    df["qid"], df["docid"] = df.qid.astype(str), df.docid.astype(str)
    ro = readout or spec.readout
    if ro == "lenient" and "p_rel_lenient" in df:
        df["p_rel"] = df["p_rel_lenient"]
    qrels = load_qrels(ds)
    dense = spec.dense_pool if dense is None else dense
    rows = []
    for q, d, p in zip(df.qid, df.docid, df.p_rel):
        if q not in qrels:
            continue
        if d in qrels[q]:
            rows.append((q, p, int(qrels[q][d] >= spec.rel_threshold)))
        elif not dense:
            rows.append((q, p, 0))
    return pd.DataFrame(rows, columns=["qid", "p", "y"])


def readouts():
    out = []
    for s in SYSTEMS:
        if not s.startswith(("laya", "jeff", "jev", "clef")):
            continue
        for ds in [d for d in CALIB if has_run(s, d)]:
            for ro in ("strict", "lenient"):
                t = load_pairs(s, ds, ro)
                m = calib_metrics(t.p.to_numpy(), t.y.to_numpy())
                out.append(dict(system=s, dataset=ds, readout=ro, used=ro == DATASETS[ds].readout, ECE=m["ECE"], Brier=m["Brier"]))
    pd.DataFrame(out).to_csv(RESULTS_DIR / "calib_readouts.csv", index=False)


def transfer():
    out = []
    for s in SYSTEMS:
        avail = [ds for ds in CALIB if has_run(s, ds)]
        if len(avail) < 3:
            continue
        data = {ds: load_pairs(s, ds) for ds in avail}
        for ds in avail:
            tgt = data[ds]
            src = pd.concat([data[o] for o in avail if o != ds])
            a, b = fit_platt(src.p.to_numpy(), src.y.to_numpy())
            loco = calib_metrics(apply_platt(tgt.p.to_numpy(), a, b), tgt.y.to_numpy())
            row = dict(system=s, dataset=ds, ECE_raw=calib_metrics(tgt.p.to_numpy(), tgt.y.to_numpy())["ECE"],
                       ECE_LOCO=loco["ECE"], Brier_LOCO=loco["Brier"])
            if ds not in ("dl19", "dl20") and "dl19" in data and "dl20" in data:
                dlsrc = pd.concat([data["dl19"], data["dl20"]])
                a2, b2 = fit_platt(dlsrc.p.to_numpy(), dlsrc.y.to_numpy())
                row["ECE_DLfit"] = calib_metrics(apply_platt(tgt.p.to_numpy(), a2, b2), tgt.y.to_numpy())["ECE"]
            out.append(row)
    pd.DataFrame(out).to_csv(RESULTS_DIR / "calib_transfer.csv", index=False)


STRICT_VARIANTS = ["laya-choice3", "jeff-0.8b-choice3", "clef-choice3", "clef-score4", "jev-choice3", "jev-score4"]


def bounds():
    """Also reports the strict readout on every collection ('<system>@strict') for the Choice-3/Score-4 systems."""
    out = []
    for name in SYSTEMS + [f"{s}@strict" for s in STRICT_VARIANTS]:
        s, _, ro = name.partition("@")
        for ds in [d for d in CALIB if has_run(s, d)]:
            j, u = load_pairs(s, ds, ro or None), load_pairs(s, ds, ro or None, dense=False)
            out.append(dict(system=name, dataset=ds, ECE_judged=100 * ece(j.p.to_numpy(), j.y.to_numpy()),
                            ECE_unjudged0=100 * ece(u.p.to_numpy(), u.y.to_numpy()),
                            mean_p_pos=j[j.y == 1].p.mean(), mean_p_neg=j[j.y == 0].p.mean(),
                            judged_share=len(j) / max(len(u), 1)))
    pd.DataFrame(out).to_csv(RESULTS_DIR / "calib_bounds.csv", index=False)


def boot_ece(t, reps=1000):
    """Query-cluster bootstrap of ECE."""
    groups = {q: g for q, g in t.groupby("qid")}
    qids = np.array(list(groups))
    vals = []
    for _ in range(reps):
        s = pd.concat([groups[q] for q in RNG.choice(qids, len(qids))])
        vals.append(100 * ece(s.p.to_numpy(), s.y.to_numpy()))
    return np.percentile(vals, [2.5, 97.5])


def ci():
    out = []
    for s in ["bge-m3", "minilm", "jev-choice3", "jev-noul", "laya-choice3", "laya-noul"]:
        for ds in CALIB:
            t = load_pairs(s, ds)
            lo, hi = boot_ece(t, 500)
            out.append(dict(system=s, dataset=ds, ECE=100 * ece(t.p.to_numpy(), t.y.to_numpy()), ci_lo=lo, ci_hi=hi))
    # mean ECE difference vs bge over the 9 collections (bootstrap queries within each collection)
    for s in ["jev-choice3", "jev-noul", "laya-choice3", "laya-noul"]:
        diffs = []
        tabs = {ds: (load_pairs(s, ds), load_pairs("bge-m3", ds)) for ds in CALIB}
        for _ in range(300):
            per = []
            for ds, (a, b) in tabs.items():
                qids = np.array(sorted(set(a.qid) & set(b.qid)))
                pick = RNG.choice(qids, len(qids))
                ga, gb = a.groupby("qid"), b.groupby("qid")
                sa = pd.concat([ga.get_group(q) for q in pick])
                sb = pd.concat([gb.get_group(q) for q in pick])
                per.append(100 * (ece(sa.p.to_numpy(), sa.y.to_numpy()) - ece(sb.p.to_numpy(), sb.y.to_numpy())))
            diffs.append(np.mean(per))
        observed = np.mean([100 * (ece(a.p.to_numpy(), a.y.to_numpy()) - ece(b.p.to_numpy(), b.y.to_numpy()))
                            for a, b in tabs.values()])
        out.append(dict(system=s, dataset="mean9_minus_bge", ECE=observed, ci_lo=np.percentile(diffs, 2.5),
                        ci_hi=np.percentile(diffs, 97.5)))
    pd.DataFrame(out).to_csv(RESULTS_DIR / "calib_ci.csv", index=False)


def judged():
    out = []
    systems = ["bm25"] + SYSTEMS + [s for s in ["qwen3-reranker-0.6B", "qwen235-yesno", "qwen27base-yesno"] if s not in SYSTEMS]
    for s in systems:
        for ds in ALL:
            f = RUNS_DIR / f"{ds}.top100.{s}.trec"
            if not f.exists():
                continue
            run, qrels = read_trec(f), load_qrels(ds)
            j10, cond = [], {}
            for q, docs in run.items():
                if q not in qrels:
                    continue
                top = sorted(docs, key=lambda d: -docs[d])[:10]
                j10.append(np.mean([d in qrels[q] for d in top]))
                cond[q] = {d: v for d, v in docs.items() if d in qrels[q]}  # condensed list (judged only)
            out.append(dict(system=s, dataset=ds, judged_at_10=100 * np.mean(j10),
                            nDCG10_condensed=rank_metrics(cond, qrels, DATASETS[ds].rel_threshold)["nDCG@10"]))
    pd.DataFrame(out).to_csv(RESULTS_DIR / "judged.csv", index=False)


def rag_selection():
    from choicerank.rag import load_split

    out = []
    for ds in ("musique", "nq"):
        items = {it.qid: it for it in load_split(ds, "final")}
        for r, scol in (("jev", "score_score4"), ("bge-m3", "score")):
            t = pd.read_parquet(RUNS_DIR / "rag" / f"{ds}.final.{r}.parquet")
            for k in (1, 2, 3, 5):
                rec, allg = [], []
                for q, g in t.groupby("qid"):
                    gold = {p.pid for p in items[q].passages if p.gold}
                    if not gold:
                        continue
                    top = set(g.sort_values([scol, "orig_rank"], ascending=[False, True]).pid[:k])
                    rec.append(len(top & gold) / len(gold))
                    allg.append(float(gold <= top))
                out.append(dict(dataset=ds, ranker=r, k=k, gold_recall=100 * np.mean(rec), all_gold=100 * np.mean(allg)))
    pd.DataFrame(out).to_csv(RESULTS_DIR / "rag_selection.csv", index=False)


def rag_hops():
    """MuSiQue final split: is each ranker's top-1 the first-hop or the last (answer-bearing) supporting paragraph?"""
    import json

    from choicerank.rag import load_split, normalize
    items = {it.qid: it for it in load_split("musique", "final")}
    dec = {}
    for line in open(Path(__file__).resolve().parent.parent / "data" / "rag" / "musique_ans_v1.0_dev.jsonl", encoding="utf-8"):
        r = json.loads(line)
        dec["musique:" + r["id"]] = [str(h["paragraph_support_idx"]) for h in r["question_decomposition"]]
    out = []
    for rk, scol in (("jev", "score_score4"), ("bge-m3", "score")):
        t = pd.read_parquet(RUNS_DIR / "rag" / f"musique.final.{rk}.parquet")
        last, first, ans = [], [], []
        for q, g in t.groupby("qid"):
            top1 = g.sort_values([scol, "orig_rank"], ascending=[False, True]).pid.iloc[0]
            last.append(top1 == dec[q][-1]); first.append(top1 == dec[q][0])
            txt = normalize(next(p.text for p in items[q].passages if p.pid == top1))
            ans.append(any(normalize(a) and normalize(a) in txt for a in items[q].answers))
        out.append(dict(ranker=rk, top1_last_hop=100 * np.mean(last), top1_first_hop=100 * np.mean(first),
                        top1_contains_answer=100 * np.mean(ans)))
    pd.DataFrame(out).to_csv(RESULTS_DIR / "rag_hops.csv", index=False)


COMMON7 = ["dl19", "dl20", "dl21", "dl22", "dl23", "trec-covid", "robust04"]  # collections every API model was run on
BEIR_DENSE = ["trec-covid", "dbpedia", "news", "robust04"]


def common7():
    """Calibration on the 7 collections shared by every system (incl. Clef, Qwen235, Qwen27-base): raw ECE/Brier and
    transfer ECE with the SAME leave-one-collection-out source pool (the other 6) for every system."""
    out = []
    for s in SYSTEMS:
        if not all(has_run(s, d) for d in COMMON7):
            continue
        data = {d: load_pairs(s, d) for d in COMMON7}
        for d in COMMON7:
            tgt = data[d]
            src = pd.concat([data[o] for o in COMMON7 if o != d])
            a, b = fit_platt(src.p.to_numpy(), src.y.to_numpy())
            m = calib_metrics(tgt.p.to_numpy(), tgt.y.to_numpy())
            out.append(dict(system=s, dataset=d, ECE=m["ECE"], Brier=m["Brier"],
                            ECE_transfer=100 * ece(apply_platt(tgt.p.to_numpy(), a, b), tgt.y.to_numpy())))
    pd.DataFrame(out).to_csv(RESULTS_DIR / "calib_common7.csv", index=False)


def transfer_nodl():
    """Leave-DL-out transfer for the BEIR dense targets: Platt fit only on the other BEIR dense collections."""
    out = []
    for s in SYSTEMS:
        avail = [d for d in BEIR_DENSE if has_run(s, d)]
        if len(avail) < 3:
            continue
        data = {d: load_pairs(s, d) for d in avail}
        for d in avail:
            src = pd.concat([data[o] for o in avail if o != d])
            a, b = fit_platt(src.p.to_numpy(), src.y.to_numpy())
            out.append(dict(system=s, dataset=d, ECE_raw=100 * ece(data[d].p.to_numpy(), data[d].y.to_numpy()),
                            ECE_transfer_noDL=100 * ece(apply_platt(data[d].p.to_numpy(), a, b), data[d].y.to_numpy())))
    pd.DataFrame(out).to_csv(RESULTS_DIR / "calib_transfer_nodl.csv", index=False)


def rag_hit1():
    """MuSiQue final: hit@1 (top-1 is any supporting paragraph) and last-hop@1 per ranker, with McNemar tests."""
    from scipy.stats import binomtest

    from choicerank.rag import load_split
    import json as _json
    items = {it.qid: it for it in load_split("musique", "final")}
    dec = {}
    for line in open(Path(__file__).resolve().parent.parent / "data" / "rag" / "musique_ans_v1.0_dev.jsonl", encoding="utf-8"):
        r = _json.loads(line)
        dec["musique:" + r["id"]] = [str(h["paragraph_support_idx"]) for h in r["question_decomposition"]]
    top = {}
    for rk, scol in (("jev", "score_score4"), ("bge-m3", "score")):
        t = pd.read_parquet(RUNS_DIR / "rag" / f"musique.final.{rk}.parquet")
        top[rk] = {q: g.sort_values([scol, "orig_rank"], ascending=[False, True]).pid.iloc[0] for q, g in t.groupby("qid")}
    qids = sorted(top["jev"])
    out = []
    for name, fn in [("hit@1", lambda q, p: any(x.pid == p and x.gold for x in items[q].passages)),
                     ("last_hop@1", lambda q, p: p == dec[q][-1])]:
        a = np.array([fn(q, top["jev"][q]) for q in qids]); b = np.array([fn(q, top["bge-m3"][q]) for q in qids])
        n01, n10 = int((~a & b).sum()), int((a & ~b).sum())  # discordant pairs
        p = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0  # exact McNemar
        out.append(dict(metric=name, jev=100 * a.mean(), bge=100 * b.mean(), jev_only=n10, bge_only=n01, p_mcnemar=p))
    pd.DataFrame(out).to_csv(RESULTS_DIR / "rag_hit1.csv", index=False)


if __name__ == "__main__":
    todo = sys.argv[1:] or ["readouts", "transfer", "bounds", "judged", "rag_selection", "rag_hops", "common7",
                            "transfer_nodl", "rag_hit1", "ci"]
    for name in todo:
        globals()[name]()
        print("done", name, flush=True)
