"""Evaluate every run in runs/ -> results/ranking.csv and results/calibration.csv.

Calibration: raw P(rel) on judged pairs, plus temperature scaling (TS, 1 param) and Platt
scaling (2 params), each fit with 2-fold cross-fitting over queries (fit on half the
queries, apply to the other half).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from choicerank import RESULTS_DIR, RUNS_DIR
from choicerank.data import DATASETS, load_qrels
from choicerank.evaluate import (apply_platt, apply_temperature, calib_metrics, fit_platt, fit_temperature,
                                 labels_for, rank_metrics)


WARMUP_PAIRS = 10  # first pairs of each run include model loading / CUDA init


def read_trec(path):
    run = {}
    for line in open(path, encoding="utf-8"):
        qid, _, docid, _, score, _ = line.split()
        run.setdefault(qid, {})[docid] = float(score)
    return run


def crossfit(df, qrels, thr, dense, method="ts", seed=0):
    qids = np.array(sorted(df.qid.unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(qids)
    folds = [set(qids[::2]), set(qids[1::2])]
    ps, ys, ts = [], [], []
    for k in (0, 1):
        fit = df[df.qid.isin(folds[k])]
        app = df[df.qid.isin(folds[1 - k])]
        pf, yf = labels_for(to_nested(fit), qrels, thr, dense)
        pa, ya = labels_for(to_nested(app), qrels, thr, dense)
        if len(yf) == 0 or len(ya) == 0:
            continue
        if method == "ts":
            t = fit_temperature(pf, yf)
            ps.append(apply_temperature(pa, t))
        else:
            a, b = fit_platt(pf, yf)
            t = a
            ps.append(apply_platt(pa, a, b))
        ts.append(t)
        ys.append(ya)
    if not ps:
        return None, None
    return calib_metrics(np.concatenate(ps), np.concatenate(ys)), float(np.mean(ts))


def to_nested(df):
    out = {}
    for qid, docid, p in zip(df.qid, df.docid, df.p_rel):
        out.setdefault(str(qid), {})[str(docid)] = float(p)
    return out


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    rank_rows, cal_rows = [], []
    for path in sorted(RUNS_DIR.glob("*.trec")):
        ds, topk, system = path.stem.split(".", 2)
        spec = DATASETS[ds]
        qrels = load_qrels(ds)
        rank_rows.append({"dataset": ds, "depth": topk, "system": system,
                          **rank_metrics(read_trec(path), qrels, spec.rel_threshold)})
        pq = path.with_suffix(".parquet")
        if pq.exists():
            df = pd.read_parquet(pq)
            df["qid"], df["docid"] = df.qid.astype(str), df.docid.astype(str)
            # Match the probability readout to the label semantics (DatasetSpec.readout, fixed a priori):
            # where the lowest relevant grade means *partially* relevant, use P(relevant)+P(partially)
            # (prompts.p_lenient); otherwise the strict readout.
            readout = "strict"
            if spec.readout == "lenient" and "p_rel_lenient" in df:
                df["p_rel"], readout = df["p_rel_lenient"], "lenient"
            p, y = labels_for(to_nested(df), qrels, spec.rel_threshold, spec.dense_pool)
            lat = df.latency_ms.iloc[WARMUP_PAIRS:]  # rows are in processing order; drop model warm-up
            row = {"dataset": ds, "system": system, "labels": "judged" if spec.dense_pool else "unjudged=0", "readout": readout,
                   **calib_metrics(p, y),
                   "lat_p50_ms": lat.quantile(0.5), "lat_p95_ms": lat.quantile(0.95),
                   "usd_per_1k_q": 1000 * df.cost_usd.sum() / max(df.qid.nunique(), 1)}
            ts, t = crossfit(df, qrels, spec.rel_threshold, spec.dense_pool, "ts")
            if ts:
                row.update({"T": t, "ECE_TS": ts["ECE"], "Brier_TS": ts["Brier"]})
            pl, _ = crossfit(df, qrels, spec.rel_threshold, spec.dense_pool, "platt")
            if pl:
                row.update({"ECE_Platt": pl["ECE"], "Brier_Platt": pl["Brier"]})
            cal_rows.append(row)
    rank = pd.DataFrame(rank_rows)
    rank.to_csv(RESULTS_DIR / "ranking.csv", index=False)
    print(rank.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    if cal_rows:
        cal = pd.DataFrame(cal_rows)
        cal.to_csv(RESULTS_DIR / "calibration.csv", index=False)
        print()
        print(cal.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
