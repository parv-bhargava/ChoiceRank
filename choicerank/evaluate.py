"""Ranking metrics (ir_measures) + calibration metrics for P(relevant)."""

from __future__ import annotations

import ir_measures
import numpy as np
from ir_measures import MRR, R, nDCG

RANK_MEASURES = [nDCG @ 10, MRR @ 10, R @ 20]


def rank_metrics(run: dict[str, dict[str, float]], qrels: dict[str, dict[str, int]], rel_threshold: int) -> dict:
    run = {q: d for q, d in run.items() if q in qrels}  # judged queries only
    measures = [nDCG @ 10, MRR(rel=rel_threshold) @ 10, R(rel=rel_threshold) @ 20]
    agg = ir_measures.calc_aggregate(measures, qrels, run)
    names = dict(zip(measures, ["nDCG@10", "MRR@10", "R@20"]))
    # fixed column order: calc_aggregate's dict order varies between runs, which reordered ranking.csv (issue #26)
    return {names[m]: 100 * agg[m] for m in reversed(measures)}


def per_query(run, qrels, measure=nDCG @ 10) -> dict[str, float]:
    run = {q: d for q, d in run.items() if q in qrels}
    return {m.query_id: m.value for m in ir_measures.iter_calc([measure], qrels, run)}


def labels_for(run_p: dict[str, dict[str, float]], qrels, rel_threshold: int, dense_pool: bool = True):
    """Flatten (query, passage) pairs -> (p, y).

    dense_pool=True  (TREC DL, TREC-COVID): judged pairs only; qrels contain real negatives.
    dense_pool=False (most BEIR sets): qrels list positives only, so unjudged = not relevant
    (standard BEIR assumption; slightly under-counts relevance -> report as a caveat)."""
    p, y = [], []
    for qid, docs in run_p.items():
        if qid not in qrels:
            continue
        judged = qrels[qid]
        for docid, prob in docs.items():
            if docid in judged:
                p.append(prob)
                y.append(int(judged[docid] >= rel_threshold))
            elif not dense_pool:
                p.append(prob)
                y.append(0)
    return np.asarray(p, float), np.asarray(y, int)


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 15, adaptive: bool = False) -> float:
    """Expected calibration error (equal-width bins, or equal-mass if adaptive)."""
    if adaptive:
        edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, n_bins - 1)
    err = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            err += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(err)


def reliability(p, y, n_bins: int = 10):
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            rows.append((float(p[m].mean()), float(y[m].mean()), int(m.sum())))
    return rows


def calib_metrics(p, y) -> dict:
    return {
        "ECE": 100 * ece(p, y),
        "ECE_adapt": 100 * ece(p, y, adaptive=True),
        "Brier": float(np.mean((p - y) ** 2)),
        "base_rate": float(y.mean()) if len(y) else float("nan"),
        "mean_p": float(p.mean()) if len(p) else float("nan"),
        "n_pairs": int(len(y)),
    }


def fit_temperature(p, y, grid=np.exp(np.linspace(-3, 3, 241))) -> float:
    """Single temperature on the logit of P(rel), fit by NLL grid search."""
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    best, best_t = np.inf, 1.0
    for t in grid:
        q = np.clip(1 / (1 + np.exp(-z / t)), 1e-9, 1 - 1e-9)
        nll = -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
        if nll < best:
            best, best_t = nll, t
    return float(best_t)


def apply_temperature(p, t: float):
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    return 1 / (1 + np.exp(-z / t))


def _logit(p):
    return np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))


def fit_platt(p, y) -> tuple[float, float]:
    """Platt scaling on the logit of P(rel): q = sigmoid(a * logit(p) + b), fit by NLL."""
    from scipy.optimize import minimize

    z = _logit(p)

    def nll(w):
        q = np.clip(1 / (1 + np.exp(-(w[0] * z + w[1]))), 1e-9, 1 - 1e-9)
        return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))

    a, b = minimize(nll, x0=[1.0, 0.0], method="L-BFGS-B").x
    return float(a), float(b)


def apply_platt(p, a: float, b: float):
    return 1 / (1 + np.exp(-(a * _logit(p) + b)))
