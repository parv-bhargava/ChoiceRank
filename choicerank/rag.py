"""RQ3: threshold-based context selection vs fixed top-k for RAG.

Datasets (DATASETS.md):
  musique  MuSiQue-Ans dev: 20 paragraphs per question, `is_supporting` gold flags (2-4 per q).
  nq       NQ-open test with DPR top-20 retrieved passages, `has_answer` string-match flags.

Splits are seeded: `calib` (Platt fitting, 200 q), `pilot` (300 q, policy design), `main` (1,000 q
containing the pilot), and `final` (1,000 q disjoint from calib AND pilot). Policies revised after the
pilot must be evaluated on `final` only.

MuSiQue questions share decomposition sub-questions, so the ID-disjoint splits above share component
families. The family-disjoint splits `calib_fd` and `final_fd` (issue #3) assign whole families, i.e. connected
components of the graph linking questions that share any decomposition ID, to one split. Families containing an
original pilot question are excluded, so the policies locked after the pilot are evaluated on unseen families.
"""

from __future__ import annotations

import gzip
import json
import random
import re
import string
from collections import Counter
from dataclasses import dataclass

import numpy as np

from . import DATA_DIR
from .data import Candidate, Query

RAG_DIR = DATA_DIR / "rag"
SEED = 13
N_CALIB, N_PILOT, N_MAIN = 200, 300, 1000
FD_SPLITS = ("calib_fd", "final_fd")
NQ_DEPTH = 20
SOURCES = {  # fetched on first use
    "musique_ans_v1.0_dev.jsonl": "https://huggingface.co/datasets/dgslibisey/MuSiQue/resolve/main/musique_ans_v1.0_dev.jsonl",
    "dpr_nq_test.json.gz": "https://dl.fbaipublicfiles.com/dpr/data/retriever_results/single/nq-test.json.gz",
}


def _source(name: str):
    path = RAG_DIR / name
    if not path.exists():
        import httpx

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        with httpx.stream("GET", SOURCES[name], follow_redirects=True, timeout=300) as r, open(tmp, "wb") as f:
            r.raise_for_status()
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
        tmp.replace(path)
    return path


@dataclass
class Passage:
    pid: str
    title: str
    text: str
    gold: bool


@dataclass
class RagItem:
    qid: str
    question: str
    answers: list[str]
    passages: list[Passage]
    hops: int = 1
    components: tuple[str, ...] = ()  # MuSiQue decomposition sub-question IDs

    def as_query(self) -> Query:
        """Judge input; passage text uses the BEIR-style "Title: .. Content: .." format of the ranking runs."""
        return Query(self.qid, self.question,
                     [Candidate(p.pid, 0.0, f"Title: {p.title} Content: {p.text}") for p in self.passages])


def _load_musique() -> list[RagItem]:
    items = []
    for line in open(_source("musique_ans_v1.0_dev.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        items.append(RagItem(
            qid=f"musique:{r['id']}", question=r["question"],
            answers=[r["answer"], *r.get("answer_aliases", [])],
            passages=[Passage(str(p["idx"]), p["title"], p["paragraph_text"], bool(p["is_supporting"]))
                      for p in r["paragraphs"]],
            hops=int(r["id"][0]), components=tuple(str(h["id"]) for h in r["question_decomposition"])))
    return items


def musique_families(items: list[RagItem]) -> dict[str, str]:
    """qid -> family ID: connected components over shared decomposition IDs; a family's ID is its smallest qid."""
    parent = {it.qid: it.qid for it in items}

    def find(q):
        while parent[q] != q:
            parent[q] = parent[parent[q]]
            q = parent[q]
        return q

    seen = {}
    for it in sorted(items, key=lambda x: x.qid):
        for c in it.components:
            if c in seen:
                a, b = find(it.qid), find(seen[c])
                parent[max(a, b)] = min(a, b)
            else:
                seen[c] = it.qid
    return {q: find(q) for q in parent}


def _family_disjoint(items: list[RagItem], pilot: list[RagItem]) -> dict[str, list[RagItem]]:
    """Whole families not touching the original pilot, shuffled with SEED; `calib_fd` takes families until each
    hop count reaches its share of N_CALIB (in the proportions of this eligible pool, so calib_fd and final_fd have
    the same hop mix), skipping families that would overshoot it; `final_fd` gets every remaining family."""
    fam = musique_families(items)
    excluded = {fam[it.qid] for it in pilot}
    members: dict[str, list[RagItem]] = {}
    for it in sorted(items, key=lambda x: x.qid):
        if fam[it.qid] not in excluded:
            members.setdefault(fam[it.qid], []).append(it)
    order = sorted(members)
    random.Random(SEED).shuffle(order)
    pool = [it for f in order for it in members[f]]
    hops = Counter(it.hops for it in pool)
    target = {h: round(N_CALIB * n / len(pool)) for h, n in hops.items()}
    have, calib, final = Counter(), [], []
    for f in order:
        need = Counter(it.hops for it in members[f])
        if all(have[h] + n <= target.get(h, 0) for h, n in need.items()):
            calib += members[f]
            have += need
        else:
            final += members[f]
    return {"calib_fd": calib, "final_fd": final}


def _load_nq() -> list[RagItem]:
    data = json.load(gzip.open(_source("dpr_nq_test.json.gz"), "rt", encoding="utf-8"))
    items = []
    for i, r in enumerate(data):
        items.append(RagItem(
            qid=f"nq:{i}", question=r["question"], answers=list(r["answers"]),
            passages=[Passage(str(c["id"]), c["title"], c["text"], bool(c["has_answer"])) for c in r["ctxs"][:NQ_DEPTH]]))
    return items


def load_split(dataset: str, split: str) -> list[RagItem]:
    """Seeded, disjoint splits. MuSiQue is stratified by hop count (2/3/4-hop in dev proportions)."""
    if split in FD_SPLITS:
        if dataset != "musique":
            raise ValueError("family-disjoint splits are defined for MuSiQue only")
        return _family_disjoint(_load_musique(), load_split("musique", "pilot"))[split]
    items = _load_musique() if dataset == "musique" else _load_nq()
    rng = random.Random(SEED)
    if dataset == "musique":
        by_hop: dict[int, list[RagItem]] = {}
        for it in items:
            by_hop.setdefault(it.hops, []).append(it)
        order = []
        for h in sorted(by_hop):
            rng.shuffle(by_hop[h])
        # interleave hop groups proportionally so every prefix is stratified
        total = len(items)
        quota = {h: len(v) / total for h, v in by_hop.items()}
        taken = {h: 0 for h in by_hop}
        while len(order) < total:
            h = min(by_hop, key=lambda k: (taken[k] + 1) / quota[k] if taken[k] < len(by_hop[k]) else float("inf"))
            order.append(by_hop[h][taken[h]])
            taken[h] += 1
    else:
        order = items[:]
        rng.shuffle(order)
    calib, rest = order[:N_CALIB], order[N_CALIB:]
    return {"calib": calib, "pilot": rest[:N_PILOT], "main": rest[:N_MAIN],
            "final": rest[N_PILOT:N_PILOT + N_MAIN]}[split]


# ---------------------------------------------------------------- selection policies

def topk(ranked: list[str], k: int) -> list[str]:
    return ranked[:k]


def threshold(ranked: list[str], prob: dict[str, float], t: float) -> list[str]:
    """Keep every passage with P(rel) > t, in ranker order (may be empty)."""
    return [p for p in ranked if prob[p] > t]


def floor_threshold(ranked: list[str], prob: dict[str, float], t: float, kmin: int) -> list[str]:
    """Post-pilot policy: always keep the top-`kmin` passages, then add any later passage with P(rel) > t."""
    return ranked[:kmin] + [p for p in ranked[kmin:] if prob[p] > t]


def adaptive_k(ranked: list[str], score: dict[str, float], buffer: int = 0) -> list[str]:
    """Adaptive-k (Taguchi et al. 2025): cut at the largest gap in the sorted score list (+ optional buffer)."""
    s = np.array([score[p] for p in ranked])
    if len(s) < 2:
        return ranked
    k = int(np.argmax(s[:-1] - s[1:])) + 1
    return ranked[:min(len(ranked), k + buffer)]


# ---------------------------------------------------------------- answer metrics (SQuAD)

def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, golds: list[str]) -> float:
    return float(max(normalize(pred) == normalize(g) for g in golds))


def f1(pred: str, golds: list[str]) -> float:
    def one(g):
        p, t = normalize(pred).split(), normalize(g).split()
        common = Counter(p) & Counter(t)
        same = sum(common.values())
        if same == 0:
            return 0.0
        prec, rec = same / len(p), same / len(t)
        return 2 * prec * rec / (prec + rec)
    return max(one(g) for g in golds)


def contains(pred: str, golds: list[str]) -> float:
    """'Accuracy' / SubEM: a normalized gold answer appears in the normalized prediction."""
    n = normalize(pred)
    return float(any(normalize(g) and normalize(g) in n for g in golds))
