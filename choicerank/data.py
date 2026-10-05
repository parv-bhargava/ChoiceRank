"""Candidate lists and qrels.

First stage = castorini/rank_llm_data prebuilt Pyserini BM25 top-100 runs (same lineage
as RankGPT / RankLLM), which ship passage text, so no Java / Pyserini / corpus download.
Qrels = the official Pyserini qrels files (github.com/castorini/eval), i.e. exactly what
RankGPT / RankLLM evaluate with. See DATASETS.md for the selection rationale.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass

import httpx
from tqdm import tqdm

from . import DATA_DIR

RUN_URL = (
    "https://huggingface.co/datasets/castorini/rank_llm_data/resolve/main/"
    "retrieve_results/BM25/retrieve_results_{key}_top100.jsonl"
)
QRELS_URL = "https://raw.githubusercontent.com/castorini/eval/master/qrels/{file}"


@dataclass(frozen=True)
class DatasetSpec:
    run_key: str          # castorini run file key
    qrels_file: str       # Pyserini qrels file in castorini/eval/qrels
    rel_threshold: int    # min grade counted relevant (binary metrics, calibration)
    dense_pool: bool = False  # qrels include judged negatives (TREC pools); else positives-only
    # P(relevant) readout matched to the label semantics of the lowest relevant grade. NOTE: introduced after observing
    # TREC-COVID results (disclosed in the paper; both readouts reported in the appendix):
    # "lenient" if that grade means *partially* relevant (C3: p_rel + p_partial), else "strict".
    readout: str = "strict"
    max_queries: int | None = None  # default subsample size (None = all)
    qrels_url: str | None = None  # full URL when the qrels are not in castorini/eval
    build: str | None = None  # name of a local builder for collections without a published BM25 run


B = "qrels.beir-v1.0.0-{}.test.txt"
DATASETS: dict[str, DatasetSpec] = {
    # TREC DL passage: 0-3; grade 1 = "related" (non-relevant in the official binarization)
    "dl19": DatasetSpec("dl19", "qrels.dl19-passage.txt", 2, dense_pool=True),
    "dl20": DatasetSpec("dl20", "qrels.dl20-passage.txt", 2, dense_pool=True),
    "dl21": DatasetSpec("dl21", "qrels.dl21-passage.txt", 2, dense_pool=True),
    "dl22": DatasetSpec("dl22", "qrels.dl22-passage.txt", 2, dense_pool=True),
    "dl23": DatasetSpec("dl23", "qrels.dl23-passage.txt", 2, dense_pool=True),
    # BEIR-8 (RankGPT protocol)
    "trec-covid": DatasetSpec("covid", B.format("trec-covid"), 1, dense_pool=True, readout="lenient"),  # 1 = partially
    "nfcorpus": DatasetSpec("nfc", B.format("nfcorpus"), 1),
    "touche": DatasetSpec("touche", B.format("webis-touche2020"), 1),  # Pyserini file drops zero labels
    "dbpedia": DatasetSpec("dbpedia", B.format("dbpedia-entity"), 1, dense_pool=True, readout="lenient"),  # 1 = partially
    "scifact": DatasetSpec("scifact", B.format("scifact"), 1),
    "signal": DatasetSpec("signal", B.format("signal1m"), 1),
    "news": DatasetSpec("news", B.format("trec-news"), 1, dense_pool=True, readout="lenient"),  # 1 = some useful info
    "robust04": DatasetSpec("robust04", B.format("robust04"), 1, dense_pool=True),  # 1 = relevant, 2 = highly
    # Post-cutoff contamination checks (review feedback). Both predate the Sept-2026 decision models.
    # NovelEval-2306 (Sun et al. 2023): 21 queries x 20 Google-retrieved passages; 0/1/2, 1 = partially relevant.
    "noveleval": DatasetSpec("noveleval", "noveleval", 1, dense_pool=True, readout="lenient",
                             qrels_url="https://raw.githubusercontent.com/sunnweiwei/RankGPT/main/NovelEval/qrels.txt"),
    # FutureQueryEval (Abdallah et al., EMNLP 2025): 148 queries about events after April 2025, 2,787-doc corpus,
    # graded 0-2 (meaning of grade 1 not documented -> strict readout). BM25 top-100 built locally with bm25s.
    "futurequery": DatasetSpec("futurequery", "futurequery", 1, dense_pool=True, build="futurequery",
                               qrels_url="https://huggingface.co/datasets/abdoelsayed/FutureQueryEval/resolve/main/qrels.txt"),
}
FQE_URL = "https://huggingface.co/datasets/abdoelsayed/FutureQueryEval/resolve/main/{}"

DL = ["dl19", "dl20", "dl21", "dl22", "dl23"]
BEIR8 = ["trec-covid", "nfcorpus", "touche", "dbpedia", "scifact", "signal", "news", "robust04"]
ALL = DL + BEIR8
CALIB = [d for d in ALL if DATASETS[d].dense_pool]  # collections with judged negatives
CORE = ["dl19", "dl20", "trec-covid", "nfcorpus", "scifact"]  # original 2-week core (kept for reference)
NOVEL = ["noveleval", "futurequery"]  # post-cutoff checks, reported separately (not in ALL)
GROUPS = {"core": CORE, "all": ALL, "dl": DL, "beir8": BEIR8, "calib": CALIB, "novel": NOVEL}


@dataclass
class Candidate:
    docid: str
    bm25: float
    text: str


@dataclass
class Query:
    qid: str
    text: str
    cands: list[Candidate]


def run_path(name: str):
    return DATA_DIR / "bm25" / f"{DATASETS[name].run_key}_top100.jsonl"


def download_run(name: str) -> None:
    path = run_path(name)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if DATASETS[name].build == "futurequery":
        return _build_futurequery(path)
    tmp = path.with_suffix(".part")
    url = RUN_URL.format(key=DATASETS[name].run_key)
    with httpx.stream("GET", url, follow_redirects=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) or None
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=name) as bar:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.replace(path)


def doc_text(doc: dict) -> str:
    """castorini files use {"contents"}, MS MARCO v2 {"passage"} (DL21-23), or BEIR-style {"title", "text"}."""
    if "contents" in doc:
        return doc["contents"]
    if "passage" in doc:
        return doc["passage"]
    title, text = doc.get("title", "").strip(), doc.get("text", "").strip()
    return f"Title: {title} Content: {text}" if title else text


def load_queries(name: str, topk: int = 100, limit: int | None = None, seed: int = 13) -> list[Query]:
    """Load BM25 candidates, truncated to top-k; optionally a seeded query subsample.

    `limit=None` falls back to the dataset's default `max_queries`; `limit=0` means all.
    """
    download_run(name)
    queries = []
    with open(run_path(name), encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            cands = [
                Candidate(str(c["docid"]), float(c["score"]), doc_text(c["doc"]))
                for c in row["candidates"][:topk]
            ]
            queries.append(Query(str(row["query"]["qid"]), row["query"]["text"], cands))
    n = DATASETS[name].max_queries if limit is None else (limit or None)
    if n and n < len(queries):
        queries = sorted(random.Random(seed).sample(queries, n), key=lambda q: q.qid)
    return queries


def load_qrels(name: str) -> dict[str, dict[str, int]]:
    """qid -> {docid: grade}, from the Pyserini qrels file (TREC format), cached under data/qrels/."""
    spec = DATASETS[name]
    file = spec.qrels_file
    path = DATA_DIR / "qrels" / f"pyserini_{file}"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        r = httpx.get(spec.qrels_url or QRELS_URL.format(file=file), follow_redirects=True, timeout=120)
        r.raise_for_status()
        path.write_text(r.text, encoding="utf-8")
    qrels: dict[str, dict[str, int]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 4:
                qrels.setdefault(parts[0], {})[parts[2]] = int(parts[3])
    return qrels


_TOK = None


def truncate_passages(queries: list[Query], n_tokens: int) -> list[Query]:
    """Equal-context-budget ablation: cut every passage to its first `n_tokens` tokens (bge-reranker-v2-m3 /
    XLM-R tokenizer) *before* any model sees it, so all systems read the same text."""
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer

        _TOK = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
    for q in queries:
        ids = _TOK([c.text for c in q.cands], add_special_tokens=False, truncation=True, max_length=n_tokens)["input_ids"]
        for c, t in zip(q.cands, ids):
            c.text = _TOK.decode(t, skip_special_tokens=True)
    return queries


def bm25_run(queries: list[Query]) -> dict[str, dict[str, float]]:
    return {q.qid: {c.docid: c.bm25 for c in q.cands} for q in queries}


def _build_futurequery(path) -> None:
    """BM25 top-100 over the FutureQueryEval corpus (bm25s, Lucene-style defaults), written in the castorini format."""
    import csv
    import io

    import bm25s

    get = lambda f: httpx.get(FQE_URL.format(f), follow_redirects=True, timeout=120).text
    corpus = list(csv.reader(io.StringIO(get("corpus.tsv")), delimiter="\t"))
    queries = [r for r in csv.reader(io.StringIO(get("queries.tsv")), delimiter="\t") if r and r[0].startswith("query_")]
    ids, texts = [r[0] for r in corpus], [r[1] for r in corpus]
    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize(texts, stopwords="en"))
    docs, scores = retriever.retrieve(bm25s.tokenize([q[1] for q in queries], stopwords="en"), k=min(100, len(ids)))
    with open(path, "w", encoding="utf-8") as f:
        for q, d, sc in zip(queries, docs, scores):
            cands = [{"docid": ids[i], "score": float(v), "doc": {"contents": texts[i]}} for i, v in zip(d, sc)]
            f.write(json.dumps({"query": {"qid": q[0], "text": q[1]}, "candidates": cands}) + "\n")
