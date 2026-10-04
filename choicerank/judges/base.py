"""Common pointwise-judge interface with a persistent SQLite cache.

Every (judge_id, qid, docid) result is cached, so re-runs and crashes never re-pay
for API calls. Results store rank score, P(relevant), the raw typed answer, latency,
input tokens and USD cost.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field

from .. import RUNS_DIR
from ..data import Query


@dataclass
class Judgment:
    score: float                 # value used for ranking
    p_rel: float                 # calibrated-or-not probability of relevance
    raw: dict = field(default_factory=dict)
    latency_ms: float = 0.0      # per pair (batch latency / batch size for local models)
    in_tokens: int = 0
    cost_usd: float = 0.0


class JudgeCache:
    def __init__(self, path=RUNS_DIR / "cache.sqlite"):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS j (judge TEXT, qid TEXT, docid TEXT, payload TEXT,"
            " PRIMARY KEY (judge, qid, docid))"
        )

    def get_many(self, judge: str, qid: str) -> dict[str, Judgment]:
        with self.lock:
            rows = self.db.execute("SELECT docid, payload FROM j WHERE judge=? AND qid=?", (judge, qid))
            return {d: Judgment(**json.loads(p)) for d, p in rows}

    def put_many(self, judge: str, qid: str, items: dict[str, Judgment]) -> None:
        with self.lock:
            self.db.executemany(
                "INSERT OR REPLACE INTO j VALUES (?,?,?,?)",
                [(judge, qid, d, json.dumps(asdict(x))) for d, x in items.items()],
            )
            self.db.commit()


class Judge(ABC):
    """Subclasses implement `_judge(query, passages) -> list[Judgment]`."""

    paid: bool = False  # True => run_rerank requires --i-approve-spend
    id_suffix: str = ""  # e.g. "@p448": passages pre-truncated to 448 tokens (separate cache rows)

    @property
    def cache_id(self) -> str:
        return self.judge_id + self.id_suffix

    @property
    @abstractmethod
    def judge_id(self) -> str: ...

    @abstractmethod
    def _judge(self, query: str, passages: list[str]) -> list[Judgment]: ...

    def judge_query(self, q: Query, cache: JudgeCache) -> tuple[dict[str, Judgment], set[str]]:
        """Returns ({docid: Judgment}, docids newly scored this call, i.e. not from cache)."""
        have = cache.get_many(self.cache_id, q.qid)
        todo = [c for c in q.cands if c.docid not in have]
        if todo:
            new = self._judge(q.text, [c.text for c in todo])
            fresh = {c.docid: j for c, j in zip(todo, new)}
            cache.put_many(self.cache_id, q.qid, fresh)
            have.update(fresh)
        return {c.docid: have[c.docid] for c in q.cands}, {c.docid for c in todo}
