"""Laya (convaiinnovations/laya), local CPU. Same typed-question schema as Jev/Clef.

English checkpoint = ModernBERT-large, 512-token context (long passages get truncated);
`variant="multilingual"` reads up to 8,192 tokens via max_len.
"""

from __future__ import annotations

import time

from ..prompts import make_state, questions_for, readout
from .base import Judge, Judgment


class LayaJudge(Judge):
    def __init__(self, qtype: str = "choice3", variant: str = "english", max_len: int | None = None):
        from laya import Router

        self.router = Router()
        self.qtype, self.variant, self.max_len = qtype, variant, max_len

    @property
    def judge_id(self) -> str:
        return f"dm:laya-{self.variant}{'' if self.max_len is None else f'-{self.max_len}'}:{self.qtype}"

    def _judge(self, query, passages):
        out = []
        kw = {"model": self.variant}
        if self.max_len:
            kw["max_len"] = self.max_len
        primary = "choice3" if self.qtype == "all" else self.qtype
        for p in passages:
            t0 = time.perf_counter()
            res = self.router.predict(make_state(query, p), questions_for(self.qtype), **kw)
            latency = (time.perf_counter() - t0) * 1000
            score, p_rel = readout(primary, res["answers"][primary])
            out.append(Judgment(score, p_rel, raw=res["answers"], latency_ms=latency))
        return out
