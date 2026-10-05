"""Pre-flight token / USD estimate for paid judges (chars/4 heuristic + schema overhead).

The heuristic is calibrated against the first real `usage.input_tokens` we observe; until
then treat estimates as +/-30%.
"""

from __future__ import annotations

import json

from .data import Query
from .judges.decision_api import PRICES
from .prompts import questions_for

CHARS_PER_TOKEN = 4.0
# Model/system framing + JSON keys. Calibrated 2026-10-03 on one Jev call (qtype=all):
# 572 actual vs 321 estimated with 40, so use 290. Re-check after the first 1-query run.
FIXED_OVERHEAD_TOKENS = 290


def pair_tokens(query: str, passage: str, qtype: str, max_chars: int = 4000, backend: str = "jev") -> int:
    if backend.endswith(("-yesno", "-graded")):  # short yes/no or 0-4 rating prompt, passage cut to 448 tokens (~1,800 chars)
        return int((len(query) + min(len(passage), 1800)) / CHARS_PER_TOKEN) + 60
    q_chars = len(json.dumps(questions_for(qtype)))
    return int((len(query) + min(len(passage), max_chars) + q_chars) / CHARS_PER_TOKEN) + FIXED_OVERHEAD_TOKENS


def estimate(queries: list[Query], backend: str, qtype: str) -> dict:
    tokens = sum(pair_tokens(q.text, c.text, qtype, backend=backend) for q in queries for c in q.cands)
    pairs = sum(len(q.cands) for q in queries)
    price = PRICES.get(backend)
    if price is None and backend.endswith(("-yesno", "-graded")):
        from .judges.llm_yesno import MODELS

        price = MODELS[backend][2]
    usd = tokens * (price or 0.0) / 1e6
    return {"queries": len(queries), "pairs": pairs, "tokens_M": tokens / 1e6, "usd": usd}
