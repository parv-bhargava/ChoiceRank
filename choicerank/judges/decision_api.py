"""Typed decision models behind the shared Decisions schema: Jev (OpenRouter), Clef /
Clef-flash (Cloudflare Workers AI), and a local Jeff server.

Request:  {"model", "state", "questions": {id: {type, instructions, criteria}}}
Response: {"answers": {id: {choice/probabilities | noul | score/probabilities}}, "usage": {...}}
Schema refs (checked 2026-10-03):
  https://openrouter.ai/docs/guides/community/jev-tutorial.md
  https://developers.cloudflare.com/workers-ai/models/clef-flash/schema-output.json
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

from ..prompts import make_state, questions_for, readout
from .base import Judge, Judgment

# USD per 1M input tokens (output tokens are free / negligible for typed answers).
PRICES = {
    "jev": 0.042,          # OpenRouter typesafe/jev-1.13
    "clef": 0.24,          # Workers AI @cf/cloudflare/clef
    "clef-flash": 0.09,    # Workers AI @cf/cloudflare/clef-flash
    "jeff-0.8b": 0.0,
    "jeff-2b": 0.0,
}


class DecisionJudge(Judge):
    paid = True

    def __init__(self, backend: str, qtype: str = "choice3", workers: int = 8, max_chars: int = 4000):
        if backend not in PRICES:
            raise ValueError(f"unknown decision backend {backend}")
        self.backend, self.qtype, self.workers, self.max_chars = backend, qtype, workers, max_chars
        self.paid = PRICES[backend] > 0
        self.client = httpx.Client(timeout=60)
        if backend == "jev":
            self.url = "https://openrouter.ai/api/alpha/decisions"
            self.model = "typesafe/jev-1.13-20260917"  # pinned snapshot for reproducibility
            self.headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}
        elif backend.startswith("clef") and os.environ.get("OPENROUTER_API_KEY") and not os.environ.get("CF_API_TOKEN"):
            # Same model and deployment as Workers AI: OpenRouter's only provider for cloudflare/clef* is Cloudflare.
            self.url = "https://openrouter.ai/api/alpha/decisions"
            self.model = f"cloudflare/{backend}"
            self.headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}
        elif backend.startswith("clef"):
            acct = os.environ["CF_ACCOUNT_ID"]
            self.url = f"https://api.cloudflare.com/client/v4/accounts/{acct}/ai/run/@cf/cloudflare/{backend}"
            self.model = backend
            self.headers = {"Authorization": f"Bearer {os.environ['CF_API_TOKEN']}"}
        else:  # local Jeff server (github.com/firelex/jeff, `jeff-serve`); one checkpoint per server
            # The server processes one request at a time (529 when busy), so call it sequentially.
            self.url = os.environ.get("JEFF_URL", "http://localhost:8765").rstrip("/") + "/v1/systemone"
            self.model = "jeff-latest"
            self.headers = {}
            self.workers = 1

    @property
    def judge_id(self) -> str:
        return f"dm:{self.backend}:{self.qtype}"

    def _call(self, query: str, passage: str) -> Judgment:
        body = {
            "model": self.model,
            "state": make_state(query, passage[: self.max_chars]),
            "questions": questions_for(self.qtype),
        }
        r, last_exc = None, None
        for attempt in range(8):
            t0 = time.perf_counter()
            try:
                r = self.client.post(self.url, json=body, headers=self.headers)
            except httpx.TransportError as e:  # dropped connections / read timeouts under load
                last_exc = e
                time.sleep(min(2**attempt, 30))
                continue
            if r.status_code in (408, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 529):
                time.sleep(min(2**attempt, 30) * (0.05 if r.status_code == 529 else 1))
                continue
            r.raise_for_status()
            break
        else:
            if r is None:
                raise last_exc
            r.raise_for_status()
        latency = (time.perf_counter() - t0) * 1000
        data = r.json()
        data = data.get("result", data)  # Cloudflare wraps payloads in {"result": ...}
        answers, usage = data["answers"], data.get("usage", {})
        primary = "choice3" if self.qtype == "all" else self.qtype
        score, p_rel = readout(primary, answers[primary])
        tokens = int(usage.get("input_tokens", 0))
        cost = float(usage.get("cost", tokens * PRICES[self.backend] / 1e6))
        return Judgment(score, p_rel, raw=answers, latency_ms=latency, in_tokens=tokens, cost_usd=cost)

    def _judge(self, query, passages):
        with ThreadPoolExecutor(self.workers) as pool:
            return list(pool.map(lambda p: self._call(query, p), passages))
