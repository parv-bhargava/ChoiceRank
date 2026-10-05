"""Label-probability pointwise LLM baseline (paid, OpenRouter): P(yes) from next-token log-probabilities.

The "relevance generation" readout of a general instruction-tuned LLM (cf. Zhuang et al. 2023; Liang et al.
HELM): one forward step, P(yes) = p(Yes) / (p(Yes) + p(No)) over the top-5 next-token log-probabilities (provider maximum).
This is the non-typed counterpart of a decision model's Noul answer. A single provider is pinned (no fallbacks)
to two fp8 deployments (GMICloud, Parasail) for throughput; the serving provider of every call is recorded.
Passages are cut to the cross-encoders' 448-token budget.
"""

from __future__ import annotations

import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

from .base import Judge, Judgment

MODELS = {  # alias -> (OpenRouter id, allowed providers (all fp8, logprobs-capable), $/M input (max), $/M output)
    "qwen235-yesno": ("qwen/qwen3-235b-a22b-2507", ["GMICloud", "Parasail"], 0.14, 0.0),
    # Clef's own base model, served by Cloudflare (Clef's host): same-base P(yes) readout, reasoning disabled
    "qwen27base-yesno": ("qwen/qwen3.8-27b", ["Cloudflare"], 0.45, 0.0),
    "qwen235-graded": ("qwen/qwen3-235b-a22b-2507", ["GMICloud", "Parasail"], 0.14, 0.0),
}
NO_REASONING = {"qwen27base-yesno"}  # hybrid-thinking models: answer directly
SYSTEM = "You are a search relevance assessor."
USER = ("Query: {q}\nPassage: {p}\n\n"
        "Does the passage answer the query? Answer with Yes or No only.")
# Graded-label readout (Zhuang et al., 2024): 0-4 rating, rank by the expected rating over the digit tokens
GRADED_USER = ("Query: {q}\nPassage: {p}\n\n"
               "From a scale of 0 to 4, judge the relevance between the query and the passage. "
               "Answer with a single digit only.")
YES, NO = {"yes", "y"}, {"no", "n"}


_PACE_LOCK = threading.Lock()
_NEXT_SLOT = [0.0]


def _pace(rpm: float) -> None:
    """Global request pacing across threads (e.g. CHOICERANK_RPM=18 for a 20-RPM account limit)."""
    if rpm <= 0:
        return
    with _PACE_LOCK:
        now = time.monotonic()
        wait = max(0.0, _NEXT_SLOT[0] - now)
        _NEXT_SLOT[0] = max(now, _NEXT_SLOT[0]) + 60.0 / rpm
    if wait:
        time.sleep(wait)


class LLMYesNoJudge(Judge):
    paid = True

    def __init__(self, alias: str, passage_tokens: int = 448, workers: int = 24):
        self.alias = alias
        self.model, self.providers, self.pin, self.pout = MODELS[alias]
        self.passage_tokens = passage_tokens
        self.workers = int(os.environ.get("CHOICERANK_WORKERS", workers))  # some providers rate-limit hard (429)
        self.client = httpx.Client(timeout=60, limits=httpx.Limits(max_connections=64, max_keepalive_connections=32))
        self.key = os.environ["OPENROUTER_API_KEY"]
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")

    @property
    def judge_id(self) -> str:
        return f"llm:{self.alias}"

    def _call(self, query: str, passage: str) -> Judgment:
        body = {"model": self.model, "temperature": 0, "max_tokens": 1, "logprobs": True, "top_logprobs": 5,
                "usage": {"include": True},
                "provider": {"only": self.providers, "require_parameters": True},
                **({"reasoning": {"enabled": False}} if self.alias in NO_REASONING else {}),
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": (GRADED_USER if self.alias.endswith("-graded") else USER).format(q=query, p=passage)}]}
        rpm = float(os.environ.get("CHOICERANK_RPM", "0"))
        last_err = None
        for attempt in range(8):
            _pace(rpm)
            t0 = time.perf_counter()
            try:
                r = self.client.post("https://openrouter.ai/api/v1/chat/completions", json=body,
                                     headers={"Authorization": f"Bearer {self.key}"})
            except httpx.TransportError:  # transient socket/connection errors (seen on Windows under load)
                time.sleep(min(2 ** attempt, 30))
                continue
            if r.status_code in (408, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524):
                time.sleep(min(2 ** attempt, 60) + (30 if r.status_code == 429 and rpm else 0))
                continue
            r.raise_for_status()
            d = r.json()
            if "choices" not in d:  # provider error returned with HTTP 200 (body holds {"error": ...})
                last_err = d.get("error", d)
                time.sleep(min(2 ** attempt, 60))
                continue
            break
        else:
            r.raise_for_status()
            raise RuntimeError(f"no choices after retries: {last_err}")
        latency = (time.perf_counter() - t0) * 1000
        top = (d["choices"][0].get("logprobs") or {}).get("content") or []
        cands = top[0]["top_logprobs"] if top else []
        if self.alias.endswith("-graded"):  # expected rating over the digits present in the top-5 tokens
            pk = {k: sum(math.exp(c["logprob"]) for c in cands if c["token"].strip() == str(k)) for k in range(5)}
            z = sum(pk.values())
            e = sum(k * v for k, v in pk.items()) / z / 4 if z > 0 else 0.5
            p2 = (pk[2] + pk[3] + pk[4]) / z if z > 0 else 0.5
            u = d.get("usage", {})
            return Judgment(score=e, p_rel=p2, latency_ms=latency, in_tokens=int(u.get("prompt_tokens", 0)),
                            cost_usd=float(u.get("cost", 0.0)),
                            raw={"p_digits": pk, "provider": d.get("provider"), "answer": d["choices"][0]["message"]["content"]})
        py = sum(math.exp(c["logprob"]) for c in cands if c["token"].strip().lower() in YES)
        pn = sum(math.exp(c["logprob"]) for c in cands if c["token"].strip().lower() in NO)
        p = py / (py + pn) if py + pn > 0 else 0.5
        u = d.get("usage", {})
        return Judgment(score=p, p_rel=p, latency_ms=latency, in_tokens=int(u.get("prompt_tokens", 0)),
                        cost_usd=float(u.get("cost", 0.0)),
                        raw={"p_yes": py, "p_no": pn, "provider": d.get("provider"),
                             "answer": d["choices"][0]["message"]["content"], "found": bool(py + pn)})

    def _judge(self, query, passages):
        ids = self.tok(passages, add_special_tokens=False, truncation=True, max_length=self.passage_tokens)["input_ids"]
        cut = [self.tok.decode(x, skip_special_tokens=True) for x in ids]
        with ThreadPoolExecutor(self.workers) as pool:
            return list(pool.map(lambda p: self._call(query, p), cut))
