"""Cross-encoder baselines (MiniLM-L6 on CPU, bge-reranker-v2-m3 on a T4)."""

from __future__ import annotations

import math
import time

import torch

from .base import Judge, Judgment

ALIASES = {
    "minilm": "cross-encoder/ms-marco-MiniLM-L6-v2",
    "bge-m3": "BAAI/bge-reranker-v2-m3",
}


class CrossEncoderJudge(Judge):
    def __init__(self, name: str = "minilm", max_length: int = 512, batch_size: int = 32, label: str | None = None):
        from sentence_transformers import CrossEncoder

        self.name = label or name  # label distinguishes e.g. bge-m3 at 2048 tokens ("bge-m3-2048")
        kw = {"model_kwargs": {"torch_dtype": torch.float16}} if torch.cuda.is_available() else {}
        self.model = CrossEncoder(ALIASES.get(name, name), max_length=max_length, **kw)
        if max_length > 512:
            batch_size = max(1, batch_size * 512 // max_length)  # bound GPU memory for long inputs
        self.batch_size = batch_size

    @property
    def judge_id(self) -> str:
        return f"ce:{self.name}"

    def _judge(self, query, passages):
        t0 = time.perf_counter()
        # Raw logits; P(rel) = sigmoid(logit). Uncalibrated by design (that is the point).
        logits = self.model.predict(
            [(query, p) for p in passages],
            batch_size=self.batch_size,
            activation_fn=torch.nn.Identity(),
            show_progress_bar=False,
        )
        per_pair = (time.perf_counter() - t0) * 1000 / max(len(passages), 1)
        return [
            Judgment(score=float(z), p_rel=1 / (1 + math.exp(-float(z))), latency_ms=per_pair)
            for z in logits
        ]


class Qwen3RerankerJudge(Judge):
    """Qwen3-Reranker (label-probability pointwise LLM reranker): P(yes) from the yes/no logits.

    Uses the sequence-classification conversion (tomaarsen/Qwen3-Reranker-*-seq-cls), whose single
    logit is log P(yes) - log P(no), so sigmoid(logit) = P(yes). The official chat template is applied;
    the passage is pre-truncated to `passage_tokens` (default 448, the cross-encoders' budget) so that
    truncation never cuts the template suffix.
    """

    PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the '
              'Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"

    def __init__(self, size: str = "0.6B", passage_tokens: int = 448, batch_size: int = 16):
        from sentence_transformers import CrossEncoder

        self.size, self.passage_tokens, self.batch_size = size, passage_tokens, batch_size
        kw = {"model_kwargs": {"torch_dtype": torch.float16}} if torch.cuda.is_available() else {}
        self.model = CrossEncoder(f"tomaarsen/Qwen3-Reranker-{size}-seq-cls", max_length=passage_tokens + 256, **kw)
        self.tok = self.model.tokenizer

    @property
    def judge_id(self) -> str:
        return f"ce:qwen3-reranker-{self.size}"

    def _judge(self, query, passages):
        t0 = time.perf_counter()
        ids = self.tok(passages, add_special_tokens=False, truncation=True, max_length=self.passage_tokens)["input_ids"]
        docs = [self.tok.decode(x) for x in ids]
        q = f"{self.PREFIX}<Instruct>: {self.INSTRUCTION}\n<Query>: {query}\n"
        logits = self.model.predict([(q, f"<Document>: {d}{self.SUFFIX}") for d in docs], batch_size=self.batch_size,
                                    activation_fn=torch.nn.Identity(), show_progress_bar=False)
        per_pair = (time.perf_counter() - t0) * 1000 / max(len(passages), 1)
        return [Judgment(score=float(z), p_rel=1 / (1 + math.exp(-float(z))), latency_ms=per_pair) for z in logits]
