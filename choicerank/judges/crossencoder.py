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

    passage_chars (equal-context control, issue #5): first cut each passage to the same first N characters that the
    decision models read (4,000), with a token cap high enough not to cut further; `label` names the configuration.

    4B/8B load in 4-bit NF4 on CUDA (8 GB GPU) unless full_precision=True (issue #10: bf16 on a >=24 GB GPU);
    the bf16 configuration has its own judge ID ("...-bf16") so its cache rows and runs stay separate.
    """

    PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the '
              'Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
    INSTRUCTIONS = {  # prompt-sensitivity paraphrases (issue #12), fixed before any paraphrase run
        "instr1": "Given a search query, judge whether the passage is relevant to it",
        "instr2": "Given a search query, find passages that would be useful to the person who issued it",
        "instr3": "Given a search query, retrieve passages that contain the information it asks for",
    }

    def __init__(self, size: str = "0.6B", passage_tokens: int = 448, batch_size: int = 16,
                 passage_chars: int | None = None, label: str | None = None, instruction: str | None = None,
                 full_precision: bool = False):
        # Direct transformers path (sentence-transformers >= 6 rejects the 4B/8B checkpoints' chat template for pairs).
        # A query-document pair is tokenized as a text pair, which for this tokenizer is plain concatenation,
        # exactly as sentence-transformers' CrossEncoder did for the 0.6B runs (verified identical).
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.size, self.passage_tokens, self.batch_size = size, passage_tokens, batch_size
        self.passage_chars, self.label = passage_chars, label
        self.instruction = instruction  # None: the default INSTRUCTION; else a key of INSTRUCTIONS
        self.full_precision = full_precision
        name = f"tomaarsen/Qwen3-Reranker-{size}-seq-cls"
        self.tok = AutoTokenizer.from_pretrained(name)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        if size in ("4B", "8B") and self.dev == "cuda" and not full_precision:  # 8 GB GPU: 4-bit NF4, fp16 compute
            from transformers import BitsAndBytesConfig

            q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
            self.model = AutoModelForSequenceClassification.from_pretrained(name, quantization_config=q, device_map="cuda")
            self.batch_size = 8 if passage_tokens <= 512 else 2  # long inputs: bound GPU memory
        else:
            dtype = (torch.bfloat16 if full_precision else torch.float16) if self.dev == "cuda" else torch.float32
            self.model = AutoModelForSequenceClassification.from_pretrained(name, dtype=dtype).to(self.dev)
        self.model.eval()
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tok.pad_token_id

    @property
    def judge_id(self) -> str:
        if self.label:
            return f"ce:{self.label}"
        return (f"ce:qwen3-reranker-{self.size}" + (f"-{self.instruction}" if self.instruction else "")
                + ("-bf16" if self.full_precision else ""))

    def _judge(self, query, passages):
        t0 = time.perf_counter()
        if self.passage_chars:
            passages = [p[: self.passage_chars] for p in passages]
        ids = self.tok(passages, add_special_tokens=False, truncation=True, max_length=self.passage_tokens)["input_ids"]
        docs = [self.tok.decode(x) for x in ids]
        instruction = self.INSTRUCTIONS[self.instruction] if self.instruction else self.INSTRUCTION
        q = f"{self.PREFIX}<Instruct>: {instruction}\n<Query>: {query}\n"
        logits = []
        with torch.inference_mode():
            for i in range(0, len(docs), self.batch_size):
                part = [f"<Document>: {d}{self.SUFFIX}" for d in docs[i:i + self.batch_size]]
                enc = self.tok([q] * len(part), part, padding=True, truncation=True,
                               max_length=self.passage_tokens + 256, return_tensors="pt").to(self.dev)
                logits += self.model(**enc).logits[:, 0].float().tolist()
        per_pair = (time.perf_counter() - t0) * 1000 / max(len(passages), 1)
        return [Judgment(score=float(z), p_rel=1 / (1 + math.exp(-float(z))), latency_ms=per_pair) for z in logits]
