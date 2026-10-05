"""Label-probability pointwise baseline with a LOCAL base model (free): P(yes) from next-token probabilities.

Same prompt and 448-token passage budget as the OpenRouter yes/no baselines (llm_yesno.py). Used as a same-base
control for Jeff, whose base model is Qwen/Qwen3.5-0.8B (model card of mstrasser/Jeff-Qwen3.5-0.8B).
P(yes) = p(Yes) / (p(Yes) + p(No)), summing the case/space variants of each answer token over the full next-token
distribution (the API baselines can only see the top-5 tokens; with a dominant Yes/No token the two coincide).
"""

from __future__ import annotations

import time

import torch

from .base import Judge, Judgment
from .llm_yesno import GRADED_USER, SYSTEM, USER

MODELS = {"qwen08base-yesno": "Qwen/Qwen3.5-0.8B",  # Jeff's base model
          "qwen08base-graded": "Qwen/Qwen3.5-0.8B"}  # same model, 0-4 rating readout (Zhuang et al.)


class LocalYesNoJudge(Judge):
    paid = False

    def __init__(self, alias: str, passage_tokens: int = 448, batch_size: int = 16):
        from transformers import AutoTokenizer

        self.alias, self.passage_tokens, self.batch_size = alias, passage_tokens, batch_size
        name = MODELS[alias]
        self.tok = AutoTokenizer.from_pretrained(name)
        self.tok.padding_side = "left"
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if dev == "cuda" else torch.float32
        try:
            from transformers import AutoModelForCausalLM

            self.model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(dev).eval()
        except (ValueError, KeyError):  # multimodal checkpoints: load the image-text-to-text class, text only
            from transformers import AutoModelForImageTextToText

            self.model = AutoModelForImageTextToText.from_pretrained(name, dtype=dtype).to(dev).eval()
        self.dev = dev
        self.cut_tok = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")  # same budget as llm_yesno
        vocab = self.tok.get_vocab()
        variants = lambda w: {w, w.lower(), w.upper(), " " + w, " " + w.lower(), "Ġ" + w, "Ġ" + w.lower()}
        self.yes_ids = sorted({vocab[v] for v in variants("Yes") if v in vocab})
        self.no_ids = sorted({vocab[v] for v in variants("No") if v in vocab})
        assert self.yes_ids and self.no_ids
        self.graded = alias.endswith("-graded")
        self.digit_ids = [vocab[str(k)] for k in range(5)]

    @property
    def judge_id(self) -> str:
        return f"local:{self.alias}"

    def _prompt(self, query: str, passage: str) -> str:
        user = (GRADED_USER if self.graded else USER).format(q=query, p=passage)
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        try:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    @torch.inference_mode()
    def _judge(self, query, passages):
        t0 = time.perf_counter()
        ids = self.cut_tok(passages, add_special_tokens=False, truncation=True, max_length=self.passage_tokens)["input_ids"]
        cut = [self.cut_tok.decode(x, skip_special_tokens=True) for x in ids]
        out = []
        for i in range(0, len(cut), self.batch_size):
            enc = self.tok([self._prompt(query, p) for p in cut[i:i + self.batch_size]], return_tensors="pt",
                           padding=True, add_special_tokens=False).to(self.dev)
            logits = self.model(**enc, logits_to_keep=1).logits[:, -1, :].float()  # last position only (large vocab)
            probs = torch.softmax(logits, dim=-1)
            if self.graded:  # expected rating over the five digit tokens, renormalised
                pd_ = probs[:, self.digit_ids]
                pd_ = pd_ / pd_.sum(-1, keepdim=True).clamp_min(1e-12)
                exp = (pd_ * torch.arange(5, device=pd_.device)).sum(-1) / 4
                out += [(float(e), float(q[2:].sum())) for e, q in zip(exp, pd_)]  # (E[r]/4, P(r>=2))
                continue
            py, pn = probs[:, self.yes_ids].sum(-1), probs[:, self.no_ids].sum(-1)
            out += [(float(a), float(b)) for a, b in zip(py, pn)]
        per = (time.perf_counter() - t0) * 1000 / max(len(passages), 1)
        if self.graded:
            return [Judgment(score=e, p_rel=p2, latency_ms=per, raw={"expected_rating": 4 * e}) for e, p2 in out]
        return [Judgment(score=a / (a + b) if a + b > 0 else 0.5, p_rel=a / (a + b) if a + b > 0 else 0.5,
                         latency_ms=per, raw={"p_yes": a, "p_no": b}) for a, b in out]
