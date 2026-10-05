"""Judge factory. Model names:
  minilm, bge-m3, bge-m3-<N>           cross-encoders (free, local); bge-m3-2048 = 2048-token window
  laya, laya-multilingual              typed decision model (free, local CPU)
  jev, clef, clef-flash                typed decision models (PAID APIs)
  jeff-0.8b, jeff-2b                   typed decision model (free, local server / Colab T4)
"""

from __future__ import annotations

from .base import Judge, JudgeCache, Judgment


def make_judge(model: str, qtype: str = "choice3") -> Judge:
    if model in ("minilm", "bge-m3"):
        from .crossencoder import CrossEncoderJudge

        return CrossEncoderJudge(model)
    if model.startswith("bge-m3-"):  # e.g. bge-m3-2048: longer input window (model supports 8192)
        from .crossencoder import CrossEncoderJudge

        return CrossEncoderJudge("bge-m3", max_length=int(model.split("-")[-1]), label=model)
    if model in ("qwen08base-yesno", "qwen08base-graded"):  # Jeff's base model, local (free)
        from .local_yesno import LocalYesNoJudge

        return LocalYesNoJudge(model)
    if model.endswith(("-yesno", "-graded")):  # label-probability pointwise LLM baseline (PAID)
        from .llm_yesno import LLMYesNoJudge

        return LLMYesNoJudge(model)
    if model.startswith("qwen3-reranker-"):  # e.g. qwen3-reranker-0.6B: label-probability (P(yes)) LLM reranker
        from .crossencoder import Qwen3RerankerJudge

        return Qwen3RerankerJudge(model.split("-")[-1])
    if model == "laya":
        from .laya_judge import LayaJudge

        return LayaJudge(qtype)
    if model == "laya-multilingual":
        from .laya_judge import LayaJudge

        return LayaJudge(qtype, variant="multilingual", max_len=2048)
    from .decision_api import DecisionJudge

    return DecisionJudge(model, qtype)


__all__ = ["Judge", "JudgeCache", "Judgment", "make_judge"]
