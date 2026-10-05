"""Listwise reranking baseline: RankZephyr-7B (castorini/rank_zephyr_7b_v1_full) on the BM25 top-100, locally.

Sliding window as in RankGPT/RankLLM (window 20, step 10, from the bottom of the list to the top), RankLLM's
RankZephyr prompt, greedy decoding. On an 8 GB GPU the model is loaded in 4-bit NF4 (fp16 compute), which can differ
slightly from the published bf16 runs. Each finished query is cached in runs/listwise_<model>.jsonl, so an
interrupted run resumes where it stopped.

  python scripts/run_listwise.py --dataset dl19 dl20
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from tqdm import tqdm

from choicerank import RUNS_DIR
from choicerank.data import GROUPS, load_queries
from scripts.run_rerank import write_trec

MODEL = "castorini/rank_zephyr_7b_v1_full"
# RankZephyr's slow SentencePiece file is misread by transformers 5; Zephyr-7B-beta (its base) has the same
# 32k vocabulary and an identical chat template (checked), and ships a fast tokenizer.
TOKENIZER = "HuggingFaceH4/zephyr-7b-beta"
SYSTEM = "You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query."
CONTEXT, OUT_TOKENS, WINDOW, STEP = 4096, 120, 20, 10


class RankZephyr:
    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.tok = AutoTokenizer.from_pretrained(TOKENIZER)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16), device_map="cuda").eval()

    def _prompt(self, query: str, passages: list[str], per_passage: int) -> str:
        cut = [self.tok.decode(self.tok(p, add_special_tokens=False)["input_ids"][:per_passage]) for p in passages]
        n = len(passages)
        user = (f"I will provide you with {n} passages, each indicated by a numerical identifier []. "
                f"Rank the passages based on their relevance to the search query: {query}.\n\n"
                + "".join(f"[{i + 1}] {c.strip()}\n" for i, c in enumerate(cut))
                + f"\nSearch Query: {query}.\nRank the {n} passages above based on their relevance to the search query. "
                  "All the passages should be included and listed using identifiers, in descending order of relevance. "
                  "The output format should be [] > [], e.g., [2] > [1], Only respond with the ranking results, "
                  "do not say any word or explain.")
        return self.tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                                            tokenize=False, add_generation_prompt=True)

    @torch.inference_mode()
    def rank_window(self, query: str, passages: list[str]) -> list[int]:
        per = 300  # RankLLM starts at 300 tokens per passage and shrinks until the prompt fits the context
        while True:
            prompt = self._prompt(query, passages, per)
            ids = self.tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
            if ids["input_ids"].shape[1] <= CONTEXT - OUT_TOKENS or per <= 30:
                break
            per -= 20
        out = self.model.generate(**ids, max_new_tokens=OUT_TOKENS, do_sample=False, pad_token_id=self.tok.eos_token_id)
        text = self.tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
        order, seen = [], set()
        for m in re.findall(r"\[(\d+)\]", text):
            k = int(m) - 1
            if 0 <= k < len(passages) and k not in seen:
                order.append(k); seen.add(k)
        return order + [k for k in range(len(passages)) if k not in seen]  # missing ids keep their input order

    def rerank(self, query: str, docs: list[tuple[str, str]]) -> list[str]:
        docs = list(docs)
        end = len(docs)
        while True:
            start = max(0, end - WINDOW)
            window = docs[start:end]
            order = self.rank_window(query, [t for _, t in window])
            docs[start:end] = [window[k] for k in order]
            if start == 0:
                break
            end -= STEP
        return [d for d, _ in docs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="+", required=True, help=f"names or groups: {sorted(GROUPS)}")
    ap.add_argument("--limit-queries", type=int, default=0, help="smoke test: first N queries only (no TREC file)")
    args = ap.parse_args()
    names = [n for d in args.dataset for n in GROUPS.get(d, [d])]
    cache_path = RUNS_DIR / "listwise_rankzephyr.jsonl"
    cache = {}
    if cache_path.exists():
        for line in open(cache_path, encoding="utf-8"):
            r = json.loads(line); cache[(r["dataset"], r["qid"])] = r["order"]
    rz = None
    for name in names:
        queries = load_queries(name, topk=100, limit=0)[: args.limit_queries or None]
        run, t0 = {}, time.perf_counter()
        for q in tqdm(queries, desc=f"{name}.rankzephyr"):
            key = (name, q.qid)
            if key not in cache:
                rz = rz or RankZephyr()
                cache[key] = rz.rerank(q.text, [(c.docid, c.text) for c in q.cands])
                with open(cache_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"dataset": name, "qid": q.qid, "order": cache[key]}) + "\n")
            n = len(cache[key])
            run[q.qid] = {d: float(n - i) for i, d in enumerate(cache[key])}
        if args.limit_queries:
            print(f"{name}: smoke test, {len(queries)} queries in {time.perf_counter() - t0:.0f}s"); continue
        write_trec(RUNS_DIR / f"{name}.top100.rankzephyr.trec", run, "rankzephyr")
        print(f"{name}.top100.rankzephyr: {len(queries)} queries in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
