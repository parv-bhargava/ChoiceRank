"""Share of BM25 top-100 candidate passages that exceed each model's effective passage budget
-> results/truncation.csv. Token counts use the bge-reranker-v2-m3 (XLM-R) tokenizer."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from choicerank import RESULTS_DIR
from choicerank.data import ALL, load_queries

tok = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
rows = []
for name in ALL:
    texts = [c.text[:20000] for q in load_queries(name, 100, 0) for c in q.cands]
    lens = np.array([len(x) for x in tok(texts, add_special_tokens=False)["input_ids"]])
    chars = np.array([len(t) for t in texts])
    rows.append({"dataset": name, "pairs": len(lens), "median_tokens": int(np.median(lens)),
                 "pct_over_320": 100 * (lens > 320).mean(),   # Laya (English): ~320 passage tokens
                 "pct_over_448": 100 * (lens > 448).mean(),   # cross-encoders at 512 incl. query
                 "pct_over_2000": 100 * (lens > 2000).mean(),  # bge at 2048
                 "pct_over_4000ch": 100 * (chars > 4000).mean()})  # Jev/Jeff client cut
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(RESULTS_DIR / "truncation.csv", index=False)
