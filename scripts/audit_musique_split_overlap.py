"""Audit MuSiQue component reuse across the existing calibration/pilot/final splits.

This reads local data/cached scores only. It preserves all splits and manuscripts.
python scripts/audit_musique_split_overlap.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from choicerank import RESULTS_DIR, ROOT
from choicerank.rag import SEED, load_split, normalize
import pandas as pd


def main():
    source = ROOT / "data/rag/musique_ans_v1.0_dev.jsonl"
    if not source.is_file():
        raise FileNotFoundError("Local MuSiQue source required; this audit does not download data.")
    records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    raw = {"musique:" + record["id"]: record for record in records}
    assert len(raw) == len(records), "Duplicate full-question IDs in input"
    ids = {split: [item.qid for item in load_split("musique", split)]
           for split in ("calib", "pilot", "final")}
    assert all(not (set(ids[a]) & set(ids[b])) for a, b in combinations(ids, 2))
    split_rows = {split: [raw[qid] for qid in qids] for split, qids in ids.items()}
    cache_checks = {}
    for split, qids in ids.items():
        for model in ("jev", "bge-m3"):
            path = ROOT / f"runs/rag/musique.{split}.{model}.parquet"
            observed = set(pd.read_parquet(path, columns=["qid"]).qid.astype(str))
            assert observed == set(qids), (split, model, "Cached IDs differ from seeded split")
            cache_checks[path.relative_to(ROOT).as_posix()] = len(observed)

    def terminal(row):
        return row["question_decomposition"][-1]

    def support(row):
        hop = terminal(row)
        paragraph = next(p for p in row["paragraphs"] if p["idx"] == hop["paragraph_support_idx"])
        return paragraph["title"], paragraph["paragraph_text"]

    def keys(row):
        return {"terminal_subquestion_id": terminal(row)["id"],
                "terminal_subquestion_text": terminal(row)["question"],
                "terminal_support_title_text": support(row),
                "normalized_whole_question": normalize(row["question"])}

    indexes, any_components = {}, {}
    for split in ("pilot", "calib"):
        indexes[split] = {name: defaultdict(list) for name in keys(split_rows[split][0])}
        any_components[split] = set()
        for row in split_rows[split]:
            qid = "musique:" + row["id"]
            for name, value in keys(row).items():
                indexes[split][name][value].append(qid)
            any_components[split].update(hop["id"] for hop in row["question_decomposition"])

    memberships = {name: {split: set() for split in ("pilot", "calib")}
                   for name in [*keys(split_rows["final"][0]), "any_subquestion_id"]}
    details = []
    for row in split_rows["final"]:
        qid = "musique:" + row["id"]
        entry = {"final_qid": qid, "terminal_subquestion_id": terminal(row)["id"],
                 "question": row["question"], "terminal_answer": terminal(row)["answer"]}
        for split in ("pilot", "calib"):
            for name, value in keys(row).items():
                if value in indexes[split][name]:
                    memberships[name][split].add(qid)
            if any(hop["id"] in any_components[split] for hop in row["question_decomposition"]):
                memberships["any_subquestion_id"][split].add(qid)
            matches = indexes[split]["terminal_subquestion_id"].get(terminal(row)["id"], [])
            entry[f"{split}_terminal_matches"] = json.dumps(matches)
            entry[f"{split}_any_component_overlap"] = int(qid in memberships["any_subquestion_id"][split])
        details.append(entry)

    comparisons = {}
    for name, by_split in memberships.items():
        p, c = by_split["pilot"], by_split["calib"]
        comparisons[name] = {"pilot": len(p), "calib": len(c), "both": len(p & c),
                             "either": len(p | c), "neither": len(ids["final"]) - len(p | c)}
    terminal_families = defaultdict(list)
    for row in records:
        terminal_families[terminal(row)["id"]].append(row)
    conflicts = {"terminal_answer": [], "terminal_support": []}
    for component, family in terminal_families.items():
        if len({normalize(terminal(row)["answer"]) for row in family}) > 1:
            conflicts["terminal_answer"].append(component)
        if len({support(row) for row in family}) > 1:
            conflicts["terminal_support"].append(component)

    report = {"dataset": "MuSiQue-Ans v1.0 dev", "seed": SEED,
              "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "definition": "Final sub-question reuse compares question_decomposition[-1]['id'] in both questions; counts are affected final questions, not unique components or matching pairs.",
              "split_sizes": {split: len(qids) for split, qids in ids.items()},
              "question_ids_disjoint": True, "cached_score_qid_checks": cache_checks,
              "comparisons": comparisons,
              "distinct_terminal_ids": {split: len({terminal(row)["id"] for row in rows})
                                        for split, rows in split_rows.items()},
              "terminal_family_conflicts": conflicts,
              "interpretation": "Question IDs are disjoint, but component families and terminal supports recur across splits. This audit does not quantify resulting performance inflation or establish family-disjoint generalization."}
    RESULTS_DIR.mkdir(exist_ok=True)
    prefix = RESULTS_DIR / "musique_split_overlap_audit"
    prefix.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with open(prefix.with_suffix(".csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
