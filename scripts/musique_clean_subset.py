"""Retrospective MuSiQue component-clean sensitivity analysis; cached answers only.

python scripts/audit_musique_split_overlap.py
python scripts/musique_clean_subset.py

The subset contains no sub-question ID occurring in calibration or pilot.
Existing fits, readouts, policies, and generated answers are kept fixed.
Family intervals resample whole-source connected components, retaining the
question-weighted estimand. These are exploratory, unadjusted percentile CIs.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from choicerank import RESULTS_DIR, ROOT, RUNS_DIR
from choicerank.rag import load_split
from scripts.rag_summary import B_REFS, GENS, SHARED, holm, paired, per_question

BOOTSTRAP_SEED = 13
BOOTSTRAP_REPLICATES = 20_000


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean_membership(raw):
    design_qids = {x.qid for s in ("calib", "pilot") for x in load_split("musique", s)}
    design_ids = {h["id"] for q in design_qids for h in raw[q]["question_decomposition"]}
    final = sorted(x.qid for x in load_split("musique", "final"))
    clean = [q for q in final if not any(h["id"] in design_ids
                                       for h in raw[q]["question_decomposition"])]

    # Connectivity uses ALL source records, including bridges outside the subset.
    # A canonical component ID is its smallest full-question ID.
    parent = {q: q for q in raw}

    def find(q):
        while parent[q] != q:
            parent[q] = parent[parent[q]]
            q = parent[q]
        return q

    seen = {}
    for q in sorted(raw):
        for hop in raw[q]["question_decomposition"]:
            component = hop["id"]
            if component in seen:
                a, b = find(q), find(seen[component])
                parent[max(a, b)] = min(a, b)
            else:
                seen[component] = q
    families = {q: find(q) for q in raw}
    design_families = {families[q] for q in design_qids}
    membership = pd.DataFrame({
        "qid": clean,
        "terminal_subquestion_id": [raw[q]["question_decomposition"][-1]["id"] for q in clean],
        "source_family_id": [families[q] for q in clean],
        "indirect_family_connection_to_design": [families[q] in design_families for q in clean],
    })
    audit = pd.read_csv(RESULTS_DIR / "musique_split_overlap_audit.csv")
    audited = set(audit.loc[(audit.pilot_any_component_overlap == 0) &
                           (audit.calib_any_component_overlap == 0), "final_qid"])
    if set(clean) != audited or len(clean) != 329:
        raise ValueError("Clean subset differs from the independently recorded overlap audit")
    return membership, Counter(families.values())


def main():
    source = ROOT / "data/rag/musique_ans_v1.0_dev.jsonl"
    raw = {"musique:" + row["id"]: row
           for row in map(json.loads, source.read_text(encoding="utf-8").splitlines())}
    membership, source_sizes = clean_membership(raw)
    qids = membership.qid.tolist()
    family_ids = sorted(membership.source_family_id.unique())
    codes = pd.Categorical(membership.source_family_id, categories=family_ids).codes
    counts = np.bincount(codes, minlength=len(family_ids))
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, len(family_ids), size=(BOOTSTRAP_REPLICATES, len(family_ids)))
    draw_counts = counts[draws].sum(axis=1)

    def family_interval(differences):
        sums = np.bincount(codes, weights=differences, minlength=len(family_ids))
        values = sums[draws].sum(axis=1) / draw_counts
        return np.quantile(values, [0.025, 0.975]).tolist()

    summary, tests, hashes = [], [], {source.relative_to(ROOT).as_posix(): digest(source)}
    for gen, label in GENS.items():
        scores = per_question("musique", gen)
        selected = scores[scores.qid.isin(qids)]
        wide = selected.pivot(index="qid", columns="policy", values="F1").loc[qids]
        if len(wide) != 329 or wide.isna().any().any():
            raise ValueError(f"Incomplete clean-subset policy grid: {gen}")
        aggregate = selected.groupby("policy").agg(F1=("F1", "mean"), EM=("EM", "mean"),
                                                   passages=("n", "mean"), tokens=("tokens", "mean"),
                                                   empty=("empty", "mean"))
        for policy, row in aggregate.iterrows():
            summary.append(dict(gen=label, policy=policy, n_questions=len(qids),
                                F1=100 * row.F1, EM=100 * row.EM, passages=row.passages,
                                tokens=row.tokens, empty_pct=100 * row["empty"]))
        families = {
            "ranker_effect": [(f"jev:{p}", f"bge-m3:{p}") for p in SHARED],
            "threshold_rule": [("jev:floor2+thr0.3", b) for b in B_REFS],
        }
        for kind, pairs in families.items():
            rows = []
            for a, b in pairs:
                row = paired(wide[a], wide[b], dict(gen=label, comparison=kind, A=a, B=b,
                                                  n_questions=len(qids), n_source_families=len(family_ids),
                                                  analysis_design="retrospective_cached_sensitivity"))
                row["family_ci_lo"], row["family_ci_hi"] = family_interval(
                    100 * (wide[a] - wide[b]).to_numpy())
                rows.append(row)
            for row, adjusted in zip(rows, holm([row["p"] for row in rows])):
                row["p_holm_question_level"] = adjusted
            tests.extend(rows)
        slug = gen.replace("/", "_")
        for kind in ("gen", "per_q"):
            path = RUNS_DIR / "rag" / f"musique.final.{kind}.{slug}.parquet"
            hashes[path.relative_to(ROOT).as_posix()] = digest(path)

    RESULTS_DIR.mkdir(exist_ok=True)
    membership.to_csv(RESULTS_DIR / "musique_clean_subset_ids.csv", index=False)
    S, T = pd.DataFrame(summary), pd.DataFrame(tests)
    S.to_csv(RESULTS_DIR / "musique_clean_subset_summary.csv", index=False)
    T.to_csv(RESULTS_DIR / "musique_clean_subset_tests.csv", index=False)
    metadata = dict(
        design="Retrospective sensitivity analysis; no model inference or policy refitting",
        subset_definition="No decomposition sub-question ID occurs in calibration or pilot",
        n_final=1000, n_clean=len(qids), n_distinct_terminal_ids=int(membership.terminal_subquestion_id.nunique()),
        n_source_families=len(family_ids), largest_clean_family=int(counts.max()),
        n_source_graph_families=len(source_sizes), largest_source_family=max(source_sizes.values()),
        n_clean_questions_indirectly_connected_to_design=int(membership.indirect_family_connection_to_design.sum()),
        n_clean_families_indirectly_connected_to_design=int(membership.loc[
            membership.indirect_family_connection_to_design, "source_family_id"].nunique()),
        family_definition="Connected components over all 2,417 source questions linked by any shared sub-question ID",
        bootstrap=dict(seed=BOOTSTRAP_SEED, replicates=BOOTSTRAP_REPLICATES,
                       method="Paired family percentile bootstrap; sample G families with replacement; sum differences / sum questions; unadjusted 95% intervals"),
        question_tests="Paired t-tests with Holm over 9 ranker-effect or 4 threshold-rule tests separately per generator; question independence is not assumed by family CIs, but remains assumed by these legacy tests",
        unchanged="Calibration fits, probability readout, policy thresholds, and generated answers",
        limitations="Changed case mix; not a new confirmatory test or contamination audit. Direct component exclusion does not guarantee transitive family isolation from design splits.",
        input_sha256=hashes,
    )
    (RESULTS_DIR / "musique_clean_subset_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    lines = [r"\begin{table}[t]", r"\centering\footnotesize\setlength{\tabcolsep}{3pt}",
             r"\caption{Retrospective MuSiQue sensitivity on 329 questions with no sub-question ID in calibration or pilot. Top-3 answer F1; $\Delta$ is Jev minus bge. CIs resample 212 source-graph families (20,000 paired replicates), retaining question weights; intervals are unadjusted. No top-3 comparison passes the original nine-policy question-level Holm correction.}",
             r"\label{tab:rag-clean}", r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Generator & Jev & bge & $\Delta$ & Family 95\% CI \\", r"\midrule"]
    for label in GENS.values():
        subset = S[S.gen == label].set_index("policy")
        test = T[(T.gen == label) & (T.A == "jev:top3") & (T.B == "bge-m3:top3")].iloc[0]
        lines.append(f"{label} & {subset.loc['jev:top3'].F1:.1f} & {subset.loc['bge-m3:top3'].F1:.1f} & "
                     f"$+{test.delta_F1:.1f}$ & $[{test.family_ci_lo:.1f}, {test.family_ci_hi:.1f}]$ " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (ROOT / "paper/tables/rag-clean.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in metadata.items() if k != "input_sha256"}, indent=2))
    print(T[T.A.eq("jev:top3")].to_string(index=False))
    ranker = T[T.comparison.eq("ranker_effect")]
    print("Clean ranker delta range:", ranker.delta_F1.min(), ranker.delta_F1.max())
    print("Question-level Holm < .05:", int((ranker.p_holm_question_level < .05).sum()), "/", len(ranker))


if __name__ == "__main__":
    main()
