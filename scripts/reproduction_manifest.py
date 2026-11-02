"""Record full input hashes, split IDs and analysis provenance without model/API calls.

python scripts/reproduction_manifest.py -> results/reproduction_manifest.json
Historical model weight revisions are not inferred from architecture strings.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from choicerank import DATA_DIR, RESULTS_DIR, ROOT, RUNS_DIR
from choicerank.rag import FD_SPLITS, SEED, load_split, normalize


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    sources = {p.relative_to(ROOT).as_posix(): {"bytes": p.stat().st_size, "sha256": digest(p)}
               for p in sorted(DATA_DIR.rglob("*")) if p.is_file() and p.suffix != ".part"}
    splits, shared_questions = {}, []
    for dataset in ("musique", "nq"):
        loaded = {split: load_split(dataset, split) for split in ("calib", "pilot", "final")}
        splits[dataset] = {split: [item.qid for item in items] for split, items in loaded.items()}
        for a, b in (("calib", "pilot"), ("calib", "final"), ("pilot", "final")):
            if set(splits[dataset][a]) & set(splits[dataset][b]):
                raise ValueError(f"Overlapping split IDs: {dataset} {a}/{b}")
            by_question = {normalize(it.question): it.qid for it in loaded[a]}
            shared_questions += [dict(dataset=dataset, split_a=a, split_b=b,
                                      qid_a=by_question[normalize(it.question)], qid_b=it.qid)
                                 for it in loaded[b] if normalize(it.question) in by_question]
    splits["musique"] |= {split: [item.qid for item in load_split("musique", split)] for split in FD_SPLITS}
    if set(splits["musique"]["calib_fd"]) & set(splits["musique"]["final_fd"]):
        raise ValueError("Overlapping split IDs: musique calib_fd/final_fd")
    packages = {}
    for name in ("torch", "transformers", "sentence-transformers", "ir_measures", "numpy", "scipy", "pandas",
                 "pyarrow", "matplotlib", "laya", "bitsandbytes"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    provenance = RUNS_DIR / "manifest.jsonl"
    component_audit_path = RESULTS_DIR / "musique_split_overlap_audit.json"
    component_audit = None
    if component_audit_path.exists():
        audit = json.loads(component_audit_path.read_text(encoding="utf-8"))
        if audit["source_sha256"] != digest(DATA_DIR / "rag/musique_ans_v1.0_dev.jsonl"):
            raise ValueError("MuSiQue component audit is stale for the current source")
        component_audit = dict(path=component_audit_path.relative_to(ROOT).as_posix(),
                               sha256=digest(component_audit_path), comparisons=audit["comparisons"])
    fd_audit_path = RESULTS_DIR / "musique_fd_split_audit.json"
    fd_audit = (dict(path=fd_audit_path.relative_to(ROOT).as_posix(), sha256=digest(fd_audit_path))
                if fd_audit_path.exists() else None)
    out = dict(created_utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
               packages=packages, split_seed=SEED, split_ids=splits,
               normalized_question_overlap=shared_questions,
               normalized_question_overlap_scope="Whole normalized wording only; this does not measure decomposition-family reuse",
               musique_component_overlap_audit=component_audit, musique_family_disjoint_audit=fd_audit,
               input_files=sources,
               analysis_source_sha256={p.relative_to(ROOT).as_posix(): digest(p)
                                       for folder in (ROOT / "scripts", ROOT / "choicerank")
                                       for p in sorted(folder.rglob("*.py"))},
               run_manifest_sha256=digest(provenance) if provenance.exists() else None,
               git_commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                         text=True, check=True).stdout.strip(),
               model_revision_note="Historical HF model commits were not recorded. Architecture strings do not identify exact weights; preserve original run outputs and manifests.")
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / "reproduction_manifest.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}: {len(sources)} input hashes, disjoint split IDs, "
          f"{len(shared_questions)} normalized cross-split question overlap(s)")


if __name__ == "__main__":
    main()
