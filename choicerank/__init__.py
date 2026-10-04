"""ChoiceRank: typed decision models as pointwise passage rerankers."""

from pathlib import Path

try:  # use the OS certificate store (fixes TLS errors behind AV/proxy HTTPS inspection)
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "runs"
RESULTS_DIR = ROOT / "results"
for _d in (RUNS_DIR, RUNS_DIR / "rag", RESULTS_DIR, ROOT / "paper" / "tables", ROOT / "paper" / "figures"):
    _d.mkdir(parents=True, exist_ok=True)  # outputs are not version-controlled; create on a fresh clone
