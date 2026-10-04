#!/usr/bin/env bash
# Free local runs on all 13 collections at BM25 top-100 (GPU). Resumable: every judgment is
# cached in runs/cache.sqlite, so re-running skips finished pairs.
#   bash scripts/run_local_all.sh            # from the ChoiceRank folder (Git Bash)
# Log: runs/local_all.log
set -u
cd "$(dirname "$0")/.."
PY="$HOME/.venvs/choicerank/Scripts/python"
JEFF_DIR="$HOME/tools/jeff"
LOG=runs/local_all.log
# Must be Git Bash: in PowerShell, `bash` is WSL, where this Windows venv does not exist.
[ -x "$PY" ] || [ -x "$PY.exe" ] || { echo "Python venv not found at $PY (use Git Bash, or scripts/run_local_all.ps1 from PowerShell)" | tee -a "$LOG"; exit 1; }
step() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

step "MiniLM (GPU)";      "$PY" scripts/run_rerank.py --model minilm --dataset all 2>&1 | grep -E --line-buffered "queries in|Error|Traceback" | tee -a "$LOG"
step "bge-reranker-v2-m3"; "$PY" scripts/run_rerank.py --model bge-m3 --dataset all 2>&1 | grep -E --line-buffered "queries in|Error|Traceback" | tee -a "$LOG"
step "Laya (3 formats)";  "$PY" scripts/run_rerank.py --model laya --qtype all --dataset all 2>&1 | grep -E --line-buffered "queries in|Error|Traceback" | tee -a "$LOG"

step "Jeff 0.8B: starting server"
( cd "$JEFF_DIR" && JEFF_CHECKPOINT=Jeff-Qwen3.5-0.8B-v1.2 JEFF_DEVICE=cuda PORT=8765 .venv/Scripts/jeff-serve >> "$HOME/tools/jeff-serve.log" 2>&1 ) &
SERVER=$!
for _ in $(seq 1 100); do curl -s localhost:8765/health | grep -q ready && break; sleep 3; done
step "Jeff 0.8B (3 formats)"; "$PY" scripts/run_rerank.py --model jeff-0.8b --qtype all --dataset all 2>&1 | grep -E --line-buffered "queries in|Error|Traceback" | tee -a "$LOG"
kill $SERVER 2>/dev/null; taskkill //F //IM jeff-serve.exe >/dev/null 2>&1
step "DONE"
