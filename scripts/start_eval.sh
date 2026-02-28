#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# start_eval.sh — Launch full RAG evaluation over all fixed combinations.
#
# LLM models  (6):  qwen3:8b, llama3.2:latest, llama3.2:3b,
#                   gpt-oss:20b, llama2:latest, deepseek-r1:14b
# Embed models (5): bge-m3, nomic-embed-text, mxbai-embed-large,
#                   all-minilm, snowflake-arctic-embed
# Chunk configs(3): small, medium, large
# KB questions(108)
# Total: 9,720 runs
#
# Usage:
#   ./scripts/start_eval.sh             # run in foreground
#   ./scripts/start_eval.sh --background # run detached with nohup
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$ROOT_DIR/logs/run_eval_$(date +%Y%m%d_%H%M%S).log"

# Activate virtual environment if present
if [ -f "$ROOT_DIR/.venv/bin/activate" ]; then
    source "$ROOT_DIR/.venv/bin/activate"
fi

cd "$ROOT_DIR"

CMD=(python scripts/run_eval.py
    --llm "qwen3:8b"
    --llm "llama3.2:latest"
    --llm "llama3.2:3b"
    --llm "gpt-oss:20b"
    --llm "llama2:latest"
    --llm "deepseek-r1:14b"
    --embed "bge-m3"
    --embed "nomic-embed-text"
    --embed "mxbai-embed-large"
    --embed "all-minilm"
    --embed "snowflake-arctic-embed"
    --chunk "small"
    --chunk "medium"
    --chunk "large"
)

echo "============================================================"
echo " RAG Evaluation — $(date)"
echo " Log: $LOG_FILE"
echo " Combinations: 6 LLMs × 5 embeddings × 3 chunks × 108 Qs = 9,720 runs"
echo "============================================================"

# Dry-run first to confirm count
python scripts/run_eval.py \
    --llm "qwen3:8b" \
    --llm "llama3.2:latest" \
    --llm "llama3.2:3b" \
    --llm "gpt-oss:20b" \
    --llm "llama2:latest" \
    --llm "deepseek-r1:14b" \
    --embed "bge-m3" \
    --embed "nomic-embed-text" \
    --embed "mxbai-embed-large" \
    --embed "all-minilm" \
    --embed "snowflake-arctic-embed" \
    --chunk "small" \
    --chunk "medium" \
    --chunk "large" \
    --dry-run

echo ""
echo "Starting evaluation..."
echo ""

if [[ "${1:-}" == "--background" ]]; then
    nohup "${CMD[@]}" > "$LOG_FILE" 2>&1 &
    echo "Running in background. PID: $!"
    echo "Tail log with:  tail -f $LOG_FILE"
else
    "${CMD[@]}" 2>&1 | tee "$LOG_FILE"
fi
