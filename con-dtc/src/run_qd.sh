#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-trajectory-clustering}"
GPU_ID="${GPU_ID:-0}"

cd "$PROJECT_DIR"

CONFIG_PATH="configs/condtc_qd.yaml"
PRETRAIN_PATH="checkpoints/sttraj2vec_pretrain_best.pt"
LOG_DIR="logs"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_PATH="${LOG_DIR}/condtc_qd_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Config not found: $CONFIG_PATH"
    exit 1
fi

if [[ ! -f "$PRETRAIN_PATH" ]]; then
    echo "Pretrain checkpoint not found: $PRETRAIN_PATH"
    exit 1
fi

echo "Project: $PROJECT_DIR"
echo "Conda env: $CONDA_ENV"
echo "GPU: $GPU_ID"
echo "Config: $CONFIG_PATH"
echo "Log: $LOG_PATH"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1

conda run \
    --no-capture-output \
    -n "$CONDA_ENV" \
    python -m src.experiment.run \
    --config "$CONFIG_PATH" \
    2>&1 | tee "$LOG_PATH"