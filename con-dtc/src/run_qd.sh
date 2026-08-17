#!/usr/bin/env bash
# 用法: bash src/run_qd.sh <config路径>
# 必须显式指定实验 YAML 配置
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"

cd "$PROJECT_DIR"

if [[ $# -ne 1 ]]; then
    echo "用法: bash src/run_qd.sh <config路径>"
    echo "示例: bash src/run_qd.sh configs/baseline/condtc_qd_baseline_pre15.yaml"
    exit 1
fi

CONFIG_PATH="$1"
LOG_DIR="logs"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_PATH="${LOG_DIR}/$(basename "$CONFIG_PATH" .yaml)_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Config not found: $CONFIG_PATH"
    exit 1
fi

# 从配置里解析预训练 checkpoint 路径并检查存在性
PRETRAIN_PATH="$(grep -E '^\s*pretrain_path:' "$CONFIG_PATH" | awk '{print $2}')"
if [[ -z "$PRETRAIN_PATH" ]]; then
    echo "pretrain_path not found in config: $CONFIG_PATH"
    exit 1
fi
if [[ ! -f "$PRETRAIN_PATH" ]]; then
    echo "Pretrain checkpoint not found: $PRETRAIN_PATH"
    exit 1
fi

echo "Project: $PROJECT_DIR"
echo "Python: $PYTHON_BIN"
echo "GPU: $GPU_ID"
echo "Config: $CONFIG_PATH"
echo "Pretrain: $PRETRAIN_PATH"
echo "Log: $LOG_PATH"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1

"$PYTHON_BIN" -m src.experiment.run \
    --config "$CONFIG_PATH" \
    2>&1 | tee "$LOG_PATH"
