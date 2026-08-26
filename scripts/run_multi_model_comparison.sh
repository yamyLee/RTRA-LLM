#!/usr/bin/env bash
# =============================================================================
# scripts/run_multi_model_comparison.sh  ——  RTRA-LLM 多模型对比实验
# =============================================================================
# 用法：
#   bash scripts/run_multi_model_comparison.sh [deepseek|zhipu|all|custom]
#
# 例子：
#   bash scripts/run_multi_model_comparison.sh                     # 默认 deepseek + zhipu
#   bash scripts/run_multi_model_comparison.sh deepseek            # 仅 deepseek
#   bash scripts/run_multi_model_comparison.sh all                 # deepseek zhipu qwen openai
#   SMOKE=1 bash scripts/run_multi_model_comparison.sh             # 冒烟测试（仅 case 1）
#   CASES="1 2 7 8" bash scripts/run_multi_model_comparison.sh     # 指定场景子集
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."          # 切换到项目根目录

# ── 参数解析 ─────────────────────────────────────────────────────────────────
MODE="${1:-default}"

case "$MODE" in
  deepseek) PROVIDERS="deepseek" ;;
  minimax)    PROVIDERS="minimax" ;;
  qwen)   PROVIDERS="qwen" ;;
  all)      PROVIDERS="deepseek minimax qwen" ;;
  *)        PROVIDERS="deepseek minimax qwen" ;;   # 默认
esac

# 允许环境变量覆盖
SMOKE="${SMOKE:-0}"
CASES="${CASES:-}"
SIM_TIME="${SIM_TIME:-450.0}"
RISK_THRESHOLD="${RISK_THRESHOLD:-0.30}"

# ── 构建命令 ─────────────────────────────────────────────────────────────────
CMD=(python3 run_multi_model_comparison.py
     --providers $PROVIDERS
     --sim_time "$SIM_TIME"
     --risk_threshold "$RISK_THRESHOLD"
)

if [[ "$SMOKE" == "1" ]]; then
    CMD+=(--smoke)
    echo "[SMOKE] 冒烟测试模式：仅 case 1"
fi

if [[ -n "$CASES" ]]; then
    CMD+=(--cases $CASES)
fi

# ── 打印运行信息 ──────────────────────────────────────────────────────────────
echo "======================================================================"
echo " CORALL MULTI-MODEL COMPARISON"
echo "======================================================================"
echo " Providers   : $PROVIDERS"
echo " Smoke       : $SMOKE"
echo " Cases       : ${CASES:-all 23}"
echo " Sim time    : ${SIM_TIME}s"
echo " Command     : ${CMD[*]}"
echo "======================================================================"

# ── 执行 ─────────────────────────────────────────────────────────────────────
"${CMD[@]}"
