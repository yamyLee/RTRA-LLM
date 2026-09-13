#!/usr/bin/env bash
# =============================================================================
# scripts/run_prompt_ablation.sh  ——  RTRA-LLM 提示词消融实验
# =============================================================================
# 用法：
#   bash scripts/run_prompt_ablation.sh [provider]
#
# 例子：
#   bash scripts/run_prompt_ablation.sh deepseek            # 使用 deepseek
#   bash scripts/run_prompt_ablation.sh zhipu               # 使用 zhipu
#   SMOKE=1 bash scripts/run_prompt_ablation.sh deepseek    # 冒烟测试
#   VARIANTS="none full" bash scripts/run_prompt_ablation.sh deepseek   # 仅测2种变体
#   CASES="1 2 7 8" bash scripts/run_prompt_ablation.sh deepseek
#
# 提示词变体（对应 prompt/ 目录）：
#   none     ── 无COLREGs知识（prompt_none.txt）
#   weak     ── 弱海事引导（prompt_weak.txt）
#   partial  ── 部分COLREGs引导（prompt_partial.txt）
#   full     ── 完整规则感知（prompt_full.txt，论文方法）
#
# 注意：实验自动备份并恢复 prompt/prompt.txt，无需手动操作。
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

# ── 参数解析 ─────────────────────────────────────────────────────────────────
PROVIDER="${1:-deepseek}"
SMOKE="${SMOKE:-0}"
CASES="${CASES:-}"
VARIANTS="${VARIANTS:-}"
SIM_TIME="${SIM_TIME:-450.0}"
RISK_THRESHOLD="${RISK_THRESHOLD:-0.30}"

# ── 检查 prompt 目录 ──────────────────────────────────────────────────────────
for f in prompt_none.txt prompt_weak.txt prompt_partial.txt prompt_full.txt; do
    if [[ ! -f "prompt/$f" ]]; then
        echo "[WARN] prompt/$f not found — this variant will be skipped"
    fi
done

# ── 构建命令 ─────────────────────────────────────────────────────────────────
CMD=(python3 run_prompt_ablation.py
     --provider "$PROVIDER"
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

if [[ -n "$VARIANTS" ]]; then
    CMD+=(--variants $VARIANTS)
fi

# ── 打印运行信息 ──────────────────────────────────────────────────────────────
echo "======================================================================"
echo " RTRA-LLM PROMPT ABLATION"
echo "======================================================================"
echo " Provider    : $PROVIDER"
echo " Smoke       : $SMOKE"
echo " Variants    : ${VARIANTS:-none weak partial full}"
echo " Cases       : ${CASES:-all 23}"
echo " Sim time    : ${SIM_TIME}s"
echo " Command     : ${CMD[*]}"
echo "======================================================================"

# ── 执行 ─────────────────────────────────────────────────────────────────────
"${CMD[@]}"
