#!/usr/bin/env bash
# =============================================================================
# scripts/run_module_ablation.sh  ——  RTRA-LLM 模块消融实验
# =============================================================================
# 用法：
#   bash scripts/run_module_ablation.sh [provider]
#
# 例子：
#   bash scripts/run_module_ablation.sh deepseek            # 使用 deepseek
#   bash scripts/run_module_ablation.sh zhipu               # 使用 zhipu
#   SMOKE=1 bash scripts/run_module_ablation.sh deepseek    # 冒烟测试
#   CASES="1 2 7" bash scripts/run_module_ablation.sh deepseek
#   SKIP="5" bash scripts/run_module_ablation.sh deepseek   # 跳过条件5（wo_memory+wo_val）
#
# 消融条件 ID：
#   0=baseline  1=full  2=wo_risk_trigger  3=wo_memory  4=wo_validator  5=wo_memory_wo_validator
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

# ── 参数解析 ─────────────────────────────────────────────────────────────────
PROVIDER="${1:-deepseek}"
SMOKE="${SMOKE:-0}"
CASES="${CASES:-}"
SKIP="${SKIP:-}"
SIM_TIME="${SIM_TIME:-450.0}"
RISK_THRESHOLD="${RISK_THRESHOLD:-0.30}"

# ── 构建命令 ─────────────────────────────────────────────────────────────────
CMD=(python3 run_module_ablation.py
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

if [[ -n "$SKIP" ]]; then
    CMD+=(--skip_conditions $SKIP)
fi

# ── 打印运行信息 ──────────────────────────────────────────────────────────────
echo "======================================================================"
echo " RTRA-LLM MODULE ABLATION"
echo "======================================================================"
echo " Provider    : $PROVIDER"
echo " Smoke       : $SMOKE"
echo " Cases       : ${CASES:-all 23}"
echo " Skip IDs    : ${SKIP:-none}"
echo " Sim time    : ${SIM_TIME}s"
echo ""
echo " 消融条件说明："
echo "   0  baseline          无 LLM（纯反应式避碰，参考基准）"
echo "   1  full              完整 RTRA-LLM"
echo "   2  wo_risk_trigger   无风险触发（每次高层决策更新均调用）"
echo "   3  wo_memory         无机动记忆"
echo "   4  wo_validator      无规则验证器"
echo "   5  wo_memory+val     同时关闭记忆+验证器"
echo "======================================================================"
echo " Command     : ${CMD[*]}"
echo "======================================================================"

# ── 执行 ─────────────────────────────────────────────────────────────────────
"${CMD[@]}"
