#!/usr/bin/env bash
# =============================================================================
# scripts/run_paper_experiments.sh  ——  RTRA-LLM 一键运行批量实验
# =============================================================================
# 按顺序运行多模型对比、模块消融和提示词消融，所有结果保存到同一个带时间戳的输出目录。
#
# 用法：
#   bash scripts/run_paper_experiments.sh [provider]
#
# 例子：
#   bash scripts/run_paper_experiments.sh deepseek        # 全部23场景
#   SMOKE=1 bash scripts/run_paper_experiments.sh deepseek  # 冒烟测试
#   PROVIDERS="deepseek zhipu" bash scripts/run_paper_experiments.sh  # 多模型对比使用多 provider
#
# 可通过环境变量定制：
#   PROVIDER      模块消融和提示词消融的主提供商（默认 deepseek）
#   PROVIDERS     多模型对比的提供商列表（默认 "$PROVIDER zhipu"）
#   SMOKE         1=冒烟测试模式
#   SIM_TIME      仿真时长（默认 450s）
#   CASES         场景子集（默认全部23个）
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

# ── 参数 ─────────────────────────────────────────────────────────────────────
PROVIDER="${1:-deepseek}"
PROVIDERS="${PROVIDERS:-$PROVIDER qwen minimax}"
SMOKE="${SMOKE:-0}"
SIM_TIME="${SIM_TIME:-450.0}"
CASES="${CASES:-}"
TS=$(date +"%Y%m%d_%H%M%S")
BASE_OUT="output/experiments/full_experiments_${PROVIDER}_${TS}"

# ── 打印总计划 ────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║          CORALL  批量实验  (多模型 + 消融 + 提示词)             ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║  Multi-model   : $PROVIDERS"
echo "║  Main provider : $PROVIDER"
echo "║  Smoke         : $SMOKE"
echo "║  Cases         : ${CASES:-all 23}"
echo "║  Sim time      : ${SIM_TIME}s"
echo "║  Output dir    : $BASE_OUT"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

mkdir -p "$BASE_OUT"

# ── 辅助函数 ─────────────────────────────────────────────────────────────────
run_step() {
    local STEP_NAME="$1"
    shift
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  ▶  $STEP_NAME"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    if "$@"; then
        echo "  ✓  $STEP_NAME 完成"
    else
        echo "  ✗  $STEP_NAME 失败（退出码 $?），继续执行后续表格..."
    fi
}

# ── 多模型对比 ───────────────────────────────────────────────────────────────
MULTI_MODEL_CMD=(python3 run_multi_model_comparison.py
    --providers $PROVIDERS
    --sim_time "$SIM_TIME"
    --output_dir "${BASE_OUT}/multi_model"
)
[[ "$SMOKE" == "1" ]] && MULTI_MODEL_CMD+=(--smoke)
[[ -n "$CASES" ]] && MULTI_MODEL_CMD+=(--cases $CASES)

run_step "多模型对比 (${PROVIDERS})" "${MULTI_MODEL_CMD[@]}"

# ── 模块消融 ─────────────────────────────────────────────────────────────────
MODULE_ABLATION_CMD=(python3 run_module_ablation.py
    --provider "$PROVIDER"
    --sim_time "$SIM_TIME"
    --output_dir "${BASE_OUT}/module_ablation"
)
[[ "$SMOKE" == "1" ]] && MODULE_ABLATION_CMD+=(--smoke)
[[ -n "$CASES" ]] && MODULE_ABLATION_CMD+=(--cases $CASES)

run_step "模块消融 (${PROVIDER})" "${MODULE_ABLATION_CMD[@]}"

# ── 提示词消融 ───────────────────────────────────────────────────────────────
PROMPT_ABLATION_CMD=(python3 run_prompt_ablation.py
    --provider "$PROVIDER"
    --sim_time "$SIM_TIME"
    --output_dir "${BASE_OUT}/prompt_ablation"
)
[[ "$SMOKE" == "1" ]] && PROMPT_ABLATION_CMD+=(--smoke)
[[ -n "$CASES" ]] && PROMPT_ABLATION_CMD+=(--cases $CASES)

run_step "提示词消融 (${PROVIDER})" "${PROMPT_ABLATION_CMD[@]}"

# ── 完成 ─────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  批量实验完成"
echo "║  输出目录: $BASE_OUT"
echo "║"
echo "║  文件结构："
echo "║    multi_model/      ── 多模型对比结果"
echo "║    module_ablation/  ── 模块消融结果"
echo "║    prompt_ablation/  ── 提示词消融结果"
echo "╚══════════════════════════════════════════════════════════════════╝"
