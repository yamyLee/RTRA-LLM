#!/usr/bin/env bash
# =============================================================================
# scripts/run_paper_experiments.sh  ——  RTRA-LLM 一键运行批量实验
# =============================================================================
# 按论文实验设置运行多模型对比、模块消融、阈值扫描、固定间隔敏感性和提示词消融。
#
# 用法：
#   bash scripts/run_paper_experiments.sh [provider]
#
# 例子：
#   bash scripts/run_paper_experiments.sh deepseek        # 全部23场景
#   SMOKE=1 bash scripts/run_paper_experiments.sh deepseek  # 冒烟测试
#   PROVIDERS="qwen deepseek minimax" bash scripts/run_paper_experiments.sh
#
# 可通过环境变量定制：
#   PROVIDER      模块消融和提示词消融的主提供商（默认 qwen）
#   PROVIDERS     多模型对比的提供商列表（默认 "qwen deepseek minimax"）
#   SMOKE         1=冒烟测试模式
#   SIM_TIME      仿真时长（默认 450s）
#   CASES         场景子集（默认全部23个）
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

# ── 参数 ─────────────────────────────────────────────────────────────────────
PROVIDER="${1:-qwen}"
PROVIDERS="${PROVIDERS:-qwen deepseek minimax}"
SMOKE="${SMOKE:-0}"
SIM_TIME="${SIM_TIME:-450.0}"
CASES="${CASES:-}"
TS=$(date +"%Y%m%d_%H%M%S")
BASE_OUT="output/experiments/full_experiments_${PROVIDER}_${TS}"
export QWEN_MODEL="qwen3.5-plus"
export DEEPSEEK_MODEL="deepseek-v3.2"
export MINIMAX_MODEL="MiniMax-M2.5"

# ── 打印总计划 ────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                   RTRA-LLM 论文实验                            ║"
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
    --include_llm_baseline
    --include_rule_trigger
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

THRESHOLD_CMD=(python3 run_threshold_sweep.py
    --provider "$PROVIDER"
    --sim_time "$SIM_TIME"
    --output_dir "${BASE_OUT}/threshold_sweep"
)
[[ "$SMOKE" == "1" ]] && THRESHOLD_CMD+=(--smoke)
[[ -n "$CASES" ]] && THRESHOLD_CMD+=(--cases $CASES)

run_step "风险阈值敏感性 (${PROVIDER})" "${THRESHOLD_CMD[@]}"

FIXED_INTERVAL_CMD=(python3 run_fixed_interval_sensitivity.py
    --provider "$PROVIDER"
    --configuration legacy
    --sim_time "$SIM_TIME"
    --output_dir "${BASE_OUT}/fixed_interval"
)
[[ "$SMOKE" == "1" ]] && FIXED_INTERVAL_CMD+=(--smoke)
[[ -n "$CASES" ]] && FIXED_INTERVAL_CMD+=(--cases $CASES)

run_step "固定间隔敏感性 (${PROVIDER})" "${FIXED_INTERVAL_CMD[@]}"

FIXED_MATCHED_CMD=(python3 run_fixed_interval_sensitivity.py
    --provider "$PROVIDER"
    --configuration matched
    --intervals 500
    --sim_time "$SIM_TIME"
    --output_dir "${BASE_OUT}/fixed_interval_matched"
)
[[ "$SMOKE" == "1" ]] && FIXED_MATCHED_CMD+=(--smoke)
[[ -n "$CASES" ]] && FIXED_MATCHED_CMD+=(--cases $CASES)

run_step "等调用预算固定间隔对照 (${PROVIDER})" "${FIXED_MATCHED_CMD[@]}"

# ── 提示词消融 ───────────────────────────────────────────────────────────────
PROMPT_ABLATION_CMD=(python3 run_prompt_ablation.py
    --provider "$PROVIDER"
    --sim_time "$SIM_TIME"
    --output_dir "${BASE_OUT}/prompt_ablation"
)
[[ "$SMOKE" == "1" ]] && PROMPT_ABLATION_CMD+=(--smoke)
[[ -n "$CASES" ]] && PROMPT_ABLATION_CMD+=(--cases $CASES)

run_step "提示词消融 (${PROVIDER})" "${PROMPT_ABLATION_CMD[@]}"

if [[ -f "${BASE_OUT}/multi_model/multi_model_all_raw.csv" && \
      -f "${BASE_OUT}/fixed_interval/raw_results.csv" && \
      -f "${BASE_OUT}/fixed_interval_matched/raw_results.csv" ]]; then
    run_step "场景级配对统计" python3 scripts/analyze_paper_statistics.py \
        --input "${BASE_OUT}/multi_model/multi_model_all_raw.csv" \
                "${BASE_OUT}/fixed_interval/raw_results.csv" \
                "${BASE_OUT}/fixed_interval_matched/raw_results.csv" \
        --output-dir "${BASE_OUT}/statistics" \
        --method-column comparison_method \
        --reference RTRA-LLM \
        --comparators "Rule-trigger baseline" Fixed-LLM-Matched Fixed-LLM-250 \
        --metrics R_avg
fi

# ── 完成 ─────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  批量实验完成"
echo "║  输出目录: $BASE_OUT"
echo "║"
echo "║  文件结构："
echo "║    multi_model/      ── 多模型对比结果"
echo "║    module_ablation/  ── 模块消融结果"
echo "║    threshold_sweep/  ── 风险阈值敏感性"
echo "║    fixed_interval/   ── 固定间隔敏感性"
echo "║    fixed_interval_matched/ ── 500步等模块配置对照"
echo "║    prompt_ablation/  ── 提示词消融结果"
echo "║    statistics/       ── 配对检验与Bootstrap结果"
echo "╚══════════════════════════════════════════════════════════════════╝"
