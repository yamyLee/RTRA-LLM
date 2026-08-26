#!/usr/bin/env bash
# =============================================================================
# scripts/run_threshold_sweep.sh  ——  RTRA-LLM 风险阈值灵敏度扫描实验
# =============================================================================
# 用法：
#   bash scripts/run_threshold_sweep.sh [provider]
#
# 例子：
#   bash scripts/run_threshold_sweep.sh deepseek          # 默认阈值序列
#   bash scripts/run_threshold_sweep.sh zhipu             # 使用 zhipu
#   SMOKE=1 bash scripts/run_threshold_sweep.sh deepseek  # 冒烟测试
#   THRESHOLDS="0.15 0.20 0.25 0.30" bash scripts/run_threshold_sweep.sh deepseek
#   PAPER_Q=0.30 bash scripts/run_threshold_sweep.sh deepseek
#   CASES="1 2 7 8" bash scripts/run_threshold_sweep.sh deepseek
#
# 默认阈值序列（DEFAULT_THRESHOLDS）：
#   0.05  0.10  0.15  0.20  0.25  0.30  0.35  0.40  0.50
#   ↑ paper_q (0.30) 会自动标注为论文方法值（★），elbow 点标注为（◆）
#
# 输出文件（保存在 --output_dir 下）：
#   sweep_q0p05.csv  ...  sweep_q0p50.csv   每个阈值的原始结果
#   sweep_all_raw.csv                        全部阈值合并原始数据
#   sweep_summary.csv                        各阈值汇总统计
#   sweep_results.json                       结构化 JSON（含 elbow 建议）
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

# ── 参数解析 ─────────────────────────────────────────────────────────────────
PROVIDER="${1:-deepseek}"
SMOKE="${SMOKE:-0}"
CASES="${CASES:-}"
THRESHOLDS="${THRESHOLDS:-}"
PAPER_Q="${PAPER_Q:-0.30}"
SIM_TIME="${SIM_TIME:-450.0}"

# ── 构建命令 ─────────────────────────────────────────────────────────────────
CMD=(python3 run_threshold_sweep.py
     --provider "$PROVIDER"
     --sim_time "$SIM_TIME"
     --paper_q "$PAPER_Q"
)

if [[ "$SMOKE" == "1" ]]; then
    CMD+=(--smoke)
    echo "[SMOKE] 冒烟测试模式：仅 case 1"
fi

if [[ -n "$CASES" ]]; then
    CMD+=(--cases $CASES)
fi

if [[ -n "$THRESHOLDS" ]]; then
    CMD+=(--thresholds $THRESHOLDS)
fi

# ── 打印运行信息 ──────────────────────────────────────────────────────────────
echo "======================================================================"
echo " CORALL THRESHOLD SWEEP — RISK TRIGGER SENSITIVITY"
echo "======================================================================"
echo " Provider    : $PROVIDER"
echo " Smoke       : $SMOKE"
echo " Paper q     : $PAPER_Q  (★ 论文方法值)"
echo " Thresholds  : ${THRESHOLDS:-0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.50}"
echo " Cases       : ${CASES:-all 23}"
echo " Sim time    : ${SIM_TIME}s"
echo ""
echo " 实验目的："
echo "   验证风险阈值 q 的选择合理性（效率-安全 Pareto 曲线）"
echo "   自动检测 elbow 点 q*：N_call 降幅 ≥ 70% 且 R_max 升幅 < 30%"
echo "   生成可直接写入论文的阈值分析文本（paper-ready advice）"
echo ""
echo " 输出指标（每个 q 值）："
echo "   N_call   ── LLM 调用次数（触发效率）"
echo "   R_max    ── 最大碰撞风险（安全指标）"
echo "   A_turn   ── 转向一致率 vs 基准（COLREGs 符合度）"
echo "   ΔD       ── 路径效率偏差 vs 基准（%）"
echo "======================================================================"
echo " Command     : ${CMD[*]}"
echo "======================================================================"

# ── 执行 ─────────────────────────────────────────────────────────────────────
"${CMD[@]}"
