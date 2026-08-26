# 如何查看仿真结果

## 1. 运行时输出

当你运行仿真时，终端会显示：

### 基本信息示例
```
=== Starting Simulation ===
Case: 1
Duration: 450 seconds
LLM enabled: Yes
LLM provider: zhipu
Animation: Yes
--------------------------------------------------
```

### 进度信息（每5秒）
```
Time:   0.0s | Avg Risk: 0.123 | Kdir: 0.0
Time:   5.0s | Avg Risk: 0.456 | Kdir: 0.0
Time:  10.0s | Avg Risk: 0.789 | Kdir: 1.0
```

### LLM决策信息
```
[LLM Decision at t=10.0s]
[Claude] Rule 14 (head-on situation), Action: [Give-way, turn to starboard], Explanation: ...
Action: Turn STARBOARD
```

### 汇总信息
```
=== Simulation Summary ===
Case: 1
Duration: 450 seconds
Total simulation steps: 4500
Minimum DCPA: 1.23 nautical miles
Maximum Risk: 0.890

Maneuver Statistics:
  - Starboard turns: 5
  - Port turns: 2
  - Stand on: 4493

Output files generated:
  - Animation: img/scenario_animation1.gif
  - Plots: img/plot_dcpa_tcpa_risk_1.png
==================================================
```

## 2. 生成的文件

### 动画文件
- **文件名**: `img/scenario_animation{case_number}.gif`
- **内容**: 显示整个仿真过程中所有船舶的运动轨迹
- **查看方法**:
  ```bash
  # 查看文件列表
  ls img/

  # 用浏览器打开
  xdg-open img/scenario_animation1.gif
  # 或用其他图片查看器
  feh img/scenario_animation1.gif
  ```

### 数据图表
- **文件名**: `img/plot_dcpa_tcpa_risk_{case_number}.png`
- **内容**:
  - 左上角：DCPA（最近接近距离）随时间变化
  - 右上角：距离（Distance）随时间变化
  - 左下角：TCPA（到达最近接近点的时间）随时间变化
  - 右下角：风险值（Risk）随时间变化

## 3. 理解输出指标

### 重要参数说明
- **DCPA** (Distance at Closest Point of Approach): 最近接近距离，单位海里
  - > 2 nmi：安全
  - 1-2 nmi：注意
  - < 1 nmi：危险

- **TCPA** (Time to Closest Point of Approach): 到达最近接近点的时间，秒
  - > 300s：充足时间
  - 60-300s：需要关注
  - < 60s：紧急情况

- **Risk**: 风险值 (0-1)
  - 0-0.3：低风险
  - 0.3-0.6：中等风险
  - 0.6-1.0：高风险

- **Kdir**: 转向指令
  - 0：保持航向（Stand on）
  - +1：向右转向（Starboard）
  - -1：向左转向（Port）

## 4. 案例分析

### Case 1: 对遇情况
- 初始两个船舶相向而行
- LLM 应该建议在合适时机转向
- 动画显示两个船舶的避让机动

### 查看步骤
1. 先看动画文件，了解整体运动情况
2. 查看终端输出的汇总统计
3. 分析风险值变化，判断避让是否及时
4. 如果启用了LLM，查看决策是否合理

## 5. 常用查看命令

```bash
# 进入目录
cd /path/to/your/project

# 查看所有生成文件
ls img/

# 查看实时输出（带进度条）
python main.py --case_number 1 --sim_time 450 --llm 1 | grep -E "(Time|LLM Decision|Summary)"

# 不显示动画，只生成文件
python main.py --no_animation --case_number 1

# 保存输出到文件
python main.py --case_number 1 --sim_time 450 > output.log
cat output.log
```