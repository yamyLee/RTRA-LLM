# 命令行参数说明

## **基本参数**

### `--case_number` (整数)
- **默认值**: 1
- **说明**: 选择要运行的仿真案例编号
- **取值范围**: 1-23（共23个预定义的海上会遇场景）
- **示例**:
  ```bash
  # 运行案例1（对遇情况）
  python main.py --case_number 1

  # 运行案例8（交叉相遇）
  python main.py --case_number 8
  ```

### `--sim_time` (浮点数)
- **默认值**: 450.0
- **说明**: 仿真持续时间，单位秒
- **取值范围**: 正数
- **示例**:
  ```bash
  # 运行5分钟
  python main.py --sim_time 300

  # 运行10分钟
  python main.py --sim_time 600
  ```

### `--dt` (浮点数)
- **默认值**: 0.1
- **说明**: 时间步长，单位秒
- **说明**: 仿真精度，值越小越精确但计算越慢
- **示例**:
  ```bash
  # 默认步长
  python main.py

  # 更精细的仿真（步长0.05秒）
  python main.py --dt 0.05
  ```

## **输出控制参数**

### `--no_animation` (布尔标志)
- **默认值**: False（不使用此参数）
- **说明**: 禁用动画生成，只生成数据图表
- **用途**: 当只需要数据分析而不需要可视化时使用
- **示例**:
  ```bash
  # 不生成动画，只生成图表
  python main.py --no_animation
  ```

### `--output_dir` (字符串)
- **默认值**: 'img/'
- **说明**: 输出文件保存目录
- **用途**: 指定结果文件的保存位置
- **示例**:
  ```bash
  # 保存到results目录
  python main.py --output_dir results/

  # 保存到自定义路径
  python main.py --output_dir /tmp/sim_results/
  ```

## **LLM 参数**

### `--llm` (整数)
- **默认值**: 0
- **说明**: 是否使用LLM进行决策
  - 0: 不使用LLM（使用传统的避让规则）
  - 1: 使用LLM进行智能避让决策
- **示例**:
  ```bash
  # 使用LLM决策
  python main.py --llm 1

  # 不使用LLM（传统方法）
  python main.py --llm 0
  ```

### `--llm_provider` (字符串)
- **默认值**: None（从环境变量LLM读取）
- **说明**: 指定LLM服务提供商
- **可选值**: openai, claude, zhipu
- **示例**:
  ```bash
  # 使用智谱AI
  python main.py --llm 1 --llm_provider zhipu

  # 使用OpenAI
  python main.py --llm 1 --llm_provider openai

  # 使用Claude
  python main.py --llm 1 --llm_provider claude
  ```

## **高级参数**

### `--compare` (布尔标志)
- **默认值**: False
- **说明**: 运行LLM与传统方法的对比仿真
- **用途**: 比较使用LLM和传统避让规则的效果差异
- **示例**:
  ```bash
  # 运行对比测试
  python main.py --compare --case_number 1
  ```

## **常用命令示例**

### 1. **基本运行**
```bash
# 使用默认参数（案例1，450秒，开启动画）
python main.py
```

### 2. **快速测试**
```bash
# 60秒快速测试，不生成动画
python main.py --sim_time 60 --no_animation
```

### 3. **使用LLM进行决策**
```bash
# 使用智谱AI进行避让决策
python main.py --llm 1 --llm_provider zhipu --case_number 1
```

### 4. **不同场景测试**
```bash
# 测试交叉相遇场景
python main.py --case_number 8 --sim_time 600

# 测试追越场景
python main.py --case_number 12 --sim_time 800
```

### 5. **完整测试**
```bash
# 完整测试：LLM决策+动画+输出到指定目录
python main.py --llm 1 --llm_provider zhipu \
               --case_number 1 \
               --sim_time 450 \
               --output_dir test_results/
```

### 6. **对比测试**
```bash
# 比较LLM和传统方法的效果
python main.py --compare --llm 1 --llm_provider zhipu --case_number 1
```

## **案例说明**

### Case 1: 对遇情况
- 两艘船舶相向而行
- 需要在合适时机转向避让

### Case 8: 交叉相遇
- 两艘船舶交叉航行
- 需要判断谁给谁让路

### Case 12: 追越情况
- 一艘船舶追越另一艘
- 追越船需要给被追越船让路

### 其他案例
- 包含各种复杂海况：多船相遇、能见度限制、速度限制等

## **环境变量配置**

部分参数可以通过环境变量设置（在`.env`文件中）：
```bash
# 默认LLM提供商
LLM_PROVIDER=zhipu

# 模型参数
OPENAI_MODEL=gpt-4
ZHIPU_MODEL=glm-4
CLAUDE_MODEL=claude-sonnet-4-20250514

# 其他参数
OPENAI_TEMPERATURE=0.1
ZHIPU_TEMPERATURE=0.1
```