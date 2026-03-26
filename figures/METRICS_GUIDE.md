# WandB 指标说明文档

本文档说明 verl 框架训练过程中记录的所有重要 wandb 指标的含义、计算方式和论文中的用途。

数据文件：`figures/all_runs_metrics.json`（~20MB，包含 5 个 run × 280+ 指标 × 1000+ steps）

---

## 1. Actor 指标 (`actor/*`)

训练策略模型（Actor）的梯度更新相关指标，每个 training step 记录一次。

| 指标 | 含义 | 论文用途 |
|------|------|---------|
| `actor/entropy` | 策略的 token 级别熵（Shannon entropy），衡量输出分布的多样性 | 监控探索能力，熵下降过快说明模型坍缩 |
| `actor/grad_norm` | 梯度范数（L2），反映更新幅度 | 训练稳定性诊断 |
| `actor/lr` | 当前学习率 | 确认 LR scheduler 正常工作 |
| `actor/pg_loss` | Policy Gradient 损失，PPO/GRPO 的主损失项 | 训练收敛曲线 |
| `actor/pg_clipfrac` | PPO clip 触发比例（advantage > 0 时 ratio 超过 1+ε） | clip 比例过高说明更新步长过大 |
| `actor/pg_clipfrac_lower` | PPO lower clip 触发比例（advantage < 0 时 ratio 低于 1-ε） | 同上 |
| `actor/ppo_kl` | 当前策略与参考策略之间的 KL 散度 | 衡量策略偏移程度，KL 暴涨可能导致崩溃 |

**代码位置**：`ray_trainer.py` → `_update_actor()` 返回的 `meta_info["metrics"]`

---

## 2. Critic / Reward 指标 (`critic/*`)

评价函数和奖励信号的统计信息。verl 中 GRPO 没有独立的 critic 网络，这里的 "critic" 实际是奖励信号的统计。

### 2.1 Score（原始奖励分数）

| 指标 | 含义 |
|------|------|
| `critic/score/mean` | **训练 batch 的平均奖励**，最重要的训练信号指标 |
| `critic/score/max` | batch 内最高奖励 |
| `critic/score/min` | batch 内最低奖励 |

**计算方式**：对每个 response，将 token 级别的 score 求和得到 sequence-level score。ORM 下为 0 或 1（二值），PRM 下为连续值。

**论文用途**：Fig 2(a) Training Reward 即 `critic/score/mean`。PRM reward hacking 的核心证据——reward 暴涨但 accuracy 崩溃。

### 2.2 Rewards（含 KL penalty 的奖励）

| 指标 | 含义 |
|------|------|
| `critic/rewards/mean` | 加了 KL penalty 后的平均奖励 |
| `critic/rewards/max` | 同上，最大值 |
| `critic/rewards/min` | 同上，最小值 |

**注意**：当 `kl_penalty_coeff=0`（默认）时，`rewards` 与 `score` 完全相同。

### 2.3 Advantages（优势值）

| 指标 | 含义 |
|------|------|
| `critic/advantages/mean` | token 级别优势值的均值（GRPO 中已归一化，接近 0） |
| `critic/advantages/std` | 优势值标准差，反映 reward 信号强度 |
| `critic/advantages/median` | 中位数（GRPO 中通常为 0，因为错误 response 的 advantage 为 0） |
| `critic/advantages/p10` | 第 10 百分位 |
| `critic/advantages/p90` | 第 90 百分位 |
| `critic/advantages/max` | 最大优势值 |
| `critic/advantages/min` | 最小优势值 |

**计算方式**（GRPO）：同一 prompt 的 N 个 response 的 reward 做 group-level 归一化（减均值除标准差），错误 response 得到负 advantage，正确 response 得到正 advantage。

### 2.4 Returns

| 指标 | 含义 |
|------|------|
| `critic/returns/mean` | token 级别 return 的均值 |
| `critic/returns/max` | return 最大值 |
| `critic/returns/min` | return 最小值 |

**注意**：GRPO 中 returns = advantages（无 value baseline），两者数值相同。

**代码位置**：`metric_utils.py` → `compute_data_metrics()`

---

## 3. 优势分布指标 (`adv_distribution/*`)

按正确/错误分组的 **sample 级别**优势分布统计，用于诊断 RL 信号质量。

| 指标 | 含义 |
|------|------|
| `adv_distribution/correct_count` | 当前 batch 中正确 response 数量（score >= 0.5） |
| `adv_distribution/correct_mean` | 正确 response 的平均优势值（应为正） |
| `adv_distribution/correct_std` | 正确 response 优势值的标准差 |
| `adv_distribution/correct_min` | 正确 response 中最小优势值（GRPO 下通常为 0） |
| `adv_distribution/correct_max` | 正确 response 中最大优势值 |
| `adv_distribution/wrong_count` | 错误 response 数量 |
| `adv_distribution/wrong_mean` | 错误 response 的平均优势值（应为负） |
| `adv_distribution/wrong_std` | 错误 response 优势值的标准差 |
| `adv_distribution/wrong_min` | 错误 response 中最小优势值（绝对值最大的负值） |
| `adv_distribution/wrong_max` | 错误 response 中最大优势值（GRPO 下通常为 0） |
| `adv_distribution/positive_ratio` | 正优势样本比例（被鼓励的 response） |
| `adv_distribution/negative_ratio` | 负优势样本比例（被惩罚的 response） |
| `adv_distribution/zero_ratio` | 零优势样本比例（无梯度信号的 response） |

**sample-level advantage 计算**：`sample_adv = sum(advantages * response_mask) / sum(response_mask)`

**论文用途**：
- `correct_count / (correct_count + wrong_count)` = 训练准确率
- `zero_ratio` 过高说明 reward 信号太稀疏（所有 response 同分，advantage 全为 0）
- `positive_ratio` vs `negative_ratio` 反映训练的有效梯度比例

**代码位置**：`metric_utils.py` → `compute_data_metrics()` L216-243

---

## 4. Response / Prompt 长度指标

### 4.1 Response Length (`response_length/*`)

| 指标 | 含义 |
|------|------|
| `response_length/mean` | **平均 response 长度**（token 数） |
| `response_length/max` | 最长 response |
| `response_length/min` | 最短 response |
| `response_length/clip_ratio` | 达到 `max_response_length` 上限的比例（被截断） |

**论文用途**：Fig 2(b) Response Length 即 `response_length/mean`。PRM reward hacking 时 response 暴涨到 7000+ token。

### 4.2 Non-aborted Response Length (`response_length_non_aborted/*`)

排除 aborted sample（response_length=0）后的统计，字段名同上。

### 4.3 Prompt Length (`prompt_length/*`)

| 指标 | 含义 |
|------|------|
| `prompt_length/mean` | 平均 prompt 长度 |
| `prompt_length/max` | 最长 prompt |
| `prompt_length/min` | 最短 prompt |
| `prompt_length/clip_ratio` | 达到上限的比例 |

### 4.4 Aborted Ratio (`response/*`)

| 指标 | 含义 |
|------|------|
| `response/aborted_ratio` | 生成被中止的比例（response_length=0） |

---

## 5. 验证集指标 (`val-core/*`, `val-aux/*`)

每 `test_freq` 步评估一次，对每个 prompt 生成 N=8 个 response。

### 5.1 指标命名规则

```
val-{core|aux}/{data_source}/{metric_type}/{aggregation}
```

- **`val-core`**：核心指标（论文主表），只包含 `mean@N_max` 和 `best@N_max`
- **`val-aux`**：辅助指标（补充分析），包含 std、worst、smaller N 等

### 5.2 数据源 (`data_source`)

| data_source | 数据集 | 样本数 |
|-------------|--------|--------|
| `math__math` | MATH-500 | 500 |
| `olympiad` | OlympiadBench (text-only) | 674 |
| `aime2026` | AIME 2026 | 30 |
| `gpqa_diamond` | GPQA-Diamond (STEM 单选) | 198 |
| `codegen__humaneval` | HumanEval (代码) | 164 |
| `logic__zebra_puzzle_dataset` | Zebra Logic Puzzle | 200 |

### 5.3 指标类型 (`metric_type`)

| metric_type | 含义 |
|-------------|------|
| `acc` | 准确率（rule-based 判断，0 或 1） |
| `reward` | 奖励值（等于 acc，因为 ORM reward = accuracy） |
| `score` | 同 reward |
| `llm_corrected` | LLM 判分修正量（LLM 判对但 rule 判错的比例） |

### 5.4 聚合方式 (`aggregation`)

每个 prompt 生成 N=8 个 response，聚合方式：

| aggregation | 含义 | 说明 |
|-------------|------|------|
| `mean@4` | **N=4 时的平均准确率** | 随机采样 4 个 response 的平均正确率（bootstrap 1000 次取均值） |
| `std@4` | N=4 时准确率的标准差 | 衡量样本间方差 |
| `best@4/mean` | **N=4 时的 best-of-4 准确率** | 4 个 response 中最好的一个的期望准确率（≈ pass@4） |
| `best@4/std` | best-of-4 的标准差 | |
| `best@2/mean` | N=2 时的 best-of-2 | |
| `best@2/std` | | |
| `worst@4/mean` | N=4 时的 worst-of-4 | 4 个中最差的，衡量鲁棒性 |
| `worst@4/std` | | |
| `worst@2/mean` | N=2 时的 worst-of-2 | |
| `worst@2/std` | | |

**关键理解**：
- `mean@4` ≈ `avg@4`：平均水平，论文中用此作主要对比指标
- `best@4/mean` ≈ `pass@4`：至少有一个对的概率，反映模型覆盖能力
- `worst@4/mean`：所有 response 都对的概率，反映模型一致性

**论文中的使用**：
- 主图/主表：`val-core/olympiad/acc/mean@4`（OlympiadBench avg@4）
- 补充：`val-core/math__math/acc/mean@4`（MATH-500 avg@4）

**代码位置**：`metric_utils.py` → `process_validation_metrics()` L544-707，bootstrap 采样 1000 次

---

## 6. Dual-Objective 指标 (`dual_objective/*`)

PA-GRPO 独有指标，记录双目标优势分解的详细信息。

### 6.1 优势分解

PA-GRPO 的总优势：`A_total = A_out + λ × A_proc`

| 指标 | 含义 |
|------|------|
| `dual_objective/outcome_mean` | A_out（ORM 优势）的均值 |
| `dual_objective/outcome_std` | A_out 的标准差 |
| `dual_objective/process_mean` | A_proc（PRM 优势）的均值（仅在正确 response 中非零） |
| `dual_objective/process_std` | A_proc 的标准差 |
| `dual_objective/lambda_process` | λ 参数值 |
| `dual_objective/process_active_ratio` | 有 ≥2 个正确 response 的 prompt 比例（仅这些 prompt 的 A_proc 非零） |
| `dual_objective/total_advantage_mean` | A_total 的均值 |
| `dual_objective/total_advantage_std` | A_total 的标准差 |

### 6.2 按正确性分组

| 指标 | 含义 |
|------|------|
| `dual_objective/correct_adv_mean` | 正确 response 的 A_total 均值 |
| `dual_objective/correct_adv_min` | 正确 response 的 A_total 最小值 |
| `dual_objective/correct_adv_max` | 正确 response 的 A_total 最大值 |
| `dual_objective/correct_Aout_mean` | 正确 response 中 A_out 的均值 |
| `dual_objective/correct_Aproc_mean` | 正确 response 中 A_proc 的均值 |
| `dual_objective/wrong_adv_mean` | 错误 response 的 A_total 均值 |
| `dual_objective/wrong_adv_min` | 错误 response 的 A_total 最小值 |

**论文用途**：
- `process_active_ratio`：说明 PRM 信号的覆盖率
- `correct_Aproc_mean`：说明 PRM 是否在正确 response 间产生了有效区分
- `outcome_mean` vs `process_mean`：两个目标的相对强度

**代码位置**：`core_algos.py` → `compute_grpo_dual_objective_advantage()` L334-463

---

## 7. Fullnorm 指标 (`dual_fullnorm/*`)

Fullnorm ablation 独有，字段含义与 `dual_objective/*` 相同，区别在于：
- **PA-GRPO**：A_proc 仅在正确 response 间归一化
- **Fullnorm**：A_proc 在所有 response（含错误）间归一化

额外指标：`dual_fullnorm/wrong_Aproc_mean`——错误 response 的 A_proc 均值（PA-GRPO 中此值恒为 0）。

**代码位置**：`core_algos.py` → `compute_grpo_dual_fullnorm_advantage()` L468-578

---

## 8. 性能指标 (`perf/*`)

| 指标 | 含义 |
|------|------|
| `perf/throughput` | 吞吐量（tokens/sec/gpu） |
| `perf/time_per_step` | 每步训练时间（秒） |
| `perf/total_num_tokens` | 当前 batch 总 token 数 |
| `perf/max_memory_allocated_gb` | GPU 显存峰值使用（GB） |
| `perf/max_memory_reserved_gb` | GPU 显存峰值预留（GB） |
| `perf/cpu_memory_used_gb` | CPU 内存使用（GB） |
| `perf/mfu/actor` | Model FLOPs Utilization（MFU） |
| `perf/mfu/actor_infer` | 推理阶段 MFU |

---

## 9. 训练进度 (`training/*`)

| 指标 | 含义 |
|------|------|
| `training/global_step` | 当前训练步数 |
| `training/epoch` | 当前 epoch 数 |

---

## 数据文件结构

```json
{
  "ORM": {
    "0": {
      "actor/entropy": 5.123,
      "critic/score/mean": 0.234,
      "val-core/olympiad/acc/mean@4": 0.338,
      ...
    },
    "10": { ... },
    ...
  },
  "PRM": { ... },
  "PA-GRPO": { ... },
  "Fullnorm": { ... },
  "Mult": { ... }
}
```

- 键为 step（字符串），值为该 step 所有记录的指标
- 训练指标（actor/critic/adv_distribution/response_length 等）每步记录
- 验证指标（val-core/val-aux）每 `test_freq=10` 步记录
- 某些 step 可能缺少部分指标（如 resume 起始点）
