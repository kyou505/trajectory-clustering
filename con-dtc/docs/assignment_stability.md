# 分配决策稳定性

当前定义：当前领先簇的剩余优势，相对于历史上出现过的优势收缩，是否具有足够余量。

入口：`src.training.assignment_stability.compute_assignment_stability`。输入是时间递增的原始软分配 `[T, N, K]`，至少两次观测；同一样本使用固定输入，簇编号必须对应。输入不包含真实标签。计算在CPU双精度下进行，输出已停止梯度。

固定末次观测的领先簇 `a=argmax(q[-1])`，整个窗口中保持a不变：

\[
g_{ij}^{(r)}=\log(q_{ia}^{(r)}/q_{ij}^{(r)}),\qquad
u_{ij}=\max_r[g_{ij}^{(r-1)}-g_{ij}^{(r)}]_+,
\]

\[
R_i=\max_{j\ne a}\frac{u_{ij}}{g_{ij}^{(t)}+u_{ij}},\qquad S_i=1-R_i.
\]

- `stability`：S，越大表示当前优势相对于历史收缩越充足。
- `risk`：R，是排序分数，不是错误概率。
- `at_risk`：S≤0.5的布尔标记；没有分位数处理，也不固定选10%。
- `current_cluster`：当前领先簇。
- `competitor_cluster`：使相对风险最大的竞争簇，不一定是当前第二大概率簇。
- `current_margin`：当前top1/top2对数概率间隔。
- `competitor_margin`、`historical_contraction`、`remaining_margin`：上述最大风险竞争簇的当前优势、历史最大收缩和两者之差。

无收缩且存在优势时R=0；并列领先或对数优势≤1e-14时R=1。概率对数使用1e-12的数值下限。恒定但模糊、稍微领先的分配可能得到S=1，因为本定义描述历史收缩下的余量，不代替熵或置信度。

如果下一次对每个竞争簇的优势减少量都不超过对应历史u，且S>0.5，则领先簇保持不变。但历史u不是未来变化的严格上界。该定义既不能判断当前簇是否正确，也不能保证未来不换簇。

## 相邻两轮与三轮

两轮只观察一次收缩，三轮取两次收缩中的较大值。三轮不是数学要求，也未证明优于两轮。`AssignmentStabilityHistory(window_size=3)`提供滚动历史，允许设为2或更大的整数。

```python
from src.training.assignment_stability import AssignmentStabilityHistory

history = AssignmentStabilityHistory(window_size=3)
# 每轮结束，对同一份原始输入评估；不把第0轮初始化计入历史。
next_epoch_info = history.observe(
    q, epoch=epoch, sample_ids=sample_ids, input_sha256=input_sha256,
)
# 前两次返回None；第3轮末返回的指标只能用于第4轮。
if next_epoch_info is not None:
    score = next_epoch_info["stability"]
    mask = next_epoch_info["at_risk"]
```

历史容器检查连续轮次、样本ID/顺序、概率形状与输入哈希；无法识别未告知它的簇编号置换。调用方须保持中心编号连续，重新K-means后不能直接接续旧历史。`for_epoch(epoch)`只接受最后记录轮次的下一轮。

## 接入训练加权

另一种独立使用方式是按稳定性控制EMA动量，见[`assignment_decision_momentum`运行说明](../configs/assignment_decision_momentum/README.md)。该模式不与本节的损失加权或熵插值叠加。

也可以保持EMA动量固定，用稳定性控制当前目标与EMA目标的插值比例，见[`assignment_decision_target_mix`运行说明](../configs/assignment_decision_target_mix/README.md)。该模式关闭熵插值与损失加权，低稳定性样本采用更高的EMA目标占比。

在`target_history`中指定以下配置，以二值权重替换旧JS加权：

```yaml
weighting_signal: decision_stability
decision_window_size: 3
decision_unstable_weight: 1.5
minimum_weight: null
weight_warmup: false
```

稳定样本（S>0.5）权重为1，余量不足样本（S≤0.5）权重为1.5。1.5是初始实验设置，并非经验证的最优值；允许设为1作为等权对照。此模式不能同时设置旧`minimum_weight`或启用线性warm-up，避免两套加权混用。已有配置仍使用原加权逻辑。

EMA与熵插值继续构造训练目标；新权重只作用于跨视图DEC损失，不改变MSTM、实例对比等损失的样本权重。批次内仍按`sum(w_i * loss_i) / sum(w_i)`归一化。因此1和1.5描述的是样本间相对贡献，不意味着某个样本相对等权训练的绝对梯度恰好增加50%。

每轮结束以eval模式对固定原始输入计算q，恢复模型模式与随机状态；若已保存表示诊断则复用对应快照。默认三轮窗口在第3轮末首次就绪，第4轮使用第1～3轮的信息。权重在整轮内固定，最早启用轮次为`max(decision_window_size+1, target_history.start_epoch)`。第0轮初始化不计入窗口，历史不足时全体权重为1。仅启用训练配置即可自动采集所需历史，不依赖旧实验日志或真实标签。

每次运行保存：

- `assignment_history/epoch_*.pt`：轮末固定输入q、样本ID、输入哈希。
- `target_history/epoch_*.pt`：实际训练权重、`decision_stability`完整指标及来源轮次；旧JS字段仅作原有诊断。
- `training_history.json`与训练日志：是否启用、加权数量和平均权重。
- checkpoint中的`assignment_history_state_dict`：滚动历史缓存，供审计；当前训练入口没有新增断点恢复功能。

可直接运行的Porto、QD单seed配置位于`configs/assignment_decision_stability/`，沿用原DART-C的pre15、EMA和熵插值设置，结果写入`runs/assignment_decision_stability/`。例如在`con-dtc`目录运行：

```sh
python -m src.experiment.run --config configs/assignment_decision_stability/condtc_porto_decision_stability_w150_pre15.yaml
python -m src.experiment.run --config configs/assignment_decision_stability/condtc_qd_decision_stability_w150_pre15.yaml
```

风险识别结果不能直接推出加权的训练收益，需通过这些对照实验验证。

验证：`python3 -m unittest discover -s con-dtc/tests -p 'test_assignment_stability.py' -v`（仓库根目录运行）。

训练接入验证：`python3 -m unittest discover -s con-dtc/tests -p 'test_decision_weighting.py' -v`。涵盖阈值权重、真实跨视图DEC梯度与归一化、推理的随机状态保持、配置兼容，以及使用模拟轮次输出的完整训练调度与日志检查；不代替真实数据集上的性能实验。

已通过12项单元测试，并使用已有Porto、QD baseline快照复核第1～3、2～4、3～5轮三个窗口。共6组、116,727个样本窗口的风险分数与此前独立分析一致：最大绝对误差为3.42e-14，S≤0.5的分组无差异。复核记录见仓库根目录`output/assignment_decision_stability_20260923/implementation_verification.json`。这项核对验证实现一致性，不代表新增训练实验已完成。
