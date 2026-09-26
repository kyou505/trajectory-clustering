# 簇历史与实例历史的表示稳定性诊断

用于检查固定原始轨迹的相邻轮表示变化，及其与聚类正确率、正确分配退化的关系。旧版直接历史对比实验包含每个数据集四种配置，共8份；本次另增加2份 LFSS 表示约束配置，见下一节。所有配置均沿用 pre15 设置、联合训练6轮，开启 `training.save_representation_history`，并在每个数据集内共享相同的 seed 与预训练权重。

## LFSS 表示约束版本

新增 `condtc_porto_dart_full_lfss_pre15.yaml` 与 `condtc_qd_dart_full_lfss_pre15.yaml`，
保留对应 DART-C 的簇分配历史设置，增加 LS=1.0、LI=0.1 的表示约束。
表示 EMA 动量为0.996，从第1轮开始逐步更新；前代网络为上一轮末 EMA 目标快照。
LS 使用跨视图自蒸馏，LI 使用同视图跨轮实例对比，不接入 LC。
这些新配置与下文旧版 Direct History 分开命名，不能视为同一种实例损失。

新配置评估 `checkpoint_selection: last`，对应固定6轮训练；旧配置默认仍按基础损失选择 best。
比较时需确认 checkpoint 轮次一致，或统一选择 last。
启用 LFSS 后固定输入快照增加 `projected_embeddings`、`projection_norms`、
`projection_cosine_previous`、`projection_cosine_valid`，分别诊断编码器 z 和投影 h。
运行方式与完整参数说明见 [主 README](../../README.md#dart-c-与-lfss-表示约束)。

## 旧版直接历史对比配置

DART-C 仅使用簇分配历史；DART-Full 在相同的 C 设置上增加历史实例对比。本轮选择直接冻结上一轮在线编码器的版本，未开启编码器 EMA。目录名保留，已完成的 Baseline 运行无需重跑。

| 数据集 | seed | 簇数 | 预训练权重 |
|---|---:|---:|---|
| QD | 19721013 | 12 | sttraj2vec_pretrain_paper_15ep.pt |
| Porto | 42 | 27 | sttraj2vec_porto_finegrained_pre15.pt |

在 `con-dtc` 目录运行：

```bash
python -m src.experiment.run --config configs/baseline_representation_stability/condtc_qd_baseline_pre15.yaml
python -m src.experiment.run --config configs/baseline_representation_stability/condtc_porto_finegrained_baseline_pre15.yaml
```

结果位于 `runs/baseline_representation_stability/<experiment>/<timestamp>_seed<seed>/`，每次运行独立目录。

## 新增 DART 对照配置

| 数据集 | 方法 | 配置文件 | 历史实例 λ |
|---|---|---|---:|
| Porto | DART-C | `condtc_porto_dart_c_pre15.yaml` | 0 |
| Porto | DART-Full | `condtc_porto_dart_full_history_direct_s3_l010_pre15.yaml` | 0.10 |
| QD | DART-C | `condtc_qd_dart_c_pre15.yaml` | 0 |
| QD | DART-Full | `condtc_qd_dart_full_history_direct_s3_l020_pre15.yaml` | 0.20 |

四组的簇历史均从第3轮引入，动量0.7，熵分位数0.7、历史插值参数0.25、反向加权参数1.5。Full 的实例对比从第3轮开启，温度0.5，`history_encoder_ema_momentum: null`。实例损失仍使用统一样本权重，不继承 DART 的聚类样本权重。

参数沿用此前 direct/pre15 的 Porto λ=0.10 和 QD λ=0.20 配置，用于复现已有方法并补充诊断；这不是新增的参数搜索，也不表示这组参数对所有评价指标或缺失率都最优。每个数据集的 C / Full 配置只有历史实例损失权重不同（实验名除外）。

在 `con-dtc` 目录按需分别执行：

```bash
python -m src.experiment.run --config configs/baseline_representation_stability/condtc_porto_dart_c_pre15.yaml
python -m src.experiment.run --config configs/baseline_representation_stability/condtc_porto_dart_full_history_direct_s3_l010_pre15.yaml
python -m src.experiment.run --config configs/baseline_representation_stability/condtc_qd_dart_c_pre15.yaml
python -m src.experiment.run --config configs/baseline_representation_stability/condtc_qd_dart_full_history_direct_s3_l020_pre15.yaml
```

本配置只训练并记录，不自动执行缺失评估。当前记录覆盖固定原始输入的跨轮表示与簇分配，不包含固定增强双视图的表示；分析跨视图一致性还需另补固定视图诊断，不能用训练时随机视图的记录直接代替。

## Baseline＋历史实例对照

这两份配置用于检查历史实例约束单独加入 Baseline 时的作用，补全两种历史机制的四组消融。相比对应 DART-Full，仅将 `target_history` 设为 `null`（实验名另设）：关闭簇历史 EMA、熵插值与反向加权，保留 Baseline 原有的聚类损失。实例历史的启用轮次、权重、温度、直接快照方式以及全部基础训练参数与 Full 一致。

| 数据集 | 配置文件 | seed | 实例历史 λ |
|---|---|---:|---:|
| Porto | `condtc_porto_baseline_history_direct_s3_l010_pre15.yaml` | 42 | 0.10 |
| QD | `condtc_qd_baseline_history_direct_s3_l020_pre15.yaml` | 19721013 | 0.20 |

在 `con-dtc` 目录分别执行：

```bash
python -m src.experiment.run --config configs/baseline_representation_stability/condtc_porto_baseline_history_direct_s3_l010_pre15.yaml
python -m src.experiment.run --config configs/baseline_representation_stability/condtc_qd_baseline_history_direct_s3_l020_pre15.yaml
```

结果同样写入 `runs/baseline_representation_stability`，以各自实验名建立独立运行目录。已完成的另外三种方法无需重跑。

| 方法 | 簇分配历史 | 实例历史 |
|---|---|---|
| Baseline | 关闭 | 关闭 |
| DART-C | 开启 | 关闭 |
| Baseline＋历史实例 | 关闭 | 开启 |
| DART-Full | 开启 | 开启 |

比较“Baseline＋历史实例 − Baseline”和“Full − C”，可以检查实例历史的作用是否随簇历史的启用而变化。单 seed 的性能或表示变化差异仅作当前运行的诊断，不直接证明两项损失存在梯度冲突。

## 新增记录

`representation_history/epoch_000.pt` 为 K-means 初始化后、联合训练前的快照；`epoch_001.pt` 至 `epoch_006.pt` 为各轮训练结束时快照。记录时不重新执行 K-means，不更新参数，使用 `model.eval()` 与无梯度前向。真实标签只保存供离线诊断，不构造训练监督。

| 字段 | 含义 |
|---|---|
| `sample_ids` | 固定原始数据集行索引，完整覆盖0至N-1 |
| `labels`、`lengths` | 真实标签与有效轨迹点数 |
| `embeddings` | 在线编码器对未增强原始输入的表示，未经归一化 |
| `q`、`predicted_clusters` | 同一时刻学习中心给出的软分配及原始簇编号 |
| `cluster_centers` | 同一时刻的可学习簇中心 |
| `input_sha256`、`input_field_sha256` | 对模型实际输入token和mask的校验；跨轮不一致会报错 |
| `embedding_norms` | 表示范数，用于识别零向量和辅助分析 |
| `representation_cosine_previous` | 第1轮起，同一样本与上一轮表示的余弦相似度；不是EMA目标稳定性 |
| `representation_cosine_valid` | 非零向量标记，零向量相似度为NaN，离线分组时排除 |
| `phase`、`previous_epoch` | 记录时刻和上一份快照轮次 |
| `temporal_modes` | 数据表存在该字段时记录原始/时移/变速类型，不推测QD类型 |

`summary.json` 保存每轮有效样本数、输入校验值及相邻轮余弦相似度均值、10%分位数、中位数。它只是诊断摘要，不代表已经验证稳定性与准确率相关。

开启记录会增加每轮一次全量原始输入前向和CPU文件写入。记录前后恢复torch CPU/CUDA、Python、NumPy随机状态及各子模块的训练模式；正常训练增强、目标和损失不变。即使冒烟测试设置了 `max_initialization_batches` 或 `max_train_batches`，诊断仍遍历完整原始数据，避免不同轮次记录不同子集。默认关闭时不创建表示记录目录。

## 后续分析口径

1. 按相邻轮表示余弦相似度由低到高分成10个等量组，统计当前轮正确率。
2. 在上一轮正确的样本中按稳定性分组，统计正确→错误率。
3. 根据跨轮稳定性排名，检查低稳定性群体是否反复出现。

`predicted_clusters` 未做类别映射。分析正确性前须处理簇编号置换，并对固定映射与逐轮最优映射做敏感性检查，尤其是QD pre15早期轮次。第0轮是初始化状态，报告第0→1轮时应与后续联合训练转移区分。

现有 `target_history/epoch_*.pt` 的 `q1/q2/train_p1/train_p2` 来自该轮开始时的增强视图；其中中心来自轮末，新增两个phase字段注明这一点。不要将它与 `representation_history` 的轮末固定输入预测混合。

这些文件足以离线计算表示稳定性和分配转移，但不是逐轮完整模型/优化器恢复点。每个完整运行请保留全部7份表示快照、summary、config和训练日志；缺少历史表示时，无法从最终模型还原真实历史稳定性。

## 验证

```bash
python -m unittest discover -s tests -p 'test_representation_history.py' -v
```

测试覆盖：配置与baseline训练参数一致、开启记录前后真实训练流程的指标和参数一致、异常后恢复模型模式及随机状态、输入哈希与批大小无关且跨轮输入变化报错、快照软分配和余弦相似度可复算。
