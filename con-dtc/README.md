
实现参考：
- Con-DTC 论文
- 作者开源源码：https://github.com/TrajResearch/ConDTC

## 执行流程
```mermaid
flowchart TD
    A["QD HDF5 数据"] --> B["词表与数据编码"]
    B --> C["MSTM 遮盖样本"]
    C --> D["STTraj2Vec 预训练"]
    D --> E["加载最佳预训练权重"]
    E --> F["提取全量轨迹表示"]
    F --> G["KMeans 初始化聚类中心"]
    G --> H["构造两个增强视图"]
    H --> I["MSTM + InfoNCE + DEC 联合训练"]
    I --> J["保存 Con-DTC checkpoint"]
    J --> K["UACC / NMI / RI 评估"]
```

## 预训练

预训练阶段采用 Masked Spatial-Temporal Modeling：

1. 随机选择轨迹点；
2. 同时遮盖对应的位置 token 和时间 token；
3. 使用 Transformer 编码轨迹；
4. 分别预测原始位置和时间；
5. 保存验证集损失最小的模型。


## 微调阶段
1. 加载预训练权重；
2. 提取全量原始轨迹表示；
3. KMeans 初始化 DEC 中心；
4. 每个 epoch 更新全局 \(Q\) 和 \(P\)；
5. 构造点丢弃和时间偏移视图；
6. 计算联合损失；
7. 更新编码器、投影头和聚类中心。