"""LFSS 表示辅助分支：LS 跨视图 EMA 自蒸馏与 LI 同视图前代实例对比。

从 condtc-lfss 工作区迁入表示分支，不包含 LC、样本排除或额外 K-means。
LI 沿用已实验版本的仅负样本分母，因此损失值允许为负。
"""

from contextlib import contextmanager
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from src.models.encoder import mask_mean_pooling


def mlp(input_dim, hidden_dim, output_dim):
    """构造投影头或预测头：线性变换、批归一化、激活和输出变换。"""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


@contextmanager
def singleton_batchnorm(module, batch_size):
    """最后一个批次只有一个样本时，BN 使用已累积的统计量。"""
    states = [
        (m, m.training)
        for m in module.modules()
        if isinstance(m, nn.BatchNorm1d)
    ]
    try:
        if batch_size == 1:
            for m, _ in states:
                m.eval()
        yield
    finally:
        for m, training in states:
            m.train(training)


class FrozenRepresentation(nn.Module):
    """编码器与投影头的冻结副本，用于 EMA 目标网络和前代网络。"""

    def __init__(self, encoder, projector):
        super().__init__()
        self.encoder = deepcopy(encoder)
        self.projector = deepcopy(projector)
        self.requires_grad_(False)
        self.eval()

    def forward(self, view):
        # 对当前实际视图重新编码；池化后得到 z，再投影为 HRR 使用的 h。
        hidden = self.encoder(
            location_ids=view["location_ids"],
            time_ids=view["time_ids"],
            attention_mask=view["attention_mask"],
        )
        return self.projector(mask_mean_pooling(hidden, view["pooling_mask"]))


def historical_instance_loss(online, predecessor, temperature):
    """计算同视图的跨轮次实例损失，输入均为 [批大小, 投影维度]。

    两侧样本顺序必须一致：对角线是同实例正对，非对角线是其他实例负对。
    损失为负对相似度的 logsumexp 减去正对相似度，分母不包含正对，
    因此损失允许为负。只对在线侧传播梯度，不使用簇标签筛选负样本。
    """
    if online.shape[0] < 2:
        # 单实例批次没有负样本，返回与在线计算图连接的零损失。
        return online.sum() * 0.0
    logits = F.normalize(online, dim=1) @ F.normalize(predecessor.detach(), dim=1).T
    logits = logits / temperature
    negatives = logits.masked_fill(
        torch.eye(len(logits), device=logits.device, dtype=torch.bool),
        -torch.inf,
    )
    return (torch.logsumexp(negatives, dim=1) - logits.diagonal()).mean()


class LFSSAuxiliary(nn.Module):
    """HRR 表示约束，在线编码器由外部 ConDTC 模型共享。

    本模块持有可训练的投影头、预测头，以及不接受梯度的两套历史网络。
    LS 使用持续更新的 EMA 目标；LI 使用上一轮末 EMA 目标的冻结快照。
    """

    def __init__(self, model, config, seed):
        super().__init__()
        self.config = config
        self.seed = seed
        dim = model.clustering_layer.cluster_centers.shape[1]
        device = model.clustering_layer.cluster_centers.device

        # 新增头的初始化不改变原有训练增强和初始化的随机序列。
        devices = (
            list(range(torch.cuda.device_count()))
            if torch.cuda.is_available()
            else []
        )
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed + 31001)
            # 投影头 z -> h；预测头只用于 LS，将加噪的 h 映射到目标空间。
            self.projector = mlp(
                dim, config.hidden_dim, config.projection_dim
            ).to(device)
            self.predictor = mlp(
                config.projection_dim, config.hidden_dim, config.projection_dim
            ).to(device)
            # 与 LFSS 官方预测头的初始化一致。
            for layer in self.predictor.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, std=0.01)
                    nn.init.zeros_(layer.bias)

        self.target = FrozenRepresentation(model.encoder, self.projector)
        self.predecessor = deepcopy(self.target)
        # 特征噪声使用独立随机数生成器，不消耗训练增强和 dropout 的随机序列。
        self.noise_generator = torch.Generator(device="cpu").manual_seed(seed + 47001)
        # 记录已完成的训练轮次；0 表示尚未开始联合训练。
        self.target_epoch = 0
        self.predecessor_epoch = 0

    def train(self, mode=True):
        """在线头切换模式时，历史网络始终保持 eval，固定 dropout 和 BN 行为。"""
        super().train(mode)
        self.target.eval()
        self.predecessor.eval()
        return self

    def trainable_parameters(self):
        """优化器只接收新增在线头参数；共享编码器已由主模型加入优化器。"""
        return list(self.projector.parameters()) + list(self.predictor.parameters())

    @torch.no_grad()
    def update_target(self, model):
        """每次优化器更新后更新 EMA 参数；缓冲区复制，前代网络保持不动。"""
        for source, destination in (
            (model.encoder, self.target.encoder),
            (self.projector, self.target.projector),
        ):
            # 目标参数 <- m * 旧目标参数 + (1 - m) * 更新后的在线参数。
            for q, k in zip(source.parameters(), destination.parameters()):
                k.mul_(self.config.target_momentum).add_(
                    q, alpha=1 - self.config.target_momentum
                )
            # BN 运行均值、方差和计数等缓冲区直接复制，不执行 EMA。
            for q, k in zip(source.buffers(), destination.buffers()):
                k.copy_(q)

    @torch.no_grad()
    def begin_epoch(self, epoch):
        """将上一轮末 EMA 目标复制为本轮 LI 的固定参照；首轮使用初始化状态。"""
        if epoch != self.target_epoch + 1:
            raise RuntimeError("LFSS 目标网络轮次不连续")
        self.predecessor.load_state_dict(self.target.state_dict())
        self.predecessor.eval()
        self.predecessor_epoch = epoch - 1
        return {
            "training_epoch": epoch,
            "target_epoch": self.target_epoch,
            "instance_predecessor_epoch": self.predecessor_epoch,
        }

    def noise_like(self, x):
        """生成与输入同形状的标准高斯噪声；噪声幅度在 LS 分支中缩放。"""
        return torch.randn(
            x.shape, generator=self.noise_generator, dtype=x.dtype
        ).to(x.device)

    def forward(self, z1, z2, view1, view2):
        """使用两个本轮实际增强视图计算 LS、LI 及加权合计。

        z1、z2 是在线编码器对 view1、view2 的输出，保留与编码器的计算图。
        两项损失均作用于全部批次实例，不使用 DART 的聚类样本权重。
        """
        self.train()
        # 此处 q1/q2 表示在线投影 h1/h2，不是 DEC 的软分配 q。
        with singleton_batchnorm(self.projector, len(z1)):
            q1, q2 = self.projector(z1), self.projector(z2)

        # 历史网络重新编码同一批实际输入，不读取上轮随机增强的缓存表示。
        with torch.no_grad():
            if self.config.noise_weight > 0:
                k1, k2 = self.target(view1), self.target(view2)
            if self.config.instance_weight > 0:
                old1, old2 = self.predecessor(view1), self.predecessor(view2)
        zero = q1.sum() * 0.0
        ls = li = zero

        # LS：加噪在线投影经过预测头，匹配另一视图的 EMA 投影，无负样本。
        if self.config.noise_weight > 0:
            with singleton_batchnorm(self.predictor, len(z1)):
                p1 = self.predictor(q1 + self.config.noise_std * self.noise_like(q1))
                p2 = self.predictor(q2 + self.config.noise_std * self.noise_like(q2))
            ls = (
                (2 - 2 * F.cosine_similarity(p1, k2, dim=1)).mean()
                + (2 - 2 * F.cosine_similarity(p2, k1, dim=1)).mean()
            ) / 2

        # LI：同视图、同实例为正对，批内其他实例为负对，两个视图取平均。
        if self.config.instance_weight > 0:
            li = (
                historical_instance_loss(q1, old1, self.config.temperature)
                + historical_instance_loss(q2, old2, self.config.temperature)
            ) / 2

        total = self.config.noise_weight * ls + self.config.instance_weight * li
        return {"lfss_loss": total, "lfss_ls": ls, "lfss_li": li}

    def end_epoch(self, epoch):
        """轮末登记 EMA 网络所属轮次，供下一轮复制前代快照时检查顺序。"""
        if epoch != self.target_epoch + 1:
            raise RuntimeError("LFSS 轮末状态顺序错误")
        self.target_epoch = epoch

    def get_extra_state(self):
        # 保存独立噪声随机状态，便于重载后继续复现实例约束。
        return {
            "target_epoch": self.target_epoch,
            "predecessor_epoch": self.predecessor_epoch,
            "noise_rng": self.noise_generator.get_state(),
        }

    def set_extra_state(self, state):
        # torch.load 的 map_location 可能改变状态张量的设备，CPU 生成器需转回 CPU。
        self.target_epoch = state["target_epoch"]
        self.predecessor_epoch = state["predecessor_epoch"]
        self.noise_generator.set_state(state["noise_rng"].cpu())
