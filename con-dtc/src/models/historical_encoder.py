"""在训练轮次边界更新并冻结的编码器快照，与在线模型相互独立。"""
from copy import deepcopy
import math

import torch
from torch import nn

from src.models.encoder import mask_mean_pooling


class HistoricalEncoder(nn.Module):
    def __init__(self, encoder, source_epoch):
        super().__init__()
        self.encoder = deepcopy(encoder)
        self.refresh(encoder, source_epoch)

    @torch.no_grad()
    def refresh(self, encoder, source_epoch):
        """在每轮训练开始前调用，轮内各批次之间不刷新快照。"""
        self.encoder.load_state_dict(encoder.state_dict(), strict=True)
        self.source_epoch = source_epoch
        self.requires_grad_(False)
        for parameter in self.parameters():
            parameter.grad = None
        self.eval()

    def train(self, mode=True):
        # 即使外层模块调用 train()，也不能启用历史编码器的随机失活。
        return super().train(False)

    @torch.no_grad()
    def forward(self, view):
        hidden = self.encoder(
            location_ids=view["location_ids"],
            time_ids=view["time_ids"],
            attention_mask=view["attention_mask"],
        )
        return mask_mean_pooling(hidden, view["pooling_mask"])


class EMATargetEncoder(HistoricalEncoder):
    """逐步对在线编码器进行 EMA 更新，包括其浮点缓冲区。

    在训练轮次边界保存该目标编码器的快照，用于历史对比。
    目标编码器不通过梯度优化，其更新也不会改变当前轮已冻结的历史快照。
    """

    def __init__(self, encoder, momentum=0.8):
        if isinstance(momentum, bool) or not math.isfinite(momentum) or not 0 <= momentum < 1:
            raise ValueError("history_encoder_ema_momentum must be finite in [0, 1)")
        super().__init__(encoder, source_epoch=0)
        self.momentum = momentum
        self.num_updates = 0

    @torch.no_grad()
    def update(self, online_encoder):
        """在线优化器每次更新后调用一次，预热阶段也执行。"""
        source = online_encoder.state_dict()
        target = self.encoder.state_dict()
        if source.keys() != target.keys() or any(source[k].shape != v.shape for k, v in target.items()):
            raise ValueError("EMA target and online encoder states do not match")
        for name, value in target.items():
            if value.is_floating_point() or value.is_complex():
                value.mul_(self.momentum).add_(source[name], alpha=1 - self.momentum)
            else:
                # 计数器及整数、布尔类型的缓冲区不能取加权平均，直接复制。
                value.copy_(source[name])
        self.num_updates += 1
