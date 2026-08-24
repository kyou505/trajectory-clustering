import math
import torch
from torch import nn

NUM_TIME_SLOTS = 1440
TIME_TOKEN_OFFSET = 4

def extract_temporal_features(
        time_ids,
        pooling_mask
):
    """
    从一批轨迹中提取：
    1. 起始时间 sin
    2. 起始时间 cos
    3. 总持续时间
    4. 相邻点平均时间间隔
    5. 相邻点时间间隔标准差
    :param time_ids: [batch_size, sequence_length]
    :param pooling_mask: [batch_size, sequence_length]
    :return:
        features: [batch_size, 5]
    """
    valid_mask = pooling_mask.bool()
    batch_size, sequence_length = time_ids.shape
    time_slots = (time_ids.to(torch.float32) - TIME_TOKEN_OFFSET).clamp_min(0.0)
    positions = torch.arange(sequence_length, device=time_ids.device).unsqueeze(0).expand(batch_size, -1)
    first_positions = torch.where(valid_mask, positions, sequence_length).min(dim=1).values
    last_positions = torch.where(valid_mask, positions, -1).max(dim=1).values
    # 理论上每条轨迹至少存在一个有效点，这里避免异常索引
    first_positions = first_positions.clamp(min=0, max=sequence_length - 1)
    last_positions = last_positions.clamp(min=0, max=sequence_length - 1)
    start_time = time_slots.gather(dim=1, index=first_positions.unsqueeze(1)).squeeze(1)
    end_time = time_slots.gather(dim=1, index=last_positions.unsqueeze(1)).squeeze(1)
    # 使用取模处理跨午夜轨迹
    duration = torch.remainder(end_time - start_time, NUM_TIME_SLOTS)
    pair_mask = valid_mask[:, :-1] & valid_mask[:, 1:]
    time_differences = torch.remainder(time_slots[:, 1:] - time_slots[:, :-1], NUM_TIME_SLOTS)
    time_differences = time_differences * pair_mask.to(time_differences.dtype)
    pair_count = pair_mask.sum(dim=1).clamp_min(1)
    pair_count_float = pair_count.to(time_differences.dtype)
    mean_interval = time_differences.sum(dim=1) / pair_count_float
    centered = (time_differences - mean_interval.unsqueeze(1)) * pair_mask.to(time_differences.dtype)
    interval_variance = centered.square().sum(dim=1) / pair_count_float
    interval_std = interval_variance.sqrt()
    # 周期时间使用 sin/cos，避免 23:59 和 00:00 距离很远
    start_angle = 2.0 * math.pi * start_time / NUM_TIME_SLOTS
    start_sin = torch.sin(start_angle)
    start_cos = torch.cos(start_angle)
    # 使用 log 归一化，避免 duration 与 interval 数值尺度差异过大
    log_scale = math.log1p(NUM_TIME_SLOTS)
    normalized_duration = torch.log1p(duration) / log_scale
    normalized_mean_interval = torch.log1p(mean_interval) / log_scale
    normalized_interval_std = torch.log1p(interval_std) / log_scale
    return torch.stack(
        [
            start_sin,
            start_cos,
            normalized_duration,
            normalized_mean_interval,
            normalized_interval_std,
        ],
        dim=1,
    )

class TemporalFeatureEncoder(nn.Module):
    def __init__(
            self,
            output_dim,
            hidden_dim=64
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, time_ids, pooling_mask):
        temporal_features = extract_temporal_features(
            time_ids=time_ids,
            pooling_mask=pooling_mask,
        )
        return self.network(temporal_features)