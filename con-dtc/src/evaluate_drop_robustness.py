import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from src.data.data_process import QDTrajectoryDataset
from src.evaluate_condtc import (
    predict_clusters,
    clustering_metrics,
)
from src.models.contrastive_model import (
    ContrastiveTrajectoryModel,
)
from src.training.train_condtc import get_device

PAD_ID = 0


def apply_observation_dropout(
        sample,
        dropout_rate,
        corruption_seed=0,
        min_keep_points=2,
):
    """
    对一条轨迹模拟真实观测点丢失。

    特性：
    1. 同步删除 location 和 time；
    2. 不使用 [MASK]，被删除的点不会出现在新轨迹中；
    3. 保留点维持原始顺序；
    4. 删除后重新 padding 到原始固定长度；
    5. CLS 保留在第一个位置；
    6. SEP 保留在最后一个位置，和当前数据预处理一致；
    7. 相同 sample index、dropout rate 和 seed 的结果可重复。

    Args:
        sample:
            QDTrajectoryDataset 返回的一条样本。

        dropout_rate:
            轨迹点删除比例，范围为 [0, 1)。

        corruption_seed:
            控制评估缺失位置的随机种子。

        min_keep_points:
            每条轨迹至少保留的真实轨迹点数量。

    Returns:
        删除轨迹点并重新 padding 后的样本。
    """
    if not 0.0 <= dropout_rate < 1.0:
        raise ValueError("dropout_rate must be in [0, 1)")

    if min_keep_points < 1:
        raise ValueError("min_keep_points must be at least 1")

    required_keys = {
        "index",
        "location_ids",
        "time_ids",
        "attention_mask",
        "pooling_mask",
        "length",
    }
    missing_keys = required_keys - set(sample)

    if missing_keys:
        raise KeyError(f"sample is missing keys: "f"{sorted(missing_keys)}")

    location_ids = sample["location_ids"]
    time_ids = sample["time_ids"]
    length = int(sample["length"].item())
    sample_index = int(sample["index"].item())

    if location_ids.ndim != 1:
        raise ValueError("location_ids must be one-dimensional")

    if time_ids.shape != location_ids.shape:
        raise ValueError("location_ids and time_ids must have the same shape")


    # 除去 CLS 和 SEP 后，轨迹能够容纳的最大点数。
    trajectory_capacity = location_ids.numel() - 2
    if length > trajectory_capacity:
        raise ValueError(f"trajectory length {length} exceeds capacity {trajectory_capacity}")

    actual_min_keep = min(min_keep_points, length)
    requested_drop_count = int(length * dropout_rate)
    max_drop_count = length - actual_min_keep
    drop_count = min(requested_drop_count, max_drop_count)
    keep_count = length - drop_count

    # 每条样本使用独立、确定的随机数生成器。
    # 相同 seed 和 index 始终生成相同排列。
    generator = torch.Generator()
    generator.manual_seed(int(corruption_seed) + sample_index)
    permutation = torch.randperm(length, generator=generator)

    # 原始轨迹点位于索引 [1, length]。
    # 排序保证剩余轨迹点保持原始先后顺序。
    keep_positions = permutation[:keep_count].sort().values + 1
    kept_location_ids = location_ids[keep_positions]
    kept_time_ids = time_ids[keep_positions]

    # 创建重新 padding 的轨迹。
    new_location_ids = torch.full_like(location_ids, fill_value=PAD_ID)
    new_time_ids = torch.full_like(time_ids, fill_value=PAD_ID)

    # 保留原始 CLS。
    new_location_ids[0] = location_ids[0]
    new_time_ids[0] = time_ids[0]

    # 放入真正保留下来的轨迹点。
    new_location_ids[1:keep_count + 1] = kept_location_ids
    new_time_ids[1:keep_count + 1] = kept_time_ids

    # 当前数据格式将 SEP 固定放在最后一个位置。
    new_location_ids[-1] = location_ids[-1]
    new_time_ids[-1] = time_ids[-1]

    # attention_mask 包含 CLS、有效轨迹点和 SEP。
    new_attention_mask = new_location_ids.ne(PAD_ID)

    # pooling_mask 只标记真实轨迹点，
    # 不包括 CLS、SEP 和 PAD。
    new_pooling_mask = torch.zeros_like(sample["pooling_mask"], dtype=torch.bool)
    new_pooling_mask[1:keep_count + 1] = True
    result = dict(sample)
    result.update({
        "location_ids": new_location_ids,
        "time_ids": new_time_ids,
        "attention_mask": new_attention_mask,
        "pooling_mask": new_pooling_mask,
        "length": sample["length"].new_tensor(keep_count),
        "drop_count": sample["length"].new_tensor(drop_count),
        "dropout_rate": torch.tensor(dropout_rate, dtype=torch.float32),
    })

    return result

class ObservationDropDataset(Dataset):
    def __init__(
            self,
            base_dataset,
            dropout_rate,
            corruption_seed,
            min_keep_points=2,
    ):
        self.base_dataset = base_dataset
        self.dropout_rate = dropout_rate
        self.corruption_seed = corruption_seed
        self.min_keep_points = min_keep_points

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        sample = self.base_dataset[index]

        return apply_observation_dropout(
            sample=sample,
            dropout_rate=self.dropout_rate,
            corruption_seed=self.corruption_seed,
            min_keep_points=self.min_keep_points,
        )

def load_trained_model(checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise ValueError(f"checkpoint_path {checkpoint_path} is not a file")
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    config = checkpoint["configs"]
    model = ContrastiveTrajectoryModel(num_clusters=config["num_clusters"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, config

def predict_with_dropout(
        model,
        dropout_rate,
        corruption_seed,
        batch_size,
        device,
):
    base_dataset = QDTrajectoryDataset()
    dropped_dataset = ObservationDropDataset(
        base_dataset=base_dataset,
        dropout_rate=dropout_rate,
        corruption_seed=corruption_seed,
    )
    loader = DataLoader(
        dropped_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    probabilities, predictions, labels = predict_clusters(
        model=model,
        loader=loader,
        device=device,
    )
    metrics = clustering_metrics(
        labels=labels.numpy(),
        predictions=predictions.numpy(),
    )
    return {
        "probabilities": probabilities,
        "predictions": predictions,
        "labels": labels,
        "metrics": metrics,
    }

def evaluate_drop_robustness(
        checkpoint_path,
        dropout_rate,
        corruption_seed=41,
        batch_size=256
):
    device = get_device()
    print("device:", device)
    model, config = load_trained_model(checkpoint_path, device)
    clean_result = predict_with_dropout(
        model=model,
        dropout_rate=0.0,
        corruption_seed=corruption_seed,
        batch_size=batch_size,
        device=device,
    )
    dropped_result = predict_with_dropout(
        model=model,
        dropout_rate=dropout_rate,
        corruption_seed=corruption_seed,
        batch_size=batch_size,
        device=device,
    )
    clean_predictions = clean_result["predictions"]
    dropped_predictions = dropped_result["predictions"]
    consistency = (clean_predictions == dropped_predictions).float().mean().item()
    clean_uacc = clean_result["metrics"]["uacc"]
    dropped_uacc = dropped_result["metrics"]["uacc"]
    result = {
        "checkpoint": str(checkpoint_path),
        "dropout_rate": dropout_rate,
        "corruption_seed": corruption_seed,
        "clean_metrics": clean_result["metrics"],
        "dropped_metrics": dropped_result["metrics"],
        "uacc_drop": clean_uacc - dropped_uacc,
        "prediction_consistency": consistency,
    }
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Con-DTC under trajectory point dropout"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dropout-rate", type=float, required=True)
    parser.add_argument("--corruption-seed", type=int, default=41)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()

def main():
    args = parse_args()
    result = evaluate_drop_robustness(
        checkpoint_path=args.checkpoint,
        dropout_rate=args.dropout_rate,
        corruption_seed=args.corruption_seed,
        batch_size=args.batch_size,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()