import torch
from torch.utils.data import Dataset

PAD_ID = 0
CLS_ID = 1
SEP_ID = 2
NUM_TIME_SLOTS = 1440

# 随机删除轨迹点
def point_dropout(sample, dropout_rate=0.1):
    location_ids = sample["location_ids"]
    time_ids = sample["time_ids"]
    length = int(sample["length"].item())

    drop_count = max(1, int(length * dropout_rate))
    keep_count = length - drop_count

    permutation = torch.randperm(length)
    keep_positions = permutation[:keep_count].sort().values + 1
    kept_locations = location_ids[keep_positions]
    kept_times = time_ids[keep_positions]
    new_locations = torch.full_like(
        location_ids,
        PAD_ID,
    )
    new_times = torch.full_like(
        time_ids,
        PAD_ID,
    )
    new_locations[0] = CLS_ID
    new_times[0] = CLS_ID
    new_locations[1: keep_count + 1] = kept_locations
    new_times[1: keep_count + 1] = kept_times
    new_locations[-1] = SEP_ID
    new_times[-1] = SEP_ID

    pooling_mask = torch.zeros_like(sample["pooling_mask"])
    pooling_mask[1 : keep_count + 1] = True
    attention_mask = new_locations.ne(PAD_ID)

    return {
        "location_ids": new_locations,
        "time_ids": new_times,
        "attention_mask": attention_mask,
        "pooling_mask": pooling_mask,
        "length": torch.tensor(keep_count, dtype=torch.long),
    }

# 时间偏移
def time_offset(sample, max_offset_minutes=2):
    new_times = sample["time_ids"].clone()
    valid_postions = sample["pooling_mask"]

    offset = torch.randint(
        low=1,
        high=max_offset_minutes + 1,
        size=()
    ).item()

    time_slots = new_times[valid_postions] - 4
    shifted_slots = (time_slots + offset) % NUM_TIME_SLOTS
    new_times[valid_postions] = shifted_slots + 4

    return {
        "location_ids": sample["location_ids"].clone(),
        "time_ids": new_times,
        "attention_mask": sample["attention_mask"].clone(),
        "pooling_mask": sample["pooling_mask"].clone(),
        "length": sample["length"].clone(),
        "offset": torch.tensor(offset, dtype=torch.long),
    }

class ContrastiveTrajectoryDataset(Dataset):
    def __init__(
            self,
            base_dataset,
            dropout_rate=0.1,
            max_offset_minutes=2,
    ):
        self.base_dataset = base_dataset
        self.dropout_rate = dropout_rate
        self.max_offset_minutes = max_offset_minutes

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        view1 = point_dropout(sample, dropout_rate=self.dropout_rate)
        view2 = time_offset(sample, max_offset_minutes=self.max_offset_minutes)
        result = dict(sample)
        result["view1"] = view1
        result["view2"] = view2
        return result

def test():
    from src.data_process import (QDTrajectoryDataset)
    torch.manual_seed(42)
    base_dataset = QDTrajectoryDataset()
    dataset = ContrastiveTrajectoryDataset(base_dataset, dropout_rate=0.1, max_offset_minutes=2)
    sample = dataset[0]
    view1 = sample["view1"]
    view2 = sample["view2"]
    print("original length:", sample["length"].item())
    print("dropout view1:", view1)
    print("time offset view2:", view2)


if __name__ == "__main__":
    test()