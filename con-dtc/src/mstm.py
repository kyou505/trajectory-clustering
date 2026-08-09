from torch.utils.data import Dataset
import torch

from src.data_process import QDTrajectoryDataset

class MSTMDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        mask_ratio=0.15,
        location_vocab_size=148,
        time_vocab_size=1444
    ):
        self.base_dataset = base_dataset
        self.mask_ratio = mask_ratio
        self.location_vocab_size = location_vocab_size
        self.time_vocab_size = time_vocab_size

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # 取出原始样本
        sample = self.base_dataset[idx]
        original_locations = sample['location_ids']
        original_times = sample['time_ids']
        pooling_mask = sample['pooling_mask']
        
        # 深拷贝
        mask_locations = original_locations.clone()
        mask_times = original_times.clone()
        
        # 初始化预测目标，CrossEntropyLoss默认ignore_index=-100，所以target 为 -100 的位置不参与算 loss。
        # 只有被mask的位置才计算损失
        location_targets = torch.full_like(original_locations, -100)
        time_targets = torch.full_like(original_times, -100)
        
        #只选择真实点进行mask
        candidate_positions = torch.where(pooling_mask)[0] # 返回所有为 True 的下标
        num_candidate = len(candidate_positions)
        num_masked = max(1, int(num_candidate * self.mask_ratio)) # 保证轨迹再短也至少mask一个点
        
        # 随机生成mask的位置
        permutation = torch.randperm(num_candidate)
        masked_positions = candidate_positions[permutation[:num_masked]]
        
        # 把答案填进target
        location_targets[masked_positions] = original_locations[masked_positions]
        time_targets[masked_positions] = original_times[masked_positions]
        
        for position in masked_positions:
            probability = torch.rand(1).item()
            if probability < 0.8:
                # 80%：替换为空间和时间 MASK
                mask_locations[position] = 3
                mask_times[position] = 3
            elif probability < 0.9:
                # 10%：替换成随机有效 token
                mask_locations[position] = torch.randint(
                    low=4, # 0 - 3是特殊token
                    high=self.location_vocab_size,
                    size=(1,)
                )
                mask_times[position] = torch.randint(
                    low=4,
                    high=self.time_vocab_size,
                    size=(1,)
                )
            else:
                pass
            
        # 返回原样本的字段，并添加扰动后的输入
        result = dict(sample)
        mstm_mask = torch.zeros_like(
            original_locations,
            dtype=torch.bool,
        )
        mstm_mask[masked_positions] = True
        result.update({
            "masked_location_ids": mask_locations,
            "masked_time_ids": mask_times,
            "location_targets": location_targets,
            "time_targets": time_targets,
            "mstm_mask": mstm_mask,
        })
        return result

def test():
    base_dataset = QDTrajectoryDataset()
    mstm_dataset = MSTMDataset(base_dataset)
    print(f"Dataset length: {len(mstm_dataset)}")

    sample = mstm_dataset[0]
    original_locations = sample["location_ids"]
    mask_locations = sample["masked_location_ids"]
    location_targets = sample["location_targets"]
    mstm_mask = sample["mstm_mask"]
    masked_positions = torch.where(mstm_mask)[0]
    pooling_mask = sample["pooling_mask"]

    print("Masked positions:", masked_positions.tolist())
    print("Original locations at masked positions:", original_locations[masked_positions].tolist())
    print("Masked input at masked positions:", mask_locations[masked_positions].tolist())
    print("Targets at masked positions:", location_targets[masked_positions].tolist())

    # mask 位置必须都落在真实轨迹点上
    assert pooling_mask[masked_positions].all(), "存在 mask 位置落在非真实点上"

    # 非 mask 位置的 target 应全为 -100，mask 位置的 target 等于原 token
    unmasked = torch.ones_like(location_targets, dtype=torch.bool)
    unmasked[masked_positions] = False
    assert (location_targets[unmasked] == -100).all(), "非 mask 位置的 target 不是 -100"
    assert (location_targets[masked_positions] == original_locations[masked_positions]).all()

    # 非 mask 位置的输入不应被改动
    assert (mask_locations[unmasked] == original_locations[unmasked]).all(), "非 mask 位置的输入被改动了"
    assert (sample["masked_time_ids"][unmasked] == sample["time_ids"][unmasked]).all()

    # mask 数量符合 mask_ratio
    expected = max(1, int(pooling_mask.sum().item() * mstm_dataset.mask_ratio))
    assert len(masked_positions) == expected, f"mask 数量 {len(masked_positions)} != 预期 {expected}"

    print("All checks passed.")

if __name__ == "__main__":
    test()
