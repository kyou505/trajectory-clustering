import torch
from torch.utils.data import random_split, DataLoader

from src.augmentation import ContrastiveTrajectoryDataset
from src.data_process import QDTrajectoryDataset
from src.mstm import MSTMDataset

def create_data_loaders(
    batch_size=32,
    seed=0
):
    base_dataset = QDTrajectoryDataset()
    total_size = len(base_dataset)
    
    train_size = int(total_size * 0.8)
    valid_size = (total_size - train_size) // 2
    test_size = total_size - train_size - valid_size
    
    train_base, valid_base, test_base = random_split(
        base_dataset,
        [train_size, valid_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    train_dataset = MSTMDataset(train_base, deterministic=False)
    valid_dataset = MSTMDataset(valid_base, deterministic=True, seed=seed + 10_000)
    test_dataset = MSTMDataset(test_base, deterministic=True, seed=seed + 20_000)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, valid_loader, test_loader

def create_contrastive_data_loader(
        batch_size=32,
        seed=0,
        mask_ratio=0.15,
        dropout_rate=0.1,
        max_offset_minutes=2,
        shuffle=True,
        base_dataset=None,
):
    if base_dataset is None:
        base_dataset = QDTrajectoryDataset()
    mstm_dataset = MSTMDataset(
        base_dataset=base_dataset,
        mask_ratio=mask_ratio,
        deterministic=False,
        seed=seed,
    )
    dataset = ContrastiveTrajectoryDataset(
        base_dataset=mstm_dataset,
        dropout_rate=dropout_rate,
        max_offset_minutes=max_offset_minutes,
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )

def test():
    train_loader, valid_loader, test_loader = (
        create_data_loaders(batch_size=32)
    )
    print("Train samples:", len(train_loader.dataset))
    print("Valid samples:", len(valid_loader.dataset))
    print("Test samples:", len(test_loader.dataset))
    loader = create_contrastive_data_loader(
        batch_size=8,
        seed=42
    )
    batch = next(iter(loader))
    print("batch keys:", batch.keys())
    print(
        "view1 locations:",
        batch["view1"]["location_ids"].shape,
    )
    print(
        "view2 locations:",
        batch["view2"]["location_ids"].shape,
    )
    print("labels:", batch["label"].shape)
    
if __name__ == "__main__":
    test()

    