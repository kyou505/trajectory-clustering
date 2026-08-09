import torch
from torch.utils.data import random_split, DataLoader

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
    
    train_dataset = MSTMDataset(train_base)
    valid_dataset = MSTMDataset(valid_base)
    test_dataset = MSTMDataset(test_base)
    
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


def test():
    train_loader, valid_loader, test_loader = (
        create_data_loaders(batch_size=32)
    )
    print("Train samples:", len(train_loader.dataset))
    print("Valid samples:", len(valid_loader.dataset))
    print("Test samples:", len(test_loader.dataset))
    
if __name__ == "__main__":
    test()

    