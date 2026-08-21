import json
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset

PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = "qdTimeNoise0424"

def load_location_vocab(vocab_path):
    with vocab_path.open("r", encoding="utf-8") as f:
        vocab = json.load(f)
    return vocab

def encode_locations(trajectory, vocab):
    tokens = trajectory.split()
    encoded = []
    for token in tokens:
        if token in vocab:
            encoded.append(vocab[token])
        else:
            raise ValueError(f"Token '{token}' not found in vocabulary.")
    return encoded

def encode_time(time_sequence):
    tokens = time_sequence.split()
    encoded = []
    for token in tokens:
        if token == "[PAD]":
            encoded.append(0)
        else:
            raw_time = int(token)
            minutes_slot = (raw_time // 4) % 1440
            encoded.append(minutes_slot + 4)
    return encoded

class QDTrajectoryDataset(Dataset):
    def __init__(self, dataset_name=DEFAULT_DATASET):
        data_dir = PROJECT_DIR / "data" / dataset_name
        self.data = pd.read_hdf(data_dir / "data_k3.h5", key="x")
        self.length = pd.read_csv(data_dir / "trj_length.csv")["length"].to_numpy()
        self.location_vocab = load_location_vocab(
            data_dir / "location_vocab.json"
        )
        self.location_vocab_size = len(self.location_vocab)
        self.time_vocab_size = 1444

        if self.data.empty:
            raise ValueError(f"Dataset is empty: {data_dir}")
        padded_length = len(self.data.iloc[0]["trajectory"].split())
        self.max_length = padded_length + 2  # [CLS] 和 [SEP]
        
        if len(self.data) != len(self.length):
            raise ValueError("Data and length files must have the same number of rows.")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        location_tokens = encode_locations(row["trajectory"], self.location_vocab)
        time_tokens = encode_time(row["time"])
        length = int(row["trajLen"])
        
        location_ids = [
            1,
            *location_tokens,
            2
        ]
        time_ids = [
            1,
            *time_tokens,
            2
        ]
        location_ids = torch.tensor(location_ids, dtype=torch.long)
        time_ids = torch.tensor(time_ids, dtype=torch.long)

        if len(location_ids) != self.max_length:
            raise ValueError(
                f"Inconsistent padded length at row {idx}: "
                f"expected {self.max_length}, got {len(location_ids)}"
            )
        
        attention_mask = location_ids.ne(0)
        #
        pooling_mask = torch.zeros_like(location_ids, dtype=torch.bool)
        pooling_mask[1 : length+1] = True
        
        return {
            "index": torch.tensor(idx, dtype=torch.long), # 给每条轨迹增加稳定索引
            "location_ids": location_ids,
            "time_ids": time_ids,
            "attention_mask": attention_mask,
            "pooling_mask": pooling_mask,
            "length": torch.tensor(length, dtype=torch.long),
            "label": torch.tensor(row["label"], dtype=torch.long)
        }
        
def main():
    dataset = QDTrajectoryDataset()
    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    print("Sample keys:", sample.keys())
    print("Location IDs:", sample["location_ids"])
    print("Time IDs:", sample["time_ids"])
    print("Attention mask:", sample["attention_mask"])
    print("Pooling mask:", sample["pooling_mask"])
    print("Length:", sample["length"])
    print("Label:", sample["label"])
            
if __name__ == "__main__":
    main()
