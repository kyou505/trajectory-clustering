import json
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "qdTimeNoise0424"
DATA_PATH = DATA_DIR / "data_k3.h5"
VOCAB_PATH = DATA_DIR / "location_vocab.json"

SPECIAL_TOKENS = {
    "[PAD]": 0,
    "[CLS]": 1,
    "[SEP]": 2,
    "[MASK]": 3,
}

def load_data():
    df = pd.read_hdf(DATA_PATH, key="x")
    return df

def inspect_data(df):
    print("Data shape:", df.shape)
    print("Data columns:", df.columns.tolist())
    print("Data types:\n", df.dtypes)
    
    row = df.iloc[0]
    location_tokens = row["trajectory"].split()
    time_tokens = row["time"].split()
    
    print("Sample row:")
    print("label:", row["label"])
    print("traj_len:", row["trajLen"])
    print("Location tokens:", location_tokens[:5]) 
    print("Time tokens:", time_tokens[:5])


def build_location_vocab(df):
    raw_location_ids = set()
    for trajectory in df["trajectory"]:
        for token in trajectory.split():
            if token != "[PAD]":
                raw_location_ids.add(int(token))
                
    raw_location_ids = sorted(raw_location_ids)
    vocab = SPECIAL_TOKENS.copy()
    for raw_id in raw_location_ids:
        token = str(raw_id)
        vocab[token] = len(vocab)
        
    return vocab

def save_vocab(vocab):
    with VOCAB_PATH.open("w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
        
    print(f"Vocabulary saved to {VOCAB_PATH}")
    
def main():
    df = load_data()
    inspect_data(df)
    vocab = build_location_vocab(df)
    save_vocab(vocab)
    
if __name__ == "__main__":
    main()
    