from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (normalized_mutual_info_score, rand_score, adjusted_rand_score)
from torch.utils.data import DataLoader
from src.data.data_process import QDTrajectoryDataset
from src.models.contrastive_model import (
    ContrastiveTrajectoryModel,
)
from src.training.train_condtc import get_device

@torch.no_grad()
def predict_clusters(
    model,
    loader,
    device,
    max_batches=None,
):
    model.eval()
    all_probabilities = []
    all_predictions = []
    all_labels = []
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        view = {
            "location_ids": batch["location_ids"].to(device),
            "time_ids": batch["time_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
            "pooling_mask": batch["pooling_mask"].to(device),
        }
        embeddings = model.encode(view)
        probabilities = model.clustering_layer(embeddings)
        predictions = probabilities.argmax(dim=1)
        all_probabilities.append(probabilities.cpu())
        all_predictions.append(predictions.cpu())
        all_labels.append(batch["label"].cpu())

    if not all_probabilities:
        raise RuntimeError("No samples were evaluated")
    probabilities = torch.cat(all_probabilities, dim=0)
    predictions = torch.cat(all_predictions, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return probabilities, predictions, labels

def clustering_accuracy(
        labels,
        predictions,
):
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    true_values, true_indices = np.unique(
        labels,
        return_inverse=True,
    )
    predicted_values, predicted_indices = np.unique(
        predictions,
        return_inverse=True,
    )
    contingency = np.zeros(
        (
            len(predicted_values),
            len(true_values),
        ),
        dtype=np.int64,
    )
    np.add.at(
        contingency,
        (predicted_indices, true_indices),
        1,
    )
    row_indices, column_indices = linear_sum_assignment(-contingency)
    matched = contingency[row_indices, column_indices].sum()
    return matched / labels.size

def clustering_metrics(
        labels,
        predictions,
):
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    uacc = clustering_accuracy(labels, predictions)
    nmi = normalized_mutual_info_score(labels, predictions, average_method="geometric")
    ri = rand_score(labels, predictions)
    ari = adjusted_rand_score(labels, predictions)
    return {
        "uacc": float(uacc),
        "nmi": float(nmi),
        "ri": float(ri),
        "ari": float(ari),
    }

def evaluate_checkpoint(
    checkpoint_path,
    batch_size=256,
    max_batches=None,
    dataset_name=None,
):
    device = get_device()
    print("device:", device)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    config = checkpoint["configs"]
    if dataset_name is None:
        dataset_name = config.get("dataset", "qdTimeNoise0424")
    print("dataset:", dataset_name)
    dataset = QDTrajectoryDataset(dataset_name=dataset_name)
    model = ContrastiveTrajectoryModel(
        location_vocab_size=config.get(
            "location_vocab_size", dataset.location_vocab_size
        ),
        time_vocab_size=config.get(
            "time_vocab_size", dataset.time_vocab_size
        ),
        max_length=config.get("max_length", dataset.max_length),
        num_clusters=config["num_clusters"],
    ).to(device)
    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    probabilities, predictions, labels = predict_clusters(
        model=model,
        loader=loader,
        device=device,
        max_batches=max_batches,
    )
    metrics = clustering_metrics(
        labels=labels.numpy(),
        predictions=predictions.numpy(),
    )
    cluster_counts = torch.bincount(
        predictions,
        minlength=config["num_clusters"],
    )
    print("cluster counts:", cluster_counts)
    print("metrics:", metrics)
    return metrics


def test():
    project_dir = Path(__file__).resolve().parent.parent
    checkpoint_path = project_dir / "checkpoints" / "condtc_smoke.pt"
    metrics = evaluate_checkpoint(
        checkpoint_path=checkpoint_path,
        batch_size=64,
        max_batches=2,
    )
    print(metrics)


if __name__ == "__main__":
    test()
