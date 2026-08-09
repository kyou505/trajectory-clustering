import torch
from sklearn.cluster import KMeans

@torch.no_grad()
def extract_trajectory_embeddings(
        model,
        loader,
        device,
        max_batches=None,
):
    model.eval()
    all_embeddings = []
    all_labels = []

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        view = {
            "location_ids": batch["location_ids"].to(device),
            "time_ids": batch["time_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
            "pooling_mask": batch["pooling_mask"].to(device),
        }
        embedding = model.encode(view)
        all_embeddings.append(embedding.cpu())
        if "label" in batch:
            all_labels.append(batch["label"].cpu())

    embedding = torch.cat(all_embeddings, dim=0)
    labels = None
    if all_labels:
        labels = torch.cat(all_labels, dim=0)

    return embedding, labels


@torch.no_grad()
def initialize_cluster_centers(
        model,
        embeddings,
        num_clusters=12,
        seed=42,
        n_init=20
):
    expected_shape = (num_clusters, embeddings.size(1))
    actual_shape = tuple(model.clustering_layer.cluster_centers.shape)
    if actual_shape != expected_shape:
        raise ValueError(f"Cluster centers shape {actual_shape} != {expected_shape}")
    kmeans = KMeans(
        n_clusters=num_clusters,
        n_init=n_init,
        random_state=seed,
    )
    kmeans.fit(embeddings.detach().cpu().numpy())
    centers = torch.from_numpy(kmeans.cluster_centers_).to(
        device=model.clustering_layer.cluster_centers.device,
        dtype = model.clustering_layer.cluster_centers.dtype
    )
    # 不能重新赋值
    model.clustering_layer.cluster_centers.copy_(centers)
    return kmeans

def test():
    from pathlib import Path
    from torch.utils.data import DataLoader

    from src.data_process import QDTrajectoryDataset
    from src.models.contrastive_model import (
        ContrastiveTrajectoryModel,
    )

    device = torch.device("cpu")
    project_dir = Path(__file__).resolve().parent.parent
    checkpoint_path = (project_dir / "checkpoints" / "sttraj2vec_pretrain_best.pt")
    dataset = QDTrajectoryDataset()
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )
    model = ContrastiveTrajectoryModel(
        num_clusters=12,
    ).to(device)
    model.load_pretrained_encoder(
        checkpoint_path,
        map_location=device,
    )
    embeddings, labels = extract_trajectory_embeddings(
        model=model,
        loader=loader,
        device=device,
        max_batches=2,  # 本地只检查两个 batch
    )

    print("embeddings:", embeddings.shape)
    print("labels:", labels.shape)
    print("unique labels:", labels.unique())
    print("finite:", torch.isfinite(embeddings).all())

    kmeans = initialize_cluster_centers(
        model=model,
        embeddings=embeddings,
        num_clusters=12,
        seed=42,
        n_init=20,
    )
    print("kmeans centers:", kmeans.cluster_centers_.shape)
    print("kmeans inertia:", kmeans.inertia_)
    print("cluster counts:", torch.bincount(
        torch.from_numpy(kmeans.labels_),
        minlength=12,
    ))

if __name__ == "__main__":
    test()