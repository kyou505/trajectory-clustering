from pathlib import Path

import torch

from torch.utils.data import DataLoader
from src.models.condtc_loss import ConDTCTotalLoss
from src.data.data_loader import (
    create_contrastive_data_loader,
)
from src.data.data_process import QDTrajectoryDataset
from src.models.contrastive_model import (
    ContrastiveTrajectoryModel,
)
from src.training.cluster_init import (
    compute_global_cross_view_targets,
    extract_trajectory_embeddings,
    initialize_cluster_centers,
)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)

    if isinstance(value, dict):
        return {
            key: move_to_device(item, device)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            move_to_device(item, device)
            for item in value
        ]

    if isinstance(value, tuple):
        return tuple(
            move_to_device(item, device)
            for item in value
        )

    return value

def create_optimizer(
        model,
        representation_learning_rate=5e-5,
        clustering_learning_rate=1.5e-4,
        weight_decay=0.01,
):
    representation_parameters = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if not name.startswith("clustering_layer"):
            representation_parameters.append(param)

    clustering_parameters = list(model.clustering_layer.parameters())
    representation_ids = {
        id(param) for param in representation_parameters
    }
    clustering_ids = {
        id(param) for param in clustering_parameters
    }
    if representation_ids & clustering_ids:
        raise RuntimeError("Optimizer parameter groups overlap")
    optimizer = torch.optim.AdamW(
        [
            {"params": representation_parameters, "lr": representation_learning_rate},
            {"params": clustering_parameters, "lr": clustering_learning_rate},
        ],
        weight_decay=weight_decay,
    )
    return optimizer

def train_one_step(
        model,
        batch,
        criterion,
        optimizer,
        device,
        global_p1,
        global_p2,
):
    model.train()
    indices = batch["index"].long()
    p1_batch = global_p1[indices].to(device)
    p2_batch = global_p2[indices].to(device)
    batch = move_to_device(batch, device)
    optimizer.zero_grad(set_to_none=True)
    mstm_output = model.forward_mstm(
        masked_location_ids=batch["masked_location_ids"],
        masked_time_ids=batch["masked_time_ids"],
        attention_mask=batch["attention_mask"],
    )
    cluster_output = model(
        view1=batch["view1"],
        view2=batch["view2"],
    )
    losses = criterion(
        location_logits=mstm_output["location_logits"],
        time_logits=mstm_output["time_logits"],
        location_targets=batch["location_targets"],
        time_targets=batch["time_targets"],
        q1=cluster_output["q1"],
        q2=cluster_output["q2"],
        p1=p1_batch,
        p2=p2_batch,
        head_in1=cluster_output["head_in1"],
        head_in2=cluster_output["head_in2"],
        head_cl1=cluster_output["head_cl1"],
        head_cl2=cluster_output["head_cl2"],
    )
    if not torch.isfinite(losses["loss"]):
        raise RuntimeError("Loss is not finite")
    losses["loss"].backward()
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0
    )
    optimizer.step()
    return {
        name: value.detach().item()
        for name, value in losses.items()
    }

def train_one_epoch(
        model,
        loader,
        criterion,
        optimizer,
        device,
        global_p1,
        global_p2,
        max_batches=None,
        log_interval=50,
):
    model.train()
    metrics_sums = {}
    process_samples = 0
    process_batches = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch_size = batch["label"].size(0)
        metrics = train_one_step(
            model=model,
            batch=batch,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            global_p1=global_p1,
            global_p2=global_p2,
        )
        for name, value in metrics.items():
            metrics_sums[name] = metrics_sums.get(name, 0.0) + value * batch_size
        process_samples += batch_size
        process_batches += 1
        if (log_interval is not None and process_batches % log_interval == 0):
            print(
                f"batch={process_batches:04d} "
                f"loss={metrics['loss']:.4f} "
                f"representation_loss={metrics['representation_loss']:.4f} "
                f"clustering_loss={metrics['clustering_loss']:.4f} "
                f"instance_nce={metrics['instance_contrastive_loss']:.4f} "
                f"cluster_nce={metrics['cluster_contrastive_loss']:.4f} "
            )
    if process_batches == 0:
        raise RuntimeError("No batches were processed")
    epoch_metrics = {
        name: total / process_samples
        for name, total in metrics_sums.items()
    }
    epoch_metrics["batches"] = process_batches
    epoch_metrics["samples"] = process_samples
    return epoch_metrics

def train_condtc(
        num_epochs,
        batch_size=32,
        initialization_batch_size=256,
        num_clusters=12,
        time_loss_weight=0.1,
        clustering_loss_weight=2,
        instance_temperature=0.5,
        cluster_temperature=1.0,
        instance_loss_weight=1.0,
        cluster_contrastive_loss_weight=1.0,
        representation_learning_rate=5e-5,
        clustering_learning_rate=1.5e-4,
        weight_decay=0.01,
        seed=42,
        max_initialization_batches=None,
        max_train_batches=None,
        log_interval=50,
        output_dir=None,
        pretrain_checkpoint_path=None,
):
    if output_dir is None:
        raise ValueError("output_dir is required")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = get_device()
    print("device:", device)
    project_dir = Path(__file__).resolve().parents[2]
    if pretrain_checkpoint_path is None:
        pretrain_checkpoint_path = project_dir / "checkpoints" / "sttraj2vec_pretrain_best.pt"
    else:
        pretrain_checkpoint_path = Path(pretrain_checkpoint_path)
        if not pretrain_checkpoint_path.is_absolute():
            pretrain_checkpoint_path = project_dir / pretrain_checkpoint_path
    print("pretrain checkpoint:", pretrain_checkpoint_path)
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = project_dir / output_dir
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_checkpoint_path = checkpoint_dir / "condtc_best.pt"
    dataset = QDTrajectoryDataset()
    model = ContrastiveTrajectoryModel(num_clusters=num_clusters).to(device)
    pretrain_metadata = model.load_pretrained_components(
        pretrain_checkpoint_path,
        map_location=device,
    )
    initialization_loader = DataLoader(
        dataset,
        batch_size=initialization_batch_size,
        shuffle=False,
        num_workers=0,
    )
    embeddings, _ = extract_trajectory_embeddings(
        model=model,
        loader=initialization_loader,
        device=device,
        max_batches=max_initialization_batches,
    )
    kmeans = initialize_cluster_centers(
        model=model,
        embeddings=embeddings,
        num_clusters=num_clusters,
        seed=seed,
        n_init=20,
    )
    optimizer = create_optimizer(
        model=model,
        representation_learning_rate=representation_learning_rate,
        clustering_learning_rate=clustering_learning_rate,
        weight_decay=weight_decay,
    )
    criterion = ConDTCTotalLoss(
        time_loss_weight=time_loss_weight,
        clustering_loss_weight=clustering_loss_weight,
        instance_temperature=instance_temperature,
        cluster_temperature=cluster_temperature,
        instance_loss_weight=instance_loss_weight,
        cluster_contrastive_loss_weight=cluster_contrastive_loss_weight,
    )
    train_loader = create_contrastive_data_loader(
        batch_size=batch_size,
        seed=seed,
        shuffle=True
    )
    target_loader = DataLoader(
        train_loader.dataset,
        batch_size=initialization_batch_size,
        shuffle=False,
        num_workers=0,
    )
    best_train_loss = float("inf")
    history = []
    for epoch in range(1, num_epochs + 1):
        print(f"epoch={epoch} / num_epochs={num_epochs}")
        # 固定本 epoch 的两个增强视图
        train_loader.dataset.set_epoch(epoch)
        global_targets = compute_global_cross_view_targets(
            model=model,
            loader=target_loader,
            device=device,
            num_samples=len(dataset),
        )
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            global_p1=global_targets["p1"],
            global_p2=global_targets["p2"],
            max_batches=max_train_batches,
            log_interval=log_interval,
        )
        print(
            "Train: "
            f"loss={train_metrics['loss']:.4f} "
            f"mstm={train_metrics['representation_loss']:.4f} "
            f"instance_nce={train_metrics['instance_contrastive_loss']:.4f} "
            f"cluster_nce={train_metrics['cluster_contrastive_loss']:.4f} "
            f"dec={train_metrics['clustering_loss']:.4f}"
        )
        history.append({"epoch": epoch, "train": train_metrics})
        if train_metrics["loss"] < best_train_loss:
            best_train_loss = train_metrics["loss"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_metrics": train_metrics,
                "kmeans_inertia": float(kmeans.inertia_),
                "configs": {
                    "batch_size": batch_size,
                    "num_clusters": num_clusters,
                    "time_loss_weight": time_loss_weight,
                    "clustering_loss_weight": clustering_loss_weight,
                    "instance_temperature": instance_temperature,
                    "cluster_temperature": cluster_temperature,
                    "instance_loss_weight": instance_loss_weight,
                    "cluster_contrastive_loss_weight": cluster_contrastive_loss_weight,
                    "representation_learning_rate": representation_learning_rate,
                    "clustering_learning_rate": clustering_learning_rate,
                    "weight_decay": weight_decay,
                    "seed": seed,
                }
            }, output_checkpoint_path)
            print("save checkpoint: ", output_checkpoint_path)
    print("best train_loss: ", best_train_loss)
    return model, history, output_checkpoint_path




if __name__ == "__main__":
    train_condtc(
        num_epochs=1,
        batch_size=8,
        initialization_batch_size=64,
        max_initialization_batches=2,
        max_train_batches=3,
        log_interval=1,
        output_dir="runs/smoke",
    )
