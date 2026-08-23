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
    CrossViewSoftTargetEMA
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
        global_sample_weight,
):
    model.train()
    indices = batch["index"].long()
    p1_batch = global_p1[indices].to(device)
    p2_batch = global_p2[indices].to(device)
    sample_weight_batch = global_sample_weight[indices].to(device)
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
        sample_weight=sample_weight_batch,
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
        global_sample_weight,
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
            global_sample_weight=global_sample_weight,
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
        dataset_name="qdTimeNoise0424",
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
        # 基础EMA
        target_ema_enabled=False,
        target_ema_start_epoch=1, # 分阶段开启
        target_ema_momentum=0.99,
        # DEC加权相关
        target_ema_weight_warmup=False, # 线性加权
        target_ema_minimum_weight=None,
        target_ema_weighting_signal="stability",
        # 基于软分配熵进行插值相关参数
        target_entropy_mix_enabled=False,
        target_entropy_quantile=0.7,
        target_entropy_upper_quantile=None,
        target_high_entropy_ema_alpha=0.5,
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
    print("dataset:", dataset_name)
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
    target_history_dir = output_dir / "target_history"
    target_history_dir.mkdir(parents=True, exist_ok=True)
    representation_history_dir = output_dir / "representation_history"
    representation_history_dir.mkdir(parents=True, exist_ok=True)
    dataset = QDTrajectoryDataset(dataset_name=dataset_name)
    model = ContrastiveTrajectoryModel(
        location_vocab_size=dataset.location_vocab_size,
        time_vocab_size=dataset.time_vocab_size,
        max_length=dataset.max_length,
        num_clusters=num_clusters,
    ).to(device)
    model.load_pretrained_components(
        pretrain_checkpoint_path,
        map_location=device,
    )
    initialization_loader = DataLoader(
        dataset,
        batch_size=initialization_batch_size,
        shuffle=False,
        num_workers=0,
    )
    embeddings, embedding_labels = extract_trajectory_embeddings(
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
    # torch.save(
    #     {
    #         "epoch": 0,
    #         "embeddings": embeddings.detach().cpu(),
    #         "labels": embedding_labels.detach().cpu(),
    #         "cluster_centers": model.clustering_layer.cluster_centers.detach().cpu()
    #     },
    #     representation_history_dir / "epoch_000.pt",
    # )
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
        shuffle=True,
        base_dataset=dataset,
    )
    target_loader = DataLoader(
        train_loader.dataset,
        batch_size=initialization_batch_size,
        shuffle=False,
        num_workers=0,
    )
    target_ema = CrossViewSoftTargetEMA(
        momentum=target_ema_momentum,
        minimum_weight=(
            target_ema_minimum_weight
            if target_ema_minimum_weight is not None
            else 1.0
        ),
        weighting_signal=target_ema_weighting_signal
    )
    best_train_loss = float("inf")
    history = []
    for epoch in range(1, num_epochs + 1):
        print(f"epoch={epoch} / num_epochs={num_epochs}")
        # 是否开启ema，target_ema_start_epoch控制开启的时机，默认直接开始，可分阶段开始
        ema_enabled = target_ema_enabled and epoch >= target_ema_start_epoch
        # 是否进行DEC加权effective_minimum_weight
        sample_weighting_enabled = target_ema_minimum_weight is not None
        # 是否启用线性权重 warm-up
        warmup_enabled = target_ema_weight_warmup
        if sample_weighting_enabled:
            effective_minimum_weight = compute_effective_minimum_weight(
                epoch=epoch,
                start_epoch=target_ema_start_epoch,
                num_epochs=num_epochs,
                target_minimum_weight=target_ema_minimum_weight,
                use_warmup=warmup_enabled,
            )
        else:
            effective_minimum_weight = 1.0
        # 固定本 epoch 的两个增强视图
        train_loader.dataset.set_epoch(epoch)
        current_targets = compute_global_cross_view_targets(
            model=model,
            loader=target_loader,
            device=device,
            num_samples=len(dataset),
        )
        num_samples = current_targets["q1"].size(0)
        nan_values = torch.full(
            (num_samples,),
            float("nan"),
            dtype=current_targets["q1"].dtype,
        )
        entropy = nan_values.clone()
        entropy_threshold = torch.tensor(float("nan"), dtype=current_targets["q1"].dtype)
        entropy_upper_threshold = torch.tensor(float("nan"), dtype=current_targets["q1"].dtype)
        high_entropy_mask = torch.zeros(num_samples, dtype=torch.bool)
        # EMA 启用前使用当前目标，历史目标占比为 0
        target_mix_alpha = torch.zeros(num_samples, dtype=current_targets["q1"].dtype)
        ema_targets = {
            # EMA尚未初始化时，用当前分配占位
            "q1": current_targets["q1"],
            "q2": current_targets["q2"],
            "p1": current_targets["p1"],
            "p2": current_targets["p2"],
            "raw_js_view1": nan_values.clone(),
            "raw_js_view2": nan_values.clone(),
            "raw_js_divergence": nan_values.clone(),
            "relative_stability": nan_values.clone(),
            "margin_confidence": nan_values.clone(),
            "reliability": nan_values.clone(),
            "sample_weight": torch.ones(
                num_samples,
                dtype=current_targets["q1"].dtype,
            ),
        }
        ema_initialized = False
        if not ema_enabled:
            # EMA未启用：baseline 或 分阶段启用EMA时未到达启用Epoch
            # 此时不做历史混合，样本权重全为1 
            train_p1 = current_targets["p1"]
            train_p2 = current_targets["p2"]
            train_sample_weight = torch.ones(num_samples, dtype=current_targets["q1"].dtype)
            # EMA 启用前的最后一个 epoch:用当前分配更新 EMA 历史，首次update只存储，不混合
            # 这样 start_epoch 当轮就有历史可比对,EMA 平滑与稳定性加权立即生效
            if epoch == target_ema_start_epoch - 1:
                target_ema.update(
                    q1=current_targets["q1"],
                    q2=current_targets["q2"],
                )
                ema_initialized = True
        else:
            # 更新EMA
            ema_targets=target_ema.update(
                q1=current_targets["q1"],
                q2=current_targets["q2"],
            )
            ema_initialized = True
            if target_entropy_mix_enabled:
                # 基于软分配熵的目标插值方法
                mix_output = compute_entropy_mix_alpha(
                    current_targets=current_targets,
                    entropy_quantile=target_entropy_quantile,
                    entropy_upper_quantile=target_entropy_upper_quantile,
                    high_entropy_ema_alpha=target_high_entropy_ema_alpha,
                )
                entropy = mix_output["entropy"]
                entropy_threshold = mix_output["entropy_threshold"]
                entropy_upper_threshold = mix_output["entropy_upper_threshold"]
                high_entropy_mask = mix_output["high_entropy_mask"]
                target_mix_alpha = mix_output["target_mix_alpha"]
                alpha = target_mix_alpha.unsqueeze(1)
                train_p1 = (1.0 - alpha) * current_targets["p1"] + alpha * ema_targets["p1"]
                train_p2 = (1.0 - alpha) * current_targets["p2"] + alpha * ema_targets["p2"]
                train_p1 = train_p1 / train_p1.sum(dim=1, keepdim=True).clamp_min(1e-12)
                train_p2 = train_p2 / train_p2.sum(dim=1, keepdim=True).clamp_min(1e-12)
            else:
                train_p1 = ema_targets["p1"]
                train_p2 = ema_targets["p2"]
                target_mix_alpha = torch.ones(
                    num_samples,
                    dtype=current_targets["q1"].dtype,
                )

            reliability = ema_targets["reliability"]
            # 是否开启DEC加权
            use_reliability_weighting = (
                sample_weighting_enabled
                and torch.isfinite(reliability).all()
            )
            if use_reliability_weighting:
                # 加权后权重
                train_sample_weight = effective_minimum_weight + (1.0 - effective_minimum_weight) * reliability
            else:
                train_sample_weight = torch.ones(num_samples, dtype=current_targets["q1"].dtype)
            ema_targets["sample_weight"] = train_sample_weight.clone()
        
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            global_p1=train_p1,
            global_p2=train_p2,
            global_sample_weight=train_sample_weight,
            max_batches=max_train_batches,
            log_interval=log_interval,
        )
        # epoch_embeddings, epoch_labels = (
        #     extract_trajectory_embeddings(
        #         model=model,
        #         loader=initialization_loader,
        #         device=device,
        #         max_batches=max_initialization_batches,
        #     )
        # )
        # torch.save(
        #     {
        #         "epoch": epoch,
        #         "embeddings": epoch_embeddings.detach().cpu(),
        #         "labels": epoch_labels.detach().cpu(),
        #         "cluster_centers": (
        #             model.clustering_layer.cluster_centers
        #             .detach()
        #             .cpu()
        #         ),
        #     },
        #     (
        #             representation_history_dir
        #             / f"epoch_{epoch:03d}.pt"
        #     ),
        # )
        
        print(
            "Train: "
            f"loss={train_metrics['loss']:.4f} "
            f"mstm={train_metrics['representation_loss']:.4f} "
            f"instance_nce={train_metrics['instance_contrastive_loss']:.4f} "
            f"cluster_nce={train_metrics['cluster_contrastive_loss']:.4f} "
            f"dec={train_metrics['clustering_loss']:.4f}"
        )
        
        torch.save(
            {
                "epoch": epoch,
                "q1": current_targets["q1"],
                "q2": current_targets["q2"],
                "ema_q1": ema_targets["q1"],
                "ema_q2": ema_targets["q2"],
                "p1": ema_targets["p1"],
                "p2": ema_targets["p2"],
                "train_p1": train_p1,
                "train_p2": train_p2,
                "ema_enabled": ema_enabled,
                "ema_initialized": ema_initialized,
                "raw_js_view1": ema_targets["raw_js_view1"],
                "raw_js_view2": ema_targets["raw_js_view2"],
                "raw_js_divergence": ema_targets["raw_js_divergence"],
                "relative_stability": ema_targets["relative_stability"],
                "margin_confidence": ema_targets["margin_confidence"],
                "entropy": entropy,
                "entropy_threshold": entropy_threshold,
                "entropy_upper_threshold": entropy_upper_threshold,
                "high_entropy_mask": high_entropy_mask,
                "target_mix_alpha": target_mix_alpha,
                "reliability": ema_targets["reliability"],
                "sample_weight": ema_targets["sample_weight"],
                "effective_minimum_weight": effective_minimum_weight,
                "cluster_centers": (
                    model.clustering_layer.cluster_centers
                    .detach()
                    .cpu()
                ),
            },
            target_history_dir / f"epoch_{epoch:03d}.pt",
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
                "target_ema_state_dict": target_ema.state_dict(),
                "configs": {
                    "dataset": dataset_name,
                    "location_vocab_size": dataset.location_vocab_size,
                    "time_vocab_size": dataset.time_vocab_size,
                    "max_length": dataset.max_length,
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

def compute_effective_minimum_weight(
        epoch,
        start_epoch,
        num_epochs,
        target_minimum_weight,
        use_warmup,
):
    # EMA 启用前不进行样本加权
    if epoch < start_epoch:
        return 1.0
    # 不启用 warm-up，EMA 开始后立即使用目标下限
    if not use_warmup:
        return target_minimum_weight
    # 从 start_epoch 到最后一个 epoch 的插值区间数
    warmup_length = max(num_epochs - start_epoch, 1)
    # 当前 warm-up 进度，限制在 [0, 1]
    progress = (epoch - start_epoch) / warmup_length
    progress = min(max(progress, 0.0), 1.0)
    # 从 1.0 线性下降到 target_minimum_weight
    return 1.0 - progress * (1.0 - target_minimum_weight)

# 根据当前软分配熵生成逐样本 EMA target 插值比例。
@torch.no_grad()
def compute_entropy_mix_alpha(
        current_targets,
        entropy_quantile, # 分位数
        high_entropy_ema_alpha,
        entropy_upper_quantile=None,
):
    # 两个视图的共识分布
    consensus = 0.5 * (current_targets["q1"] + current_targets["q2"])
    consensus = consensus / consensus.sum(
        dim=1,
        keepdim=True,
    ).clamp_min(1e-12)
    # 当前软聚类分配熵
    entropy = -(
            consensus * consensus.clamp_min(1e-12).log()
    ).sum(dim=1)
    # 当前 epoch 的高熵阈值
    entropy_threshold = torch.quantile(entropy, entropy_quantile)
    high_entropy_mask = entropy >= entropy_threshold

    if entropy_upper_quantile is None:
        # 现有硬阈值：
        # 低熵样本完全使用 EMA target；高熵样本混入当前目标
        target_mix_alpha = torch.ones_like(entropy)
        target_mix_alpha[high_entropy_mask] = high_entropy_ema_alpha
        entropy_upper_threshold = torch.tensor(float("nan"), dtype=entropy.dtype, device=entropy.device)
    else:
        # 连续插值：
        # P70以下alpha=1；
        # P70～P90从1线性下降到最小EMA比例；
        # P90以上保持最小EMA比例。
        entropy_upper_threshold = torch.quantile(entropy, entropy_upper_quantile)
        transition_width = (entropy_upper_threshold - entropy_threshold).clamp_min(1e-12)
        transition_progress = ((entropy - entropy_threshold) / transition_width).clamp(0.0, 1.0)
        target_mix_alpha = 1.0 - (1.0 - high_entropy_ema_alpha) * transition_progress

    return {
        "target_mix_alpha": target_mix_alpha,
        "entropy": entropy,
        "entropy_threshold": entropy_threshold,
        "entropy_upper_threshold": entropy_upper_threshold,
        "high_entropy_mask": high_entropy_mask,
    }

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
