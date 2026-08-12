import math
import torch
from sklearn.cluster import KMeans

from src.models.dec import target_distribution

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

@torch.no_grad()
def compute_global_cross_view_targets(
        model,
        loader,
        device,
        num_samples,
):
    model.eval()
    global_q1 = None
    global_q2 = None
    seen = torch.zeros(num_samples, dtype=torch.bool)
    for batch in loader:
        indices = batch["index"].long()
        view1 = {
            key: value.to(device)
            for key, value in batch["view1"].items()
            if key in {
                "location_ids",
                "time_ids",
                "attention_mask",
                "pooling_mask",
            }
        }
        view2 = {
            key: value.to(device)
            for key, value in batch["view2"].items()
            if key in {
                "location_ids",
                "time_ids",
                "attention_mask",
                "pooling_mask",
            }
        }
        z1 = model.encode(view1)
        z2 = model.encode(view2)
        q1 = model.clustering_layer(z1).cpu()
        q2 = model.clustering_layer(z2).cpu()
        if global_q1 is None:
            num_clusters = q1.size(1)
            global_q1 = torch.empty(num_samples, num_clusters, dtype=q1.dtype)
            global_q2 = torch.empty(num_samples, num_clusters, dtype=q2.dtype)
        if indices.min().item() < 0:
            raise ValueError("indices should be >= 0")
        if indices.max().item() >= num_samples:
            raise ValueError("indices should be < num_samples")
        if seen[indices].any():
            raise RuntimeError("indices should be unique")
        global_q1[indices] = q1
        global_q2[indices] = q2
        seen[indices] = True

    if global_q1 is None:
        raise RuntimeError("no batches were processed")
    if not seen.all():
        missing_count = (~seen).sum().item()
        raise RuntimeError(f"Missing {missing_count} samples")

    global_p1 = target_distribution(global_q1)
    global_p2 = target_distribution(global_q2)
    return {
        "q1": global_q1,
        "q2": global_q2,
        "p1": global_p1,
        "p2": global_p2,
    }

@torch.no_grad()
def compute_global_target_distribution(
        model,
        loader,
        device,
        num_samples
):
    model.eval()
    global_q = None
    seen = torch.zeros(num_samples, dtype=torch.bool)
    for batch in loader:
        indices = batch["index"].long()
        view = {
            "location_ids": batch["location_ids"].to(device),
            "time_ids": batch["time_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
            "pooling_mask": batch["pooling_mask"].to(device),
        }
        embeddings = model.encode(view)
        q = model.clustering_layer(embeddings).cpu()
        if global_q is None:
            global_q = torch.empty(
                num_samples,
                q.size(1),
                dtype=q.dtype,
            )
        if indices.min().item() < 0:
            raise ValueError("indices should be >= 0")
        if indices.max().item() >= num_samples:
            raise ValueError("indices should be < num_samples")
        if seen[indices].any(): raise RuntimeError("indices should be unique")
        global_q[indices] = q
        seen[indices] = True

    if global_q is None:
        raise RuntimeError("global_q is None")

    if not seen.all():
        missing_count = (~seen).sum().item()
        raise RuntimeError(f"Missing {missing_count} samples")

    global_p = target_distribution(global_q)
    return global_q, global_p

class CrossViewSoftTargetEMA:
    def __init__(self, momentum, minimum_weight=0.2):
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        if not 0.0 < minimum_weight <= 1.0:
            raise ValueError("minimum_weight must be in (0, 1]")
        self.momentum = momentum
        # minimum_weight=1.0 时全体样本权重恒为 1，数学上等价于关闭加权
        self.minimum_weight = minimum_weight
        self.ema_q1 = None
        self.ema_q2 = None

    @torch.no_grad()
    def update(self, q1, q2):
        q1 = q1.detach().cpu()
        q2 = q2.detach().cpu()

        # 当前 epoch 两个视图的共识分布
        current_consensus = 0.5 * (q1 + q2)
        # 计算决策置信度
        margin_confidence = compute_margin_confidence(current_consensus)

        num_samples = q1.size(0)
        if self.ema_q1 is None:
            # 第一个 epoch 没有历史
            nan_values = torch.full((num_samples,), float("nan"), dtype=q1.dtype)
            raw_js_view1 = nan_values.clone()
            raw_js_view2 = nan_values.clone()
            raw_js_divergence = nan_values.clone()
            relative_stability = nan_values.clone()
            reliability = nan_values.clone()
            sample_weight = torch.ones(num_samples, dtype=q1.dtype)

            self.ema_q1 = q1.clone()
            self.ema_q2 = q2.clone()
        else:
            # 使用更新前的 EMA 作为历史
            historical_consensus = 0.5 * (self.ema_q1 + self.ema_q2)
            # 计算当前稳定性
            view1_output = compute_temporal_stability(current_q=_sharpen(q1), history_q=_sharpen(self.ema_q1))
            view2_output = compute_temporal_stability(current_q=_sharpen(q2), history_q=_sharpen(self.ema_q2))
            raw_js_view1 = view1_output["js_divergence"]
            raw_js_view2 = view2_output["js_divergence"]
            raw_js_divergence = 0.5 * (raw_js_view1 + raw_js_view2)
            relative_stability = compute_relative_stability(raw_js_divergence)
            reliability_output = compute_reliability_weight(
                relative_stability=relative_stability,
                margin_confidence=margin_confidence,
                minimum_weight=self.minimum_weight,
            )
            reliability = reliability_output["reliability"]
            sample_weight = reliability_output["sample_weight"]
            # 更新EMA
            self.ema_q1.mul_(self.momentum).add_(q1, alpha=1 - self.momentum)
            self.ema_q2.mul_(self.momentum).add_(q2, alpha=1 - self.momentum)
        # 避免浮点误差导致每行概率和偏离 1
        self.ema_q1.div_(self.ema_q1.sum(dim=1, keepdim=True).clamp_min(1e-12))
        self.ema_q2.div_(self.ema_q2.sum(dim=1, keepdim=True).clamp_min(1e-12))
        return {
            "q1": self.ema_q1.clone(),
            "q2": self.ema_q2.clone(),
            "p1": target_distribution(self.ema_q1),
            "p2": target_distribution(self.ema_q2),
            "raw_js_view1": raw_js_view1.clone(),
            "raw_js_view2": raw_js_view2.clone(),
            "raw_js_divergence": raw_js_divergence.clone(),
            "relative_stability": relative_stability,
            "margin_confidence": margin_confidence,
            "reliability": reliability,
            "sample_weight": sample_weight,
        }

    def state_dict(self):
        return {
            "momentum": self.momentum,
            "ema_q1": self.ema_q1,
            "ema_q2": self.ema_q2,
        }

    def load_state_dict(self, state_dict):
        self.momentum = state_dict["momentum"]
        self.ema_q1 = state_dict["ema_q1"]
        self.ema_q2 = state_dict["ema_q2"]


def _sharpen(q, power=2.0, eps=1e-12):
    s = q.pow(power)
    return s / s.sum(dim=1, keepdim=True).clamp_min(eps)

@torch.no_grad()
def compute_temporal_stability(
        current_q,
        history_q,
        eps=1e-12,
):
    """
    使用归一化JS散度计算轨迹稳定性。
    :param current_q: 当前 epoch 的软聚类分布，形状 [N, K]。
    :param history_q: 更新前的历史软聚类分布，形状 [N, K]。
    :param eps: 数值稳定常数。
    :return:
        每条轨迹的稳定性分数，形状 [N]，取值范围为 [0, 1]。
    """
    # 防止输入因浮点误差没有严格归一化
    current_q = current_q / current_q.sum(dim=1, keepdim=True).clamp_min(eps) # 归一化当前软聚类分布
    history_q = history_q / history_q.sum(dim=1, keepdim=True).clamp_min(eps) # 对历史分布执行相同归一化
    mean_q = 0.5 * (current_q + history_q) # 计算 JS 散度中的中间分布

    current_log = current_q.clamp_min(eps).log() # 计算当前概率的自然对数
    history_log = history_q.clamp_min(eps).log() # 计算历史概率的自然对数
    mean_log = mean_q.clamp_min(eps).log() # 计算中间分布的自然对数

    current_kl = (current_q * (current_log - mean_log)).sum(dim=1) # 计算当前分布到中间分布的 KL 散度
    history_kl = (history_q * (history_log - mean_log)).sum(dim=1) # 计算历史分布到中间分布的 KL 散度
    js_divergence = 0.5 * (current_kl + history_kl) # 组合成 JS 散度
    normalized_js = js_divergence / math.log(2.0) # 归一化
    stability = 1.0 - normalized_js # 转成稳定性
    return {
        "js_divergence": js_divergence,
        "stability": stability.clamp(min=0.0, max=1.0),
    }


@torch.no_grad()
def compute_relative_stability(
        divergence,
        lower_quantile=0.1,
        upper_quantile=0.9,
        eps=1e-12,
):
    """
    将逐轨迹的原始分布差异转换为当前 epoch 内的相对稳定性分数。divergence 越小，稳定性越高。
    """
    lower = torch.quantile(divergence, lower_quantile)
    upper = torch.quantile(divergence, upper_quantile)
    scale = (upper - lower).clamp_min(eps)
    normalized_instability = ((divergence - lower) / scale).clamp(min=0.0, max=1.0)
    relative_stability = 1.0 - normalized_instability
    return relative_stability

@torch.no_grad()
def compute_margin_confidence(
        q,
        eps=1e-12,
):
    """
    根据最大和第二大聚类概率之间的间隔， 计算每条轨迹的决策置信度。
    :param q: 软聚类分布，形状为 [N, K]。
    :param eps: 数值稳定常数。
    :return:
        confidence: 形状为 [N]，取值范围为 [0, 1]。
    """
    q = q / q.sum(dim=1, keepdim=True).clamp_min(eps)
    top2 = torch.topk(q, k=2, dim=1).values
    top1_probability = top2[:, 0]
    top2_probability = top2[:, 1]
    confidence = (top1_probability - top2_probability) / top1_probability.clamp_min(eps)
    return confidence.clamp(min=0.0, max=1.0)

@torch.no_grad()
def compute_reliability_weight(
        relative_stability,
        margin_confidence,
        minimum_weight=0.2,
):
    """
    将时间稳定性和决策置信度转化为逐轨迹训练权重。
    :return:
        reliability: 原始综合可信度，形状 [N]，范围 [0, 1]。
        sample_weight: 带下限的训练权重，形状 [N]，范围 [minimum_weight, 1]。
    """
    reliability = (relative_stability * margin_confidence).clamp(min=0.0, max=1.0)
    sample_weight = minimum_weight + (1 - minimum_weight) * reliability
    return {
        "reliability": reliability,
        "sample_weight": sample_weight,
    }

def test():
    from pathlib import Path
    from torch.utils.data import DataLoader, Subset

    from data.data_process import QDTrajectoryDataset
    from src.models.contrastive_model import (
        ContrastiveTrajectoryModel,
    )
    from data.data_loader import (
        create_contrastive_data_loader,
    )

    device = torch.device("cpu")
    project_dir = Path(__file__).resolve().parents[2]
    checkpoint_path = (project_dir / "checkpoints" / "sttraj2vec_pretrain_best.pt")
    dataset = QDTrajectoryDataset()
    subset = Subset(dataset, range(128))
    global_loader = DataLoader(
        subset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )
    contrastive_loader = create_contrastive_data_loader(
        base_dataset=subset,
        batch_size=8,
        seed=42,
        shuffle=True
    )
    batch = next(iter(contrastive_loader))
    model = ContrastiveTrajectoryModel(
        num_clusters=12,
    ).to(device)
    model.load_pretrained_components(
        checkpoint_path,
        map_location=device,
    )

    embeddings, labels = extract_trajectory_embeddings(
        model=model,
        loader=global_loader,
        device=device,
        max_batches=None,
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

    global_q, global_p = compute_global_target_distribution(
        model=model,
        loader=global_loader,
        device=device,
        num_samples=128
    )
    print("global_q:", global_q)
    print("global_p:", global_p)

    p_batch = global_p[batch["index"]]
    print("batch indices:", batch["index"])
    print("P batch:", p_batch.shape)

if __name__ == "__main__":
    test()