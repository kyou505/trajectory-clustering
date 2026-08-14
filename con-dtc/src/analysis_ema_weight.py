"""分析多组 Con-DTC run 的聚类性能、权重分配和伪簇迁移。

对比对照 run 与实验 run 在干净输入下的最终聚类结果，并结合
target_history 分析逐 epoch 样本权重、伪簇有效总权重和相邻 epoch 迁移。

用法（在 con-dtc 目录下）:
    python -m src.analysis_ema_weight \
        --runs \
        runs/实验A/时间戳_seed... \
        runs/实验B/时间戳_seed... \
        runs/实验C/时间戳_seed...
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    rand_score,
)
from torch.utils.data import DataLoader

from src.data.data_process import QDTrajectoryDataset
from src.models.contrastive_model import ContrastiveTrajectoryModel

NUM_CLUSTERS = 12


@torch.no_grad()
def predict_clean(run_dir, loader, device):
    """加载 run 目录下的 checkpoint，对干净输入输出簇分配。"""
    checkpoint = torch.load(
        Path(run_dir) / "checkpoints" / "condtc_best.pt",
        map_location=device,
        weights_only=True,
    )
    model = ContrastiveTrajectoryModel(
        num_clusters=checkpoint["configs"]["num_clusters"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    preds, labels = [], []
    for batch in loader:
        view = {
            key: batch[key].to(device)
            for key in ("location_ids", "time_ids", "attention_mask", "pooling_mask")
        }
        preds.append(model.clustering_layer(model.encode(view)).argmax(1).cpu())
        labels.append(batch["label"])
    return torch.cat(preds), torch.cat(labels)


def to_numpy(values):
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def hungarian_mapped(preds, labels, num_clusters=NUM_CLUSTERS):
    """把任意簇编号映射到最优对应的真实标签，返回逐样本映射后标签。"""
    preds = to_numpy(preds)
    labels = to_numpy(labels)
    contingency = np.zeros((num_clusters, num_clusters), dtype=np.int64)
    np.add.at(contingency, (preds, labels), 1)
    rows, cols = linear_sum_assignment(-contingency)
    mapping = {int(r): int(c) for r, c in zip(rows, cols)}
    return np.array([mapping[int(p)] for p in preds])


def compute_clustering_metrics(labels, raw_predictions):
    """计算完整聚类指标；UACC 使用 Hungarian 最优映射。"""
    labels = to_numpy(labels)
    raw_predictions = to_numpy(raw_predictions)
    mapped_predictions = hungarian_mapped(raw_predictions, labels)
    return {
        "uacc": float((mapped_predictions == labels).mean()),
        "nmi": float(normalized_mutual_info_score(
            labels,
            raw_predictions,
            average_method="geometric",
        )),
        "ri": float(rand_score(labels, raw_predictions)),
        "ari": float(adjusted_rand_score(labels, raw_predictions)),
    }


def compute_pair_separation_metrics(
        labels,
        raw_predictions,
        label_pairs=((4, 5), (7, 8)),
        num_clusters=NUM_CLUSTERS,
):
    """计算不依赖 Hungarian 映射的真实类别对分离指标。"""
    labels = to_numpy(labels)
    raw_predictions = to_numpy(raw_predictions)
    result = []
    for label_a, label_b in label_pairs:
        pair_mask = (labels == label_a) | (labels == label_b)
        pair_labels = labels[pair_mask]
        pair_predictions = raw_predictions[pair_mask]
        counts_a = np.bincount(
            raw_predictions[labels == label_a],
            minlength=num_clusters,
        )
        counts_b = np.bincount(
            raw_predictions[labels == label_b],
            minlength=num_clusters,
        )
        dominant_a = int(counts_a.argmax())
        dominant_b = int(counts_b.argmax())
        ratio_a = counts_a[dominant_a] / counts_a.sum()
        ratio_b = counts_b[dominant_b] / counts_b.sum()
        distribution_a = counts_a / counts_a.sum()
        distribution_b = counts_b / counts_b.sum()
        overlap = np.minimum(distribution_a, distribution_b).sum()
        nmi = normalized_mutual_info_score(pair_labels, pair_predictions, average_method="geometric")
        ari = adjusted_rand_score(pair_labels, pair_predictions)
        result.append({
            "label_a": label_a,
            "label_b": label_b,
            "dominant_a": dominant_a,
            "ratio_a": float(ratio_a),
            "dominant_b": dominant_b,
            "ratio_b": float(ratio_b),
            "same_dominant": dominant_a == dominant_b,
            "overlap": float(overlap),
            "nmi": float(nmi),
            "ari": float(ari),
        })
    return result


def load_target_history(run_dir):
    """按 epoch 读取 target_history，并保留实际训练时保存的诊断量。"""
    files = sorted(
        Path(run_dir).glob("target_history/epoch_*.pt")
    )
    if not files:
        raise FileNotFoundError(
            f"target_history not found in: {run_dir}"
        )
    result = []
    for file in files:
        item = torch.load(
            file,
            map_location="cpu",
            weights_only=True,
        )
        epoch = int(item.get("epoch", int(file.stem.split("_")[-1])))
        result.append((epoch, item))
    return result


def print_label_weight_history(
        name,
        run_dir,
        labels,
        num_clusters=NUM_CLUSTERS,
):
    """输出每个真实 label 在各 epoch 的平均训练权重。"""
    history = load_target_history(run_dir)
    labels = to_numpy(labels)
    epochs = [epoch for epoch, _ in history]
    weights = [item["sample_weight"].numpy() for _, item in history]
    ema_indices = [
        index for index, (_, item) in enumerate(history)
        if item.get("ema_enabled", index > 0)
    ]

    print(f"\n=== {name}：各真实 label 平均样本权重 ===")
    header = f"{'label':>5s} " + " ".join(
        f"e{epoch:02d}".rjust(8) for epoch in epochs
    ) + f" {'全程均值':>10s} {'EMA期均值':>11s}"
    print(header)
    for label in range(num_clusters):
        mask = labels == label
        epoch_means = np.array([
            epoch_weight[mask].mean()
            for epoch_weight in weights
        ])
        values = " ".join(
            f"{value:8.3f}" for value in epoch_means
        )
        ema_mean = (
            epoch_means[ema_indices].mean()
            if ema_indices
            else float("nan")
        )
        print(
            f"{label:5d} {values} "
            f"{epoch_means.mean():10.3f} {ema_mean:11.3f}"
        )


def print_pseudo_cluster_weight_history(
        name,
        run_dir,
        num_clusters=NUM_CLUSTERS,
):
    """输出每个原始伪簇的样本数、平均权重和总权重。"""
    history = load_target_history(run_dir)
    print(f"\n=== {name}：伪簇有效训练质量 ===")
    print(
        f"{'epoch':>5s} {'伪簇':>5s} {'样本数':>8s} "
        f"{'平均权重':>10s} {'总权重':>10s} {'总权重占比':>12s}"
    )
    for epoch, item in history:
        consensus = 0.5 * (item["q1"] + item["q2"])
        assignments = consensus.argmax(dim=1).numpy()
        weights = item["sample_weight"].numpy()
        all_weight = weights.sum()
        for cluster in range(num_clusters):
            mask = assignments == cluster
            count = int(mask.sum())
            mean_weight = float(weights[mask].mean()) if count else 0.0
            total_weight = float(weights[mask].sum()) if count else 0.0
            ratio = total_weight / all_weight if all_weight > 0 else 0.0
            print(
                f"{epoch:5d} {cluster:5d} {count:8d} "
                f"{mean_weight:10.3f} {total_weight:10.1f} {ratio:12.2%}"
            )


def compute_adjacent_epoch_migrations(
        run_dir,
        num_clusters=NUM_CLUSTERS,
):
    """计算原始伪簇编号在相邻 epoch 间的迁移矩阵。"""
    history = load_target_history(run_dir)
    epochs = [epoch for epoch, _ in history]
    assignments = []
    for _, item in history:
        consensus = 0.5 * (item["q1"] + item["q2"])
        assignments.append(consensus.argmax(dim=1).numpy())

    migrations = []
    for index in range(len(assignments) - 1):
        source = assignments[index]
        target = assignments[index + 1]
        matrix = np.zeros(
            (num_clusters, num_clusters),
            dtype=np.int64,
        )
        np.add.at(matrix, (source, target), 1)
        migrations.append((epochs[index], epochs[index + 1], matrix))
    return migrations


def print_adjacent_epoch_migrations(
        name,
        run_dir,
        top_n=12,
):
    """清晰输出相邻 epoch 的总体变化率和主要非对角迁移。"""
    migrations = compute_adjacent_epoch_migrations(run_dir)
    print(f"\n=== {name}：相邻 epoch 原始伪簇迁移 ===")
    print("说明：迁移使用模型原始伪簇编号，不进行逐 epoch Hungarian 重映射。")
    for source_epoch, target_epoch, matrix in migrations:
        total = int(matrix.sum())
        unchanged = int(np.trace(matrix))
        changed = total - unchanged
        print(
            f"\nepoch {source_epoch} -> {target_epoch}: "
            f"变化 {changed}/{total} ({changed / total:.2%})"
        )
        flows = []
        row_totals = matrix.sum(axis=1)
        for source_cluster in range(matrix.shape[0]):
            for target_cluster in range(matrix.shape[1]):
                if source_cluster == target_cluster:
                    continue
                count = int(matrix[source_cluster, target_cluster])
                if count == 0:
                    continue
                source_ratio = (
                    count / row_totals[source_cluster]
                    if row_totals[source_cluster] > 0
                    else 0.0
                )
                flows.append((
                    count,
                    source_cluster,
                    target_cluster,
                    source_ratio,
                ))
        if not flows:
            print("  无跨伪簇迁移")
            continue
        print(
            f"  {'from':>5s} {'to':>5s} {'数量':>8s} "
            f"{'占来源簇':>10s}"
        )
        for count, source_cluster, target_cluster, ratio in sorted(
                flows,
                reverse=True,
        )[:top_n]:
            print(
                f"  {source_cluster:5d} {target_cluster:5d} "
                f"{count:8d} {ratio:10.2%}"
            )

def main_error_destination(
        mapped_predictions,
        labels,
        label,
):
    """
    返回某个真实 label 的主要错误去向：
    (目标 label, 错误数量, 占该 label 错误样本的比例)
    """
    label_mask = labels == label

    wrong_predictions = mapped_predictions[
        label_mask & (mapped_predictions != labels)
    ]

    if wrong_predictions.size == 0:
        return None, 0, 0.0

    destinations, counts = np.unique(
        wrong_predictions,
        return_counts=True,
    )

    main_index = counts.argmax()

    return (
        int(destinations[main_index]),
        int(counts[main_index]),
        float(counts[main_index] / wrong_predictions.size),
    )

def print_per_label_overview(
        name,
        labels,
        mapped_predictions,
        num_clusters=NUM_CLUSTERS,
):
    """输出单个模型中各 label 的最终聚类情况。"""
    correct_mask = mapped_predictions == labels

    print(f"\n--- {name} 各 label 聚类情况 ---")
    print(
        f"{'label':>5s} "
        f"{'总数':>6s} "
        f"{'正确':>6s} "
        f"{'错误':>6s} "
        f"{'召回率':>8s} "
        f"{'主要错分去向':>18s}"
    )

    for label in range(num_clusters):
        label_mask = labels == label
        total = int(label_mask.sum())

        if total == 0:
            continue

        correct = int(
            (label_mask & correct_mask).sum()
        )
        error = total - correct
        recall = correct / total

        destination, destination_count, destination_ratio = (
            main_error_destination(
                mapped_predictions=mapped_predictions,
                labels=labels,
                label=label,
            )
        )

        if destination is None:
            destination_text = "-"
        else:
            destination_text = (
                f"{destination}:"
                f"{destination_count}"
                f"({destination_ratio:.1%})"
            )

        print(
            f"{label:5d} "
            f"{total:6d} "
            f"{correct:6d} "
            f"{error:6d} "
            f"{recall:8.3f} "
            f"{destination_text:>18s}"
        )

def run_display_name(run_dir):
    run_path = Path(run_dir).resolve()
    return f"{run_path.parent.name}/{run_path.name}"


def collect_run_results(run_dirs, loader, device):
    """依次推理多个 run，每个 checkpoint 只加载一次。"""
    results = []
    reference_labels = None
    for run_dir in run_dirs:
        run_path = Path(run_dir)
        if not run_path.is_dir():
            raise FileNotFoundError(f"run directory not found: {run_dir}")
        predictions, labels = predict_clean(run_path, loader, device)
        if reference_labels is None:
            reference_labels = labels
        elif not torch.equal(reference_labels, labels):
            raise ValueError(
                f"dataset label order differs for run: {run_dir}"
            )
        results.append({
            "run_dir": str(run_path),
            "name": run_display_name(run_path),
            "predictions": predictions,
            "mapped_predictions": hungarian_mapped(predictions, labels),
            "metrics": compute_clustering_metrics(labels, predictions),
            "pair_metrics": compute_pair_separation_metrics(
                labels,
                predictions,
            ),
        })
    return results, reference_labels


def print_multi_run_overall_metrics(results):
    print("\n=== 1. 各组实验总体聚类指标对比 ===")
    print(
        f"{'ID':>3s} {'run':68s} {'UACC':>9s} {'NMI':>9s} "
        f"{'RI':>9s} {'ARI':>9s}"
    )
    for index, result in enumerate(results, start=1):
        metrics = result["metrics"]
        print(
            f"{index:3d} {result['name']:68.68s} "
            f"{metrics['uacc']:9.4f} "
            f"{metrics['nmi']:9.4f} "
            f"{metrics['ri']:9.4f} "
            f"{metrics['ari']:9.4f}"
        )


def print_multi_run_pair_metrics(results):
    print("\n=== 2. 各组实验重点类别对分离指标对比 ===")
    print(
        f"{'ID':>3s} {'run':68s} {'类别对':>8s} "
        f"{'重叠度':>9s} {'Pair NMI':>10s} {'Pair ARI':>10s}"
    )
    for index, result in enumerate(results, start=1):
        for pair in result["pair_metrics"]:
            pair_name = f"{pair['label_a']}/{pair['label_b']}"
            print(
                f"{index:3d} {result['name']:68.68s} {pair_name:>8s} "
                f"{pair['overlap']:9.4f} "
                f"{pair['nmi']:10.4f} "
                f"{pair['ari']:10.4f}"
            )


def analyze_runs(run_dirs, batch_size=256, top_migrations=12):
    """按统一顺序输出多组实验的横向对比和逐 run 训练诊断。"""
    if len(run_dirs) < 2:
        raise ValueError("--runs requires at least two run directories")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = QDTrajectoryDataset()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    results, labels_t = collect_run_results(run_dirs, loader, device)
    labels = labels_t.numpy()

    print("=" * 104)
    print("Con-DTC 多组实验聚类、权重与迁移报告")
    print("=" * 104)
    print(f"实验组数: {len(results)}")
    print(f"样本数量: {len(labels)}")
    print(f"推理设备: {device}")
    for index, result in enumerate(results, start=1):
        print(f"run {index}: {Path(result['run_dir']).resolve()}")

    print_multi_run_overall_metrics(results)
    print_multi_run_pair_metrics(results)

    print("\n=== 3. 各组实验详细诊断 ===")
    for index, result in enumerate(results, start=1):
        name = f"run {index} | {result['name']}"
        run_dir = result["run_dir"]
        print("\n" + "#" * 104)
        print(name)
        print("#" * 104)
        print(f"路径: {Path(run_dir).resolve()}")

        print_per_label_overview(
            name=name,
            labels=labels,
            mapped_predictions=result["mapped_predictions"],
        )
        print_label_weight_history(
            name=name,
            run_dir=run_dir,
            labels=labels,
        )
        print_pseudo_cluster_weight_history(
            name=name,
            run_dir=run_dir,
        )
        print_adjacent_epoch_migrations(
            name=name,
            run_dir=run_dir,
            top_n=top_migrations,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="对比多组 Con-DTC run 的聚类、权重和迁移指标"
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="依次传入至少两个 runs/实验名/时间戳_seed... 目录",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--top-migrations",
        type=int,
        default=12,
        help="每组相邻 epoch 最多展示多少条非对角迁移",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    analyze_runs(
        run_dirs=args.runs,
        batch_size=args.batch_size,
        top_migrations=args.top_migrations,
    )



if __name__ == "__main__":
    main()
