"""分析 EMA 加权训练相对 baseline 的样本级效应。

对比两个训练好的模型（baseline 与 EMA+可靠性加权）在干净输入下的逐样本预测
切分出变差组/修复组，并用加权 run 的 target_history中的训练期诊断量（权重、margin、相对稳定性、视图敏感性）刻画各组的共性特征。

用法（在 con-dtc 目录下）:
    python -m src.analysis_ema_weight
    python -m src.analysis_ema_weight --base-run runs/xxx/时间戳/ --weighted-run runs/yyy/时间戳/
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

from src.data.data_process import QDTrajectoryDataset
from src.models.contrastive_model import ContrastiveTrajectoryModel

PROJECT_DIR = Path(__file__).resolve().parents[1]
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


def hungarian_mapped(preds, labels, num_clusters=NUM_CLUSTERS):
    """把任意簇编号映射到最优对应的真实标签，返回逐样本映射后标签。"""
    contingency = np.zeros((num_clusters, num_clusters), dtype=np.int64)
    np.add.at(contingency, (preds.numpy(), labels.numpy()), 1)
    rows, cols = linear_sum_assignment(-contingency)
    mapping = {int(r): int(c) for r, c in zip(rows, cols)}
    return np.array([mapping[int(p)] for p in preds])


def load_training_diagnostics(run_dir):
    """读取 target_history，返回各诊断量的全程均值（跳过首 epoch 哨兵）。"""
    files = sorted(glob.glob(str(Path(run_dir) / "target_history" / "epoch_*.pt")))
    history = [torch.load(f, weights_only=True) for f in files][1:]
    if not history:
        raise RuntimeError(f"no usable target_history epochs in {run_dir}")
    mean_of = lambda key: torch.stack([h[key] for h in history]).mean(0).numpy()
    return {
        "sample_weight": mean_of("sample_weight"),
        "margin_confidence": mean_of("margin_confidence"),
        "relative_stability": mean_of("relative_stability"),
        "raw_js_view1": mean_of("raw_js_view1"),
        "raw_js_view2": mean_of("raw_js_view2"),
    }

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

def print_per_label_comparison(
        labels,
        mapped_run1,
        mapped_run2,
        num_clusters=NUM_CLUSTERS,
):
    """
    对比 baseline 和实验模型中每个真实 label 的聚类结果。
    """
    run1_correct = mapped_run1 == labels
    run2_correct = mapped_run2 == labels

    print("\n--- 各 label 聚类结果对比 ---")
    print(
        f"{'label':>5s} "
        f"{'n':>5s} "
        f"{'base':>8s} "
        f"{'experiment':>11s} "
        f"{'变化':>8s} "
        f"{'修复':>6s} "
        f"{'变差':>6s} "
        f"{'base主要错分':>16s} "
        f"{'实验主要错分':>16s}"
    )

    for label in range(num_clusters):
        label_mask = labels == label
        total = int(label_mask.sum())

        if total == 0:
            continue

        run1_count = int((label_mask & run1_correct).sum())
        run2_count = int(
            (label_mask & run2_correct).sum()
        )

        run1_recall = run1_count / total
        run2_recall = run2_count / total
        recall_change = run2_recall - run1_recall

        fixed_count = int(
            (
                label_mask
                & ~run1_correct
                & run2_correct
            ).sum()
        )

        degraded_count = int(
            (
                label_mask
                & run1_correct
                & ~run2_correct
            ).sum()
        )

        run1_destination, run1_error_count, run1_error_ratio = (
            main_error_destination(
                mapped_predictions=mapped_run1,
                labels=labels,
                label=label,
            )
        )

        run2_destination, run2_error_count, run2_error_ratio = (
            main_error_destination(
                mapped_predictions=mapped_run2,
                labels=labels,
                label=label,
            )
        )

        if run1_destination is None:
            run1_error_text = "-"
        else:
            run1_error_text = (
                f"{run1_destination}:"
                f"{run1_error_count}"
                f"({run1_error_ratio:.0%})"
            )

        if run2_destination is None:
            run2_error_text = "-"
        else:
            run2_error_text = (
                f"{run2_destination}:"
                f"{run2_error_count}"
                f"({run2_error_ratio:.0%})"
            )

        print(
            f"{label:5d} "
            f"{total:5d} "
            f"{run1_recall:8.3f} "
            f"{run2_recall:11.3f} "
            f"{recall_change:+8.3f} "
            f"{fixed_count:6d} "
            f"{degraded_count:6d} "
            f"{run1_error_text:>16s} "
            f"{run2_error_text:>16s}"
        )

def analyze(run_1, run_2, batch_size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = QDTrajectoryDataset()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    lengths = np.array([int(dataset.data.iloc[i]["trajLen"]) for i in range(len(dataset))])

    preds_run1, labels_t = predict_clean(run_1, loader, device)
    preds_run2, _ = predict_clean(run_2, loader, device)
    y = labels_t.numpy()
    mapped_run1 = hungarian_mapped(preds_run1, labels_t)
    mapped_run2 = hungarian_mapped(preds_run2, labels_t)
    ok_run1 = mapped_run1 == y
    ok_run2 = mapped_run2 == y

    degraded = ok_run1 & ~ok_run2
    fixed = ~ok_run1 & ok_run2
    total = len(y)
    print(f"run1 {ok_run1.mean():.1%} | run2 {ok_run2.mean():.1%}")
    print(
        f"变差组: {degraded.sum()} 条 ({degraded.sum() / total:.1%})   "
        f"修复组: {fixed.sum()} 条 ({fixed.sum() / total:.1%})   "
        f"净损失 {degraded.sum() - fixed.sum()} 条"
    )
    print_per_label_overview(
        name="Run1",
        labels=y,
        mapped_predictions=mapped_run1,
    )

    print_per_label_overview(
        name="Run2",
        labels=y,
        mapped_predictions=mapped_run2,
    )
    print_per_label_comparison(
        labels=y,
        mapped_run1=mapped_run1,
        mapped_run2=mapped_run2,
    )

    print("\n--- 变差组真实标签分布（前 6）---")
    label_counts = np.bincount(y[degraded], minlength=NUM_CLUSTERS)
    for label in np.argsort(label_counts)[::-1][:6]:
        if label_counts[label]:
            total_l = (y == label).sum()
            print(
                f"  label {label:2d} (路径{label // 3}): "
                f"{label_counts[label]:4d}/{total_l} ({label_counts[label] / total_l:.0%})"
            )

    same_path = (mapped_run2[degraded] // 3 == y[degraded] // 3).mean()
    print(f"\n变差组错到同路径时间变体簇的比例: {same_path:.1%}")

    diag = load_training_diagnostics(run_2)
    print("\n--- 特征对比（加权 run 训练期全程均值）---")
    print(f"{'组':12s} {'n':>5s} {'权重':>6s} {'margin':>7s} {'相对稳定':>7s} {'view2/view1':>10s} {'均长':>6s}")
    groups = [
        ("变差组", degraded),
        ("修复组", fixed),
        ("都对", ok_run1& ok_run2),
        ("都错", ~ok_run1 & ~ok_run2),
    ]
    for name, mask in groups:
        if mask.sum() == 0:
            continue
        view_ratio = diag["raw_js_view2"][mask].mean() / diag["raw_js_view1"][mask].mean()
        print(
            f"{name:12s} {mask.sum():5d} {diag['sample_weight'][mask].mean():6.3f} "
            f"{diag['margin_confidence'][mask].mean():7.3f} "
            f"{diag['relative_stability'][mask].mean():7.3f} "
            f"{view_ratio:10.2f} {lengths[mask].mean():6.1f}"
        )

    low = diag["sample_weight"][degraded] < 0.4
    both_ok = ok_run1 & ok_run2
    print(
        f"\n变差组中训练期均权 < 0.4 的比例: {low.mean():.0%}  "
        f"(都对组: {(diag['sample_weight'][both_ok] < 0.4).mean():.0%})"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="EMA 加权训练的样本级退化分析")
    parser.add_argument(
        "--base-run",
        type=str,
        default=None,
        help="baseline run 目录（默认取 runs/condtc_qd_pre15_base 下最新一次）",
    )
    parser.add_argument(
        "--weighted-run",
        type=str,
        default=None,
        help="加权实验 run 目录（默认取 runs/condtc_qd_pre15_ema05_weighted 下最新一次）",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def latest_run(experiment_name):
    candidates = sorted(glob.glob(str(PROJECT_DIR / "runs" / experiment_name / "*/")))
    if not candidates:
        raise FileNotFoundError(f"no runs found for {experiment_name}")
    return candidates[-1]


def main():
    args = parse_args()
    base_run = args.base_run or latest_run("condtc_qd_pre15_base_main")
    ema_unweight_run = args.weighted_run or latest_run("condtc_qd_pre15_ema05_noweight")
    ema_weighted_run = args.weighted_run or latest_run("condtc_qd_pre15_ema05_weighted")
    print("base run:", base_run)
    print("ema unweight run:", ema_unweight_run)
    print("ema weighted run:", ema_weighted_run)
    analyze(base_run, ema_unweight_run, batch_size=args.batch_size)
    analyze(ema_unweight_run, ema_weighted_run, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
