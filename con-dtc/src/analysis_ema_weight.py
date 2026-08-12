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


def analyze(base_run, weighted_run, batch_size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = QDTrajectoryDataset()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    lengths = np.array([int(dataset.data.iloc[i]["trajLen"]) for i in range(len(dataset))])

    preds_base, labels_t = predict_clean(base_run, loader, device)
    preds_weighted, _ = predict_clean(weighted_run, loader, device)
    y = labels_t.numpy()
    mapped_base = hungarian_mapped(preds_base, labels_t)
    mapped_weighted = hungarian_mapped(preds_weighted, labels_t)
    ok_base = mapped_base == y
    ok_weighted = mapped_weighted == y

    degraded = ok_base & ~ok_weighted
    fixed = ~ok_base & ok_weighted
    total = len(y)
    print(f"base 正确率 {ok_base.mean():.1%} | 加权版 {ok_weighted.mean():.1%}")
    print(
        f"变差组: {degraded.sum()} 条 ({degraded.sum() / total:.1%})   "
        f"修复组: {fixed.sum()} 条 ({fixed.sum() / total:.1%})   "
        f"净损失 {degraded.sum() - fixed.sum()} 条"
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

    same_path = (mapped_weighted[degraded] // 3 == y[degraded] // 3).mean()
    print(f"\n变差组错到同路径时间变体簇的比例: {same_path:.1%}")

    diag = load_training_diagnostics(weighted_run)
    print("\n--- 特征对比（加权 run 训练期全程均值）---")
    print(f"{'组':12s} {'n':>5s} {'权重':>6s} {'margin':>7s} {'相对稳定':>7s} {'view2/view1':>10s} {'均长':>6s}")
    groups = [
        ("变差组", degraded),
        ("修复组", fixed),
        ("都对", ok_base & ok_weighted),
        ("都错", ~ok_base & ~ok_weighted),
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
    both_ok = ok_base & ok_weighted
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
    base_run = args.base_run or latest_run("condtc_qd_pre15_base")
    weighted_run = args.weighted_run or latest_run("condtc_qd_pre15_ema05_weighted")
    print("base run:", base_run)
    print("weighted run:", weighted_run)
    analyze(base_run, weighted_run, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
