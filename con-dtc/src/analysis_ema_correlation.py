"""对 Baseline 与分阶段 EMA 进行样本级相关性分析。

输出 EMA 目标影响、结果分组、label 统计、候选指标以及最终软预测
误差百分位。EMA run 必须保存完整 target_history，且必须是未进行
DEC 样本加权和目标插值的纯 EMA 实验。

在 con-dtc 目录下运行：

    python -m src.analysis_ema_correlation \
        --baseline-runs runs/baseline/condtc_qd_baseline_pre15 \
        --ema-runs runs/ema_staged/condtc_qd_ema_staged_s3_m07

结果只输出到终端，不创建文件。
"""

import argparse
import re
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata
from torch.utils.data import DataLoader

from src.data.data_process import QDTrajectoryDataset
from src.models.contrastive_model import ContrastiveTrajectoryModel
from src.models.dec import target_distribution


EPS = 1e-12
HIGH_IMPACT_QUANTILE = 0.8
ERROR_BINS = (
    ("0%～50%", 0.0, 0.5),
    ("50%～80%", 0.5, 0.8),
    ("80%～90%", 0.8, 0.9),
    ("90%～100%", 0.9, 1.0 + EPS),
)
OUTCOME_GROUPS = ("都正确", "EMA修复", "EMA新增错误", "都错误")


def discover_runs(paths):
    """将实验父目录或具体 run 目录展开为已完成的 run。"""
    discovered = []
    for value in paths:
        path = Path(value)
        if not path.is_dir():
            raise FileNotFoundError(f"run path does not exist: {path}")
        if (path / "checkpoints" / "condtc_best.pt").is_file():
            discovered.append(path)
            continue
        children = sorted(
            child
            for child in path.iterdir()
            if child.is_dir()
            and (child / "checkpoints" / "condtc_best.pt").is_file()
        )
        if not children:
            raise FileNotFoundError(f"no completed run found under: {path}")
        discovered.extend(children)
    unique = {str(path.resolve()): path for path in discovered}
    return [unique[key] for key in sorted(unique)]


def read_run_config(run_dir):
    config_path = Path(run_dir) / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"config snapshot not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def read_run_seed(run_dir):
    config = read_run_config(run_dir)
    try:
        return int(config["training"]["seed"])
    except (KeyError, TypeError, ValueError):
        match = re.search(r"seed(-?\d+)$", Path(run_dir).name)
        if match is None:
            raise ValueError(f"cannot determine seed for run: {run_dir}")
        return int(match.group(1))


def index_runs_by_seed(paths, name):
    runs = {}
    for run_dir in discover_runs(paths):
        seed = read_run_seed(run_dir)
        if seed in runs:
            raise ValueError(
                f"duplicate {name} run for seed={seed}: "
                f"{runs[seed]} and {run_dir}"
            )
        runs[seed] = run_dir
    return runs


@torch.no_grad()
def predict_clean_q(run_dir, loader, device):
    """在干净轨迹上返回最终软聚类分配与真实标签。"""
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
    all_q = []
    all_labels = []
    for batch in loader:
        view = {
            key: batch[key].to(device)
            for key in (
                "location_ids",
                "time_ids",
                "attention_mask",
                "pooling_mask",
            )
        }
        all_q.append(model.clustering_layer(model.encode(view)).cpu())
        all_labels.append(batch["label"].cpu())
    return torch.cat(all_q), torch.cat(all_labels)


def hungarian_mapping(raw_predictions, labels, num_clusters):
    contingency = np.zeros((num_clusters, num_clusters), dtype=np.int64)
    np.add.at(contingency, (raw_predictions, labels), 1)
    rows, columns = linear_sum_assignment(-contingency)
    return {int(row): int(column) for row, column in zip(rows, columns)}


def mapped_predictions(raw_predictions, mapping):
    return np.array(
        [mapping[int(prediction)] for prediction in raw_predictions],
        dtype=np.int64,
    )


def normalize_distribution(values):
    values = values.float().cpu()
    return values / values.sum(dim=1, keepdim=True).clamp_min(EPS)


def js_divergence(left, right):
    """逐样本计算两个概率分布的 JS 散度。"""
    left = normalize_distribution(left)
    right = normalize_distribution(right)
    middle = 0.5 * (left + right)
    left_kl = (
        left
        * (left.clamp_min(EPS).log() - middle.clamp_min(EPS).log())
    ).sum(dim=1)
    right_kl = (
        right
        * (right.clamp_min(EPS).log() - middle.clamp_min(EPS).log())
    ).sum(dim=1)
    return 0.5 * (left_kl + right_kl)


def compute_entropy(q):
    q = normalize_distribution(q)
    return -(q * q.clamp_min(EPS).log()).sum(dim=1)


def validate_pure_ema_history(item, run_dir):
    """避免把加权或目标插值实验混入当前机制分析。"""
    sample_weight = item.get("sample_weight")
    if sample_weight is None or not torch.allclose(
        sample_weight.float(),
        torch.ones_like(sample_weight, dtype=torch.float32),
    ):
        raise ValueError(f"EMA run must be unweighted: {run_dir}")
    target_mix_alpha = item.get("target_mix_alpha")
    if target_mix_alpha is not None and not torch.allclose(
        target_mix_alpha.float(),
        torch.ones_like(target_mix_alpha, dtype=torch.float32),
    ):
        raise ValueError(f"EMA run must use pure EMA targets: {run_dir}")


def load_sample_signals(run_dir, expected_samples):
    """计算每个样本在所有 EMA epoch 上的平均训练动态。"""
    files = sorted(Path(run_dir).glob("target_history/epoch_*.pt"))
    if not files:
        raise FileNotFoundError(f"target_history not found: {run_dir}")

    values = {
        "raw_js": [],
        "margin": [],
        "entropy": [],
        "ema_impact": [],
        "target_flip": [],
    }
    epochs = []
    for file in files:
        item = torch.load(file, map_location="cpu", weights_only=True)
        if not bool(item.get("ema_enabled", False)):
            continue
        validate_pure_ema_history(item, run_dir)

        q1 = item["q1"].float()
        q2 = item["q2"].float()
        if q1.size(0) != expected_samples:
            raise ValueError(
                f"history sample count differs in {run_dir}: "
                f"{q1.size(0)} != {expected_samples}"
            )

        current_p1 = target_distribution(q1)
        current_p2 = target_distribution(q2)
        train_p1 = item["train_p1"].float()
        train_p2 = item["train_p2"].float()
        ema_impact = 0.5 * (
            js_divergence(current_p1, train_p1)
            + js_divergence(current_p2, train_p2)
        )
        target_flip = (
            (current_p1.argmax(dim=1) != train_p1.argmax(dim=1))
            | (current_p2.argmax(dim=1) != train_p2.argmax(dim=1))
        ).float()

        epoch_values = {
            "raw_js": item["raw_js_divergence"].float(),
            "margin": item["margin_confidence"].float(),
            "entropy": compute_entropy(0.5 * (q1 + q2)),
            "ema_impact": ema_impact,
            "target_flip": target_flip,
        }
        if not all(torch.isfinite(value).all() for value in epoch_values.values()):
            raise ValueError(f"non-finite history values found: {file}")
        for key, value in epoch_values.items():
            values[key].append(value)
        epochs.append(int(item["epoch"]))

    if not epochs:
        raise ValueError(f"no finite EMA epoch found: {run_dir}")

    result = {
        key: torch.stack(epoch_values).mean(dim=0).numpy()
        for key, epoch_values in values.items()
    }
    result["epochs"] = epochs
    return result


def final_prediction_result(q, labels, percentile_mask):
    """返回 Hungarian 映射预测、软误差及指定样本内的误差百分位。"""
    raw_predictions = q.argmax(dim=1).numpy()
    mapping = hungarian_mapping(raw_predictions, labels, q.size(1))
    predictions = mapped_predictions(raw_predictions, mapping)

    inverse_mapping = {label: cluster for cluster, label in mapping.items()}
    true_clusters = np.array(
        [inverse_mapping[int(label)] for label in labels],
        dtype=np.int64,
    )
    true_probabilities = q.numpy()[np.arange(len(labels)), true_clusters]
    errors = -np.log(np.clip(true_probabilities, EPS, 1.0))
    percentiles = np.full(len(labels), np.nan, dtype=np.float64)
    valid_errors = errors[percentile_mask]
    ranks = rankdata(valid_errors, method="average")
    if len(ranks) == 1:
        percentiles[percentile_mask] = 0.0
    else:
        percentiles[percentile_mask] = (ranks - 1.0) / (len(ranks) - 1.0)
    return {
        "correct": predictions == labels,
        "error_percentile": percentiles,
    }


def analyze_seed(seed, baseline_run, ema_run, loader, device, excluded_labels):
    baseline_q, baseline_labels = predict_clean_q(baseline_run, loader, device)
    ema_q, ema_labels = predict_clean_q(ema_run, loader, device)
    if not torch.equal(baseline_labels, ema_labels):
        raise ValueError(f"dataset order differs for seed={seed}")

    labels = ema_labels.numpy().astype(np.int64)
    included = ~np.isin(labels, np.asarray(excluded_labels, dtype=np.int64))
    baseline = final_prediction_result(baseline_q, labels, included)
    ema = final_prediction_result(ema_q, labels, included)
    signals = load_sample_signals(ema_run, expected_samples=len(labels))
    impact_threshold = float(
        np.quantile(signals["ema_impact"], HIGH_IMPACT_QUANTILE)
    )
    high_impact = signals["ema_impact"] >= impact_threshold

    baseline_correct = baseline["correct"]
    ema_correct = ema["correct"]
    groups = {
        "都正确": included & baseline_correct & ema_correct,
        "EMA修复": included & (~baseline_correct) & ema_correct,
        "EMA新增错误": included & baseline_correct & (~ema_correct),
        "都错误": included & (~baseline_correct) & (~ema_correct),
        "最终正确": included & ema_correct,
        "最终错误": included & (~ema_correct),
    }
    return {
        "seed": seed,
        "labels": labels,
        "included": included,
        "baseline": baseline,
        "ema": ema,
        "signals": signals,
        "groups": groups,
        "impact_threshold": impact_threshold,
        "high_impact": high_impact,
    }


def finite_mean(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def mask_mean(values, mask):
    return float(np.asarray(values)[mask].mean()) if np.any(mask) else float("nan")


def across_seed(results, value_fn):
    return finite_mean([value_fn(result) for result in results])


def print_title(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def print_impact_distribution(results):
    print_title("1. 整体 EMA 影响分布")
    print("平均 EMA 影响：当前 DEC 目标与实际 EMA 训练目标的平均 JS 散度。")
    print("目标翻转率：EMA 改变样本首选目标簇的 epoch 比例。")
    print(f"{'分位点':8s} {'EMA影响':>12s}  含义")
    descriptions = {
        0.5: "约50%的样本低于它，约50%高于它",
        0.8: "约80%的样本低于它，约20%高于它",
        0.9: "约90%的样本低于它，约10%高于它",
    }
    for quantile in (0.5, 0.8, 0.9):
        value = across_seed(
            results,
            lambda result, q=quantile: np.quantile(
                result["signals"]["ema_impact"], q
            ),
        )
        print(f"P{int(quantile * 100):<7d} {value:12.5f}  {descriptions[quantile]}")


def print_outcome_impact(results):
    print_title("2. 按最终结果分组的 EMA 影响")
    print(
        f"{'样本组':14s} {'平均数量':>12s} {'平均EMA影响':>16s} "
        f"{'目标翻转率':>14s} {'高影响样本占比':>18s}"
    )
    for group in OUTCOME_GROUPS:
        count = across_seed(results, lambda r, g=group: r["groups"][g].sum())
        impact = across_seed(
            results,
            lambda r, g=group: mask_mean(
                r["signals"]["ema_impact"], r["groups"][g]
            ),
        )
        flip = across_seed(
            results,
            lambda r, g=group: mask_mean(
                r["signals"]["target_flip"], r["groups"][g]
            ),
        )
        high = across_seed(
            results,
            lambda r, g=group: mask_mean(r["high_impact"], r["groups"][g]),
        )
        print(
            f"{group:14s} {count:12.1f} {impact:16.5f} "
            f"{flip:13.2%} {high:17.2%}"
        )


def print_label_impact(results):
    print_title("3. 各 label 的 EMA 影响")
    labels = sorted(np.unique(results[0]["labels"]).tolist())
    print(
        f"{'Label':>5s} {'EMA影响':>12s} {'翻转率':>11s} {'高影响占比':>13s} "
        f"{'Baseline召回':>14s} {'EMA召回':>10s} {'变化':>9s}"
    )
    for label in labels:
        def label_mask(result, value=label):
            return result["labels"] == value

        impact = across_seed(
            results,
            lambda r: mask_mean(r["signals"]["ema_impact"], label_mask(r)),
        )
        flip = across_seed(
            results,
            lambda r: mask_mean(r["signals"]["target_flip"], label_mask(r)),
        )
        high = across_seed(
            results,
            lambda r: mask_mean(r["high_impact"], label_mask(r)),
        )
        baseline_recall = across_seed(
            results,
            lambda r: mask_mean(r["baseline"]["correct"], label_mask(r)),
        )
        ema_recall = across_seed(
            results,
            lambda r: mask_mean(r["ema"]["correct"], label_mask(r)),
        )
        print(
            f"{label:5d} {impact:12.5f} {flip:10.2%} {high:12.2%} "
            f"{baseline_recall:14.3f} {ema_recall:10.3f} "
            f"{ema_recall - baseline_recall:+9.3f}"
        )
    print("注：被排除 label 仍在本表展示，但不用于结果分组的主要结论。")


def print_signal_group_table(results, groups, title, high_impact_only=False):
    print_title(title)
    print(
        f"{'样本组':14s} {'平均数量':>12s} {'JS散度':>12s} {'Margin':>11s} "
        f"{'软分配熵':>12s} {'EMA影响':>12s} {'翻转率':>11s}"
    )
    for group in groups:
        def selected_mask(result, name=group):
            mask = result["groups"][name]
            return mask & result["high_impact"] if high_impact_only else mask

        count = across_seed(results, lambda r: selected_mask(r).sum())
        row = {}
        for key in ("raw_js", "margin", "entropy", "ema_impact", "target_flip"):
            row[key] = across_seed(
                results,
                lambda r, signal=key: mask_mean(
                    r["signals"][signal], selected_mask(r)
                ),
            )
        print(
            f"{group:14s} {count:12.1f} {row['raw_js']:12.5f} "
            f"{row['margin']:11.4f} {row['entropy']:12.4f} "
            f"{row['ema_impact']:12.5f} {row['target_flip']:10.2%}"
        )


def print_error_bins(results):
    print_title("7. 不同预测误差百分位区间的候选指标")
    print(
        f"{'误差区间':14s} {'平均数量':>12s} {'JS散度':>12s} {'Margin':>11s} "
        f"{'软分配熵':>12s} {'EMA影响':>12s} {'翻转率':>11s}"
    )
    for name, lower, upper in ERROR_BINS:
        def bin_mask(result, low=lower, high=upper):
            percentile = result["ema"]["error_percentile"]
            return result["included"] & (percentile >= low) & (percentile < high)

        count = across_seed(results, lambda r: bin_mask(r).sum())
        row = {}
        for key in ("raw_js", "margin", "entropy", "ema_impact", "target_flip"):
            row[key] = across_seed(
                results,
                lambda r, signal=key: mask_mean(
                    r["signals"][signal], bin_mask(r)
                ),
            )
        print(
            f"{name:14s} {count:12.1f} {row['raw_js']:12.5f} "
            f"{row['margin']:11.4f} {row['entropy']:12.4f} "
            f"{row['ema_impact']:12.5f} {row['target_flip']:10.2%}"
        )


def print_error_change(results):
    print_title("8. EMA 修复与新增错误的预测误差百分位变化")
    print(
        f"{'样本组':14s} {'平均数量':>12s} {'Baseline百分位':>17s} "
        f"{'EMA百分位':>14s} {'变化':>12s}"
    )
    for group in OUTCOME_GROUPS:
        count = across_seed(results, lambda r, g=group: r["groups"][g].sum())
        baseline_percentile = across_seed(
            results,
            lambda r, g=group: mask_mean(
                r["baseline"]["error_percentile"], r["groups"][g]
            ),
        )
        ema_percentile = across_seed(
            results,
            lambda r, g=group: mask_mean(
                r["ema"]["error_percentile"], r["groups"][g]
            ),
        )
        print(
            f"{group:14s} {count:12.1f} {baseline_percentile:16.2%} "
            f"{ema_percentile:13.2%} "
            f"{ema_percentile - baseline_percentile:+11.2%}"
        )


def print_summary(results, excluded_labels):
    print_impact_distribution(results)
    print_outcome_impact(results)
    print_label_impact(results)
    print_signal_group_table(
        results, ("最终正确", "最终错误"), "4. 候选指标与最终错误"
    )
    print_signal_group_table(
        results, ("EMA修复", "EMA新增错误"), "5. EMA 修复与新增错误"
    )
    print_signal_group_table(
        results,
        ("EMA修复", "EMA新增错误"),
        "6. EMA 影响最高 20% 样本中的修复与新增错误",
        high_impact_only=True,
    )
    print_error_bins(results)
    print_error_change(results)
    print("\n分析口径：")
    print(f"  结果分组和误差百分位排除 label：{list(excluded_labels)}")
    print("  各 label 表仍展示全部 label，仅用于描述，不解释映射互换。")
    print("  高影响样本为每个 seed 内平均 EMA 影响最高的 20%。")
    print("  所有表格先在每个 seed 内统计，再对 seed 结果取平均。")


def analyze(baseline_paths, ema_paths, batch_size, device_name, excluded_labels):
    baseline_runs = index_runs_by_seed(baseline_paths, "Baseline")
    ema_runs = index_runs_by_seed(ema_paths, "EMA")
    if set(baseline_runs) != set(ema_runs):
        raise ValueError(
            "Baseline and EMA seeds must match exactly: "
            f"baseline={sorted(baseline_runs)}, ema={sorted(ema_runs)}"
        )
    seeds = sorted(baseline_runs)

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    loader = DataLoader(
        QDTrajectoryDataset(),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    print("=" * 100)
    print("分阶段 EMA 样本级影响、候选指标与预测误差分析")
    print("=" * 100)
    print(f"设备：{device}")
    print(f"配对 seed：{seeds}")
    print(f"主要分组排除 label：{list(excluded_labels)}")

    results = []
    for seed in seeds:
        print(f"正在分析 seed={seed} ...")
        result = analyze_seed(
            seed,
            baseline_runs[seed],
            ema_runs[seed],
            loader,
            device,
            excluded_labels,
        )
        print(
            f"  EMA epochs={result['signals']['epochs']} | "
            f"P80影响阈值={result['impact_threshold']:.5f}"
        )
        results.append(result)
    print_summary(results, excluded_labels)


def parse_args():
    parser = argparse.ArgumentParser(
        description="分析分阶段 EMA 的样本级作用和最终预测误差",
    )
    parser.add_argument(
        "--baseline-runs",
        nargs="+",
        required=True,
        help="Baseline 实验父目录或具体 run 目录",
    )
    parser.add_argument(
        "--ema-runs",
        nargs="+",
        required=True,
        help="未加权、未插值的分阶段 EMA 实验父目录或具体 run 目录",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--exclude-labels",
        nargs="*",
        type=int,
        default=[1, 2],
        help="从结果分组和误差百分位中排除的 label，默认 1 2",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    analyze(
        args.baseline_runs,
        args.ema_runs,
        args.batch_size,
        args.device,
        tuple(args.exclude_labels),
    )


if __name__ == "__main__":
    main()
