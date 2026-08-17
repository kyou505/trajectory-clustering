"""分析 JS 稳定性、margin、软分配熵与最终聚类错误的关系。

分析对象必须是未进行 DEC 样本加权的分阶段 EMA 实验。脚本按 seed
读取 EMA run，并且只在 EMA 已启用的 epoch 上计算三个指标：

1. raw JS：当前软分配相对历史 EMA 分配的变化程度；
2. margin：当前软分配 top1 与 top2 的相对距离；
3. entropy：当前软分配在全部簇上的熵。

在 con-dtc 目录下运行：

    python -m src.analysis_ema_correlation \
        --runs runs/ema_staged/condtc_qd_ema_staged_s3_m07

参数既可以是包含多个时间戳 run 的实验目录，也可以是具体 run 目录。
结果只输出到终端，不创建文件。
"""

import argparse
import re
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

from src.data.data_process import QDTrajectoryDataset
from src.models.contrastive_model import ContrastiveTrajectoryModel


EPS = 1e-12
SIGNALS = (
    ("raw_js", "JS散度"),
    ("margin", "Margin"),
    ("entropy", "软分配熵"),
)


def discover_runs(paths):
    """将实验父目录或具体 run 目录展开为已完成的 run 列表。"""
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
            raise FileNotFoundError(
                f"no completed run found under: {path}"
            )
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
    """优先读取配置中的 seed，目录名仅作为兼容回退。"""
    config = read_run_config(run_dir)
    try:
        return int(config["training"]["seed"])
    except (KeyError, TypeError, ValueError):
        match = re.search(r"seed(-?\d+)$", Path(run_dir).name)
        if match is None:
            raise ValueError(f"cannot determine seed for run: {run_dir}")
        return int(match.group(1))


def index_runs_by_seed(paths, side_name):
    runs = {}
    for run_dir in discover_runs(paths):
        seed = read_run_seed(run_dir)
        if seed in runs:
            raise ValueError(
                f"duplicate {side_name} run for seed={seed}: "
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
        embeddings = model.encode(view)
        all_q.append(model.clustering_layer(embeddings).cpu())
        all_labels.append(batch["label"].cpu())
    return torch.cat(all_q), torch.cat(all_labels)


def hungarian_mapping(raw_predictions, labels, num_clusters):
    contingency = np.zeros(
        (num_clusters, num_clusters),
        dtype=np.int64,
    )
    np.add.at(contingency, (raw_predictions, labels), 1)
    rows, columns = linear_sum_assignment(-contingency)
    return {int(row): int(column) for row, column in zip(rows, columns)}


def mapped_predictions(raw_predictions, mapping):
    return np.array(
        [mapping[int(prediction)] for prediction in raw_predictions],
        dtype=np.int64,
    )


def normalize_q(q):
    q = q.float().cpu()
    return q / q.sum(dim=1, keepdim=True).clamp_min(EPS)


def compute_entropy(q):
    q = normalize_q(q)
    return -(q * q.clamp_min(EPS).log()).sum(dim=1)


def load_ema_epoch_signals(run_dir, expected_samples):
    """读取所有 EMA epoch，并返回每个样本三个指标的跨 epoch 均值。"""
    files = sorted(Path(run_dir).glob("target_history/epoch_*.pt"))
    if not files:
        raise FileNotFoundError(f"target_history not found: {run_dir}")

    raw_js_values = []
    margin_values = []
    entropy_values = []
    epochs = []
    for file in files:
        item = torch.load(file, map_location="cpu", weights_only=True)
        if not bool(item.get("ema_enabled", False)):
            continue

        q1 = item["q1"].float()
        q2 = item["q2"].float()
        if q1.size(0) != expected_samples:
            raise ValueError(
                f"history sample count differs in {run_dir}: "
                f"{q1.size(0)} != {expected_samples}"
            )

        # 相关性分析必须使用未加权、未进行目标插值的纯 EMA 实验。
        sample_weight = item.get("sample_weight")
        if sample_weight is None or not torch.allclose(
            sample_weight.float(),
            torch.ones_like(sample_weight, dtype=torch.float32),
        ):
            raise ValueError(
                f"correlation source must be unweighted: {run_dir}"
            )
        target_mix_alpha = item.get("target_mix_alpha")
        if target_mix_alpha is not None and not torch.allclose(
            target_mix_alpha.float(),
            torch.ones_like(target_mix_alpha, dtype=torch.float32),
        ):
            raise ValueError(
                f"correlation source must use pure EMA targets: {run_dir}"
            )

        raw_js = item["raw_js_divergence"].float()
        margin = item["margin_confidence"].float()
        consensus = 0.5 * (q1 + q2)
        entropy = compute_entropy(consensus)
        if not (
            torch.isfinite(raw_js).all()
            and torch.isfinite(margin).all()
            and torch.isfinite(entropy).all()
        ):
            continue

        raw_js_values.append(raw_js)
        margin_values.append(margin)
        entropy_values.append(entropy)
        epochs.append(int(item["epoch"]))

    if not epochs:
        raise ValueError(f"no finite EMA epoch found: {run_dir}")

    return {
        "epochs": epochs,
        "raw_js": torch.stack(raw_js_values).mean(dim=0).numpy(),
        "margin": torch.stack(margin_values).mean(dim=0).numpy(),
        "entropy": torch.stack(entropy_values).mean(dim=0).numpy(),
    }


def analyze_seed(seed, ema_run, loader, device):
    """完成一个 seed 的三个信号与最终错误分析。"""
    ema_q, ema_labels = predict_clean_q(ema_run, loader, device)
    labels = ema_labels.numpy().astype(np.int64)
    ema_raw_predictions = ema_q.argmax(dim=1).numpy()
    mapping = hungarian_mapping(
        ema_raw_predictions,
        labels,
        num_clusters=ema_q.size(1),
    )
    final_predictions = mapped_predictions(ema_raw_predictions, mapping)
    final_error = final_predictions != labels

    signals = load_ema_epoch_signals(ema_run, expected_samples=len(labels))

    rows = []
    for key, display_name in SIGNALS:
        values = signals[key]
        rows.append({
            "seed": seed,
            "signal": key,
            "display_name": display_name,
            "correct_mean": float(values[~final_error].mean()),
            "wrong_mean": float(values[final_error].mean()),
        })
    return {
        "seed": seed,
        "epochs": signals["epochs"],
        "samples": len(labels),
        "correct": int((~final_error).sum()),
        "wrong": int(final_error.sum()),
        "uacc": float((~final_error).mean()),
        "rows": rows,
    }


def mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return "nan"
    if values.size == 1:
        return f"{values[0]:.4f}"
    return f"{values.mean():.4f} ± {values.std(ddof=1):.4f}"


def print_seed_result(result):
    print("\n" + "-" * 96)
    print(
        f"seed={result['seed']} | EMA epochs={result['epochs']} | "
        f"UACC={result['uacc']:.4f} | "
        f"正确={result['correct']} | 错误={result['wrong']}"
    )
    print(f"{'指标':16s} {'正确组均值':>14s} {'错误组均值':>14s}")
    for row in result["rows"]:
        print(
            f"{row['display_name']:16s} "
            f"{row['correct_mean']:14.4f} "
            f"{row['wrong_mean']:14.4f}"
        )


def print_summary(results):
    print("\n" + "=" * 118)
    print("多 seed 汇总：三个指标与最终错误的关系")
    print("=" * 118)
    print(f"{'指标':16s} {'正确组均值':>22s} {'错误组均值':>22s}")
    for key, display_name in SIGNALS:
        rows = [
            row
            for result in results
            for row in result["rows"]
            if row["signal"] == key
        ]
        print(
            f"{display_name:16s} "
            f"{mean_std([row['correct_mean'] for row in rows]):>22s} "
            f"{mean_std([row['wrong_mean'] for row in rows]):>22s}"
        )
    print("\n方向说明：")
    print("  JS 越大表示跨 epoch 的分配变化越大。")
    print("  Margin 越小表示越接近聚类决策边界。")
    print("  熵越大表示当前软分配越不确定。")


def analyze(run_paths, batch_size, device_name):
    runs_by_seed = index_runs_by_seed(run_paths, "EMA")
    runs = sorted(runs_by_seed.items())
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    dataset = QDTrajectoryDataset()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    print("=" * 96)
    print("JS稳定性、Margin、软分配熵与最终聚类错误的相关性分析")
    print("=" * 96)
    print(f"设备: {device}")
    print(f"实验 seed: {[seed for seed, _ in runs]}")

    results = []
    for seed, ema_run in runs:
        print(f"正在分析 seed={seed} ...")
        result = analyze_seed(
            seed,
            ema_run,
            loader,
            device,
        )
        results.append(result)
        print_seed_result(result)
    print_summary(results)


def parse_args():
    parser = argparse.ArgumentParser(
        description="分析 JS、margin、熵与最终聚类错误的关系",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="未加权分阶段 EMA 实验父目录或具体 run 目录",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    analyze(
        run_paths=args.runs,
        batch_size=args.batch_size,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
