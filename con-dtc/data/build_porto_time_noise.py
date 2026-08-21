"""从 Porto 原始 train.csv 构造 ConDTC 细粒度聚类数据。

论文明确的设置：9 条基础轨迹、每条轨迹生成原始/时移/变速三类、
50 米空间高斯噪声、正负 3 分钟时间噪声、Geohash-7。

论文未公开的基础轨迹、丢点率、时移量、速度倍率和簇大小由命令行参数
显式控制，并写入 generation_config.json。因此该数据集属于论文方法的可复现
重建，不等同于作者未公开的 Porto 实验数据。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


SPECIAL_TOKENS = {
    "[PAD]": 0,
    "[CLS]": 1,
    "[SEP]": 2,
    "[MASK]": 3,
}
GEOHASH_ALPHABET = "0123456789bcdefghjkmnpqrstuvwxyz"
SECONDS_PER_RAW_TIME_SLOT = 15
RAW_TIME_SLOTS_PER_DAY = 24 * 60 * 60 // SECONDS_PER_RAW_TIME_SLOT


def parse_args():
    data_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Build a paper-aligned Porto fine-grained clustering dataset"
    )
    parser.add_argument("--input", type=Path, default=data_dir / "train.csv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=data_dir / "portoTimeNoiseReconstructed",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-base-trajectories", type=int, default=9)
    parser.add_argument("--total-samples", type=int, default=27009)
    parser.add_argument("--min-base-length", type=int, default=30)
    parser.add_argument("--max-base-length", type=int, default=58)
    parser.add_argument("--candidate-pool-size", type=int, default=5000)
    parser.add_argument("--dropout-rate", type=float, default=0.1)
    parser.add_argument("--spatial-noise-meters", type=float, default=50.0)
    parser.add_argument("--time-noise-minutes", type=float, default=3.0)
    parser.add_argument("--time-shift-minutes", type=float, default=60.0)
    parser.add_argument(
        "--speed-scale",
        type=float,
        default=0.5,
        help="速度倍率；0.5 表示用时变为原来的 2 倍",
    )
    parser.add_argument("--geohash-precision", type=int, default=7)
    parser.add_argument("--max-sequence-length", type=int, default=60)
    return parser.parse_args()


def encode_geohash(longitude, latitude, precision=7):
    """无额外依赖的标准 Geohash 编码。"""
    longitude_interval = [-180.0, 180.0]
    latitude_interval = [-90.0, 90.0]
    bits = (16, 8, 4, 2, 1)
    bit_index = 0
    character = 0
    even_bit = True
    result = []

    while len(result) < precision:
        interval = longitude_interval if even_bit else latitude_interval
        value = longitude if even_bit else latitude
        midpoint = 0.5 * (interval[0] + interval[1])
        if value >= midpoint:
            character |= bits[bit_index]
            interval[0] = midpoint
        else:
            interval[1] = midpoint
        even_bit = not even_bit
        if bit_index < 4:
            bit_index += 1
        else:
            result.append(GEOHASH_ALPHABET[character])
            bit_index = 0
            character = 0
    return "".join(result)


def valid_porto_polyline(polyline, min_length, max_length):
    if not min_length <= len(polyline) <= max_length:
        return False
    coordinates = np.asarray(polyline, dtype=np.float64)
    if coordinates.shape != (len(polyline), 2):
        return False
    longitude = coordinates[:, 0]
    latitude = coordinates[:, 1]
    return bool(
        np.isfinite(coordinates).all()
        and ((-8.75 <= longitude) & (longitude <= -8.50)).all()
        and ((41.05 <= latitude) & (latitude <= 41.25)).all()
    )


def collect_candidate_bases(
        input_path,
        pool_size,
        min_length,
        max_length,
        rng,
):
    """对合格轨迹做蓄水池采样，避免把 1.8 GB CSV 全部载入内存。"""
    candidates = []
    eligible_count = 0
    with input_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required = {"TRIP_ID", "TIMESTAMP", "MISSING_DATA", "POLYLINE"}
        missing_columns = required - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"input CSV is missing columns: {sorted(missing_columns)}")

        for row in reader:
            if row["MISSING_DATA"].strip().lower() == "true":
                continue
            try:
                polyline = json.loads(row["POLYLINE"])
            except (TypeError, json.JSONDecodeError):
                continue
            if not valid_porto_polyline(polyline, min_length, max_length):
                continue

            candidate = {
                "trip_id": row["TRIP_ID"],
                "timestamp": int(row["TIMESTAMP"]),
                "polyline": np.asarray(polyline, dtype=np.float64),
            }
            eligible_count += 1
            if len(candidates) < pool_size:
                candidates.append(candidate)
            else:
                replacement = int(rng.integers(eligible_count))
                if replacement < pool_size:
                    candidates[replacement] = candidate

    if len(candidates) < pool_size:
        print(
            f"warning: requested {pool_size} candidates, "
            f"but only {len(candidates)} were available"
        )
    print("eligible base trajectories:", eligible_count)
    print("candidate pool size:", len(candidates))
    return candidates


def resample_polyline(polyline, num_points=32):
    """按累计路程重采样，用于比较不同长度的路线。"""
    segment_lengths = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    if cumulative[-1] <= 0:
        return np.repeat(polyline[:1], num_points, axis=0)
    targets = np.linspace(0.0, cumulative[-1], num_points)
    longitude = np.interp(targets, cumulative, polyline[:, 0])
    latitude = np.interp(targets, cumulative, polyline[:, 1])
    return np.column_stack((longitude, latitude))


def select_diverse_bases(candidates, num_bases, rng):
    """固定随机起点后用最远点采样选择空间路径不同的基础轨迹。"""
    if len(candidates) < num_bases:
        raise ValueError("candidate pool is smaller than num-base-trajectories")
    features = np.stack(
        [resample_polyline(item["polyline"]).reshape(-1) for item in candidates]
    )
    # 经度按 Porto 纬度修正，使经纬方向近似处于同一距离尺度。
    features[:, 0::2] *= math.cos(math.radians(41.15))

    selected = [int(rng.integers(len(candidates)))]
    minimum_distance = np.full(len(candidates), np.inf)
    while len(selected) < num_bases:
        last = features[selected[-1]]
        distance = np.sqrt(np.mean((features - last) ** 2, axis=1))
        minimum_distance = np.minimum(minimum_distance, distance)
        minimum_distance[selected] = -1.0
        selected.append(int(np.argmax(minimum_distance)))
    return [candidates[index] for index in selected]


def add_spatial_noise(polyline, sigma_meters, rng):
    latitude = polyline[:, 1]
    latitude_noise = rng.normal(0.0, sigma_meters, len(polyline)) / 111_320.0
    longitude_scale = 111_320.0 * np.cos(np.radians(latitude))
    longitude_noise = rng.normal(0.0, sigma_meters, len(polyline)) / longitude_scale
    noisy = polyline.copy()
    noisy[:, 0] += longitude_noise
    noisy[:, 1] += latitude_noise
    return noisy


def random_drop(polyline, elapsed_seconds, dropout_rate, rng):
    keep = rng.random(len(polyline)) >= dropout_rate
    # 保留首尾点，并保证至少两个点。
    keep[0] = True
    keep[-1] = True
    return polyline[keep], elapsed_seconds[keep]


def transform_time(elapsed_seconds, mode, shift_minutes, speed_scale):
    elapsed_seconds = elapsed_seconds.astype(np.float64)
    if mode == "original":
        return elapsed_seconds
    if mode == "time_shift":
        return elapsed_seconds + shift_minutes * 60.0
    if mode == "speed_scale":
        if speed_scale <= 0:
            raise ValueError("speed-scale must be positive")
        return elapsed_seconds / speed_scale
    raise ValueError(f"unknown temporal mode: {mode}")


def add_monotonic_time_noise(elapsed_seconds, max_noise_minutes, rng):
    noise_seconds = rng.uniform(
        -max_noise_minutes * 60.0,
        max_noise_minutes * 60.0,
        len(elapsed_seconds),
    )
    noisy = elapsed_seconds + noise_seconds
    return np.maximum.accumulate(noisy)


def allocate_cluster_sizes(total_samples, num_clusters):
    sizes = np.full(num_clusters, total_samples // num_clusters, dtype=np.int64)
    sizes[: total_samples % num_clusters] += 1
    return sizes


def build_rows(bases, args, rng):
    modes = ("original", "time_shift", "speed_scale")
    num_clusters = len(bases) * len(modes)
    cluster_sizes = allocate_cluster_sizes(args.total_samples, num_clusters)
    rows = []

    for base_index, base in enumerate(bases):
        base_polyline = base["polyline"]
        base_elapsed = np.arange(len(base_polyline)) * SECONDS_PER_RAW_TIME_SLOT
        start_second = base["timestamp"] % (24 * 60 * 60)

        for mode_index, mode in enumerate(modes):
            label = base_index * len(modes) + mode_index
            for _ in range(int(cluster_sizes[label])):
                polyline, elapsed = random_drop(
                    base_polyline,
                    base_elapsed,
                    args.dropout_rate,
                    rng,
                )
                polyline = add_spatial_noise(
                    polyline,
                    args.spatial_noise_meters,
                    rng,
                )
                elapsed = transform_time(
                    elapsed,
                    mode,
                    args.time_shift_minutes,
                    args.speed_scale,
                )
                elapsed = add_monotonic_time_noise(
                    elapsed,
                    args.time_noise_minutes,
                    rng,
                )
                absolute_seconds = start_second + elapsed
                raw_time = np.floor_divide(
                    absolute_seconds.astype(np.int64),
                    SECONDS_PER_RAW_TIME_SLOT,
                ) % RAW_TIME_SLOTS_PER_DAY

                geohashes = [
                    encode_geohash(lon, lat, args.geohash_precision)
                    for lon, lat in polyline
                ]
                rows.append({
                    "traj": polyline.tolist(),
                    "time_values": raw_time.astype(int).tolist(),
                    "label": label,
                    "trajLen": len(polyline),
                    "trajHash": geohashes,
                    "duration": float((elapsed[-1] - elapsed[0]) / 60.0),
                    "base_trip_id": base["trip_id"],
                    "temporal_mode": mode,
                })
    return rows


def encode_and_pad_rows(rows, max_sequence_length):
    geohashes = sorted({token for row in rows for token in row["trajHash"]})
    raw_location_id = {token: index for index, token in enumerate(geohashes)}
    pad_count_error = 0
    for row in rows:
        if row["trajLen"] > max_sequence_length:
            pad_count_error += 1
            continue
        pad_count = max_sequence_length - row["trajLen"]
        locations = [str(raw_location_id[token]) for token in row["trajHash"]]
        times = [str(value) for value in row.pop("time_values")]
        row["trajectory"] = " ".join(locations + ["[PAD]"] * pad_count)
        row["time"] = " ".join(times + ["[PAD]"] * pad_count)
    if pad_count_error:
        raise ValueError(
            f"{pad_count_error} generated trajectories exceed max-sequence-length"
        )
    location_vocab = SPECIAL_TOKENS.copy()
    for raw_id in range(len(geohashes)):
        location_vocab[str(raw_id)] = len(location_vocab)
    return location_vocab


def validate_args(args):
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.num_base_trajectories <= 0:
        raise ValueError("num-base-trajectories must be positive")
    if args.total_samples < args.num_base_trajectories * 3:
        raise ValueError("total-samples is too small")
    if not 0 <= args.dropout_rate < 1:
        raise ValueError("dropout-rate must be in [0, 1)")
    if args.max_base_length > args.max_sequence_length:
        raise ValueError("max-base-length cannot exceed max-sequence-length")


def save_dataset(rows, bases, location_vocab, args):
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    dataframe = pd.DataFrame(rows)[
        [
            "traj",
            "time",
            "label",
            "trajLen",
            "trajHash",
            "trajectory",
            "duration",
            "base_trip_id",
            "temporal_mode",
        ]
    ]
    dataframe.to_hdf(output_dir / "data_k3.h5", key="x", mode="w")
    pd.DataFrame({"length": dataframe["trajLen"]}).to_csv(
        output_dir / "trj_length.csv",
        index=False,
    )
    with (output_dir / "location_vocab.json").open("w", encoding="utf-8") as file:
        json.dump(location_vocab, file, ensure_ascii=False, indent=2)

    metadata = {
        "source": str(args.input.resolve()),
        "paper_aligned_reconstruction": True,
        "seed": args.seed,
        "num_base_trajectories": len(bases),
        "base_trip_ids": [item["trip_id"] for item in bases],
        "num_clusters": len(bases) * 3,
        "total_samples": len(dataframe),
        "cluster_sizes": {
            str(label): int(count)
            for label, count in dataframe["label"].value_counts().sort_index().items()
        },
        "temporal_modes": ["original", "time_shift", "speed_scale"],
        "dropout_rate": args.dropout_rate,
        "spatial_noise_meters": args.spatial_noise_meters,
        "time_noise_minutes": args.time_noise_minutes,
        "time_shift_minutes": args.time_shift_minutes,
        "speed_scale": args.speed_scale,
        "geohash_precision": args.geohash_precision,
        "max_sequence_length": args.max_sequence_length,
        "location_vocab_size": len(location_vocab),
        "raw_location_count": len(location_vocab) - len(SPECIAL_TOKENS),
        "trajectory_length": {
            "min": int(dataframe["trajLen"].min()),
            "mean": float(dataframe["trajLen"].mean()),
            "max": int(dataframe["trajLen"].max()),
        },
        "notes": {
            "paper_specified": [
                "9 base trajectories",
                "3 temporal modes per base trajectory",
                "50 meter Gaussian spatial noise",
                "plus/minus 3 minute temporal noise",
                "Geohash precision 7",
            ],
            "reconstruction_choices": [
                "selected base trajectory IDs",
                "dropout rate",
                "time shift",
                "speed scale",
                "cluster sizes",
            ],
        },
    }
    with (output_dir / "generation_config.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    return dataframe, metadata


def main():
    args = parse_args()
    validate_args(args)
    rng = np.random.default_rng(args.seed)
    candidates = collect_candidate_bases(
        input_path=args.input,
        pool_size=args.candidate_pool_size,
        min_length=args.min_base_length,
        max_length=args.max_base_length,
        rng=rng,
    )
    bases = select_diverse_bases(
        candidates,
        args.num_base_trajectories,
        rng,
    )
    print("selected base trip IDs:")
    for index, base in enumerate(bases):
        print(f"  path {index}: {base['trip_id']} (length={len(base['polyline'])})")

    rows = build_rows(bases, args, rng)
    location_vocab = encode_and_pad_rows(rows, args.max_sequence_length)
    dataframe, metadata = save_dataset(rows, bases, location_vocab, args)
    print("saved dataset:", args.output_dir.resolve())
    print("shape:", dataframe.shape)
    print("clusters:", metadata["cluster_sizes"])
    print("location vocab size:", metadata["location_vocab_size"])
    print("trajectory length:", metadata["trajectory_length"])


if __name__ == "__main__":
    main()
