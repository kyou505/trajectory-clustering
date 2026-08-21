"""从 Porto 原始 train.csv 构造 ConDTC 细粒度聚类数据。

论文明确的设置：9 条基础轨迹、每条轨迹生成原始/时移/变速三类、
50 米空间高斯噪声、正负 3 分钟时间噪声、Geohash-7。

公开 QD 数据表明，保存坐标确实参与了 Geohash 编码，但空间扰动在 token
层面远小于论文所称的 50 米：全数据只有 144 个位置 ID。因此默认使用 1 米
的“有效空间噪声”重建公开数据行为，同时在元数据中保留论文声明的 50 米。

论文未公开的基础轨迹、丢点率、时移量、速度倍率和簇大小由命令行参数
显式控制，并写入 generation_config.json。因此该数据集属于论文方法的可复现
重建，不等同于作者未公开的 Porto 实验数据。
"""

from __future__ import annotations

import argparse
import csv
import json
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
        default=data_dir / "portoTimeNoiseFineGrained",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-base-trajectories", type=int, default=9)
    parser.add_argument("--min-cluster-size", type=int, default=900)
    parser.add_argument("--max-cluster-size", type=int, default=1100)
    # Porto 公开 q/d/o 数据的轨迹平均约 50 点；限制在这一长度区间，
    # 既保留随机选取，又避免抽到大量明显偏短的基础轨迹。
    parser.add_argument("--min-base-length", type=int, default=45)
    parser.add_argument("--max-base-length", type=int, default=58)
    parser.add_argument(
        "--min-base-distinct-locations",
        type=int,
        default=40,
        help=(
            "基础轨迹至少覆盖的不同 Geohash 数；用于排除长时间停留或 "
            "空间覆盖不足的轨迹"
        ),
    )
    parser.add_argument("--candidate-pool-size", type=int, default=5000)
    parser.add_argument(
        "--base-selection",
        choices=("random", "shared_subpath"),
        default="shared_subpath",
    )
    parser.add_argument(
        "--min-shared-subpath-ratio",
        type=float,
        default=0.25,
        help="共享位置数 / 两条路线较短者位置数的下限",
    )
    parser.add_argument(
        "--max-shared-subpath-ratio",
        type=float,
        default=0.75,
        help="共享位置数 / 两条路线较短者位置数的上限，避免近重复路线",
    )
    parser.add_argument("--dropout-rate", type=float, default=0.1)
    parser.add_argument(
        "--spatial-noise-meters",
        type=float,
        default=1.0,
        help=(
            "实际用于坐标的高斯噪声标准差；默认 1m 用于匹配公开 QD "
            "数据中空间噪声基本不改变 Geohash token 的行为"
        ),
    )
    parser.add_argument("--time-noise-minutes", type=float, default=3.0)
    parser.add_argument("--time-shift-minutes", type=float, default=10.0)
    parser.add_argument(
        "--speed-scale",
        type=float,
        default=0.8,
        help="速度倍率；0.8 表示用时变为原来的 1.25 倍",
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


def valid_porto_polyline(
        polyline,
        min_length,
        max_length,
        min_distinct_locations,
        geohash_precision,
):
    if not min_length <= len(polyline) <= max_length:
        return False
    coordinates = np.asarray(polyline, dtype=np.float64)
    if coordinates.shape != (len(polyline), 2):
        return False
    longitude = coordinates[:, 0]
    latitude = coordinates[:, 1]
    coordinates_valid = bool(
        np.isfinite(coordinates).all()
        and ((-8.75 <= longitude) & (longitude <= -8.50)).all()
        and ((41.05 <= latitude) & (latitude <= 41.25)).all()
    )
    if not coordinates_valid:
        return False
    distinct_locations = {
        encode_geohash(lon, lat, geohash_precision)
        for lon, lat in coordinates
    }
    return len(distinct_locations) >= min_distinct_locations


def collect_candidate_bases(
        input_path,
        pool_size,
        min_length,
        max_length,
        min_distinct_locations,
        geohash_precision,
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
            if not valid_porto_polyline(
                polyline,
                min_length,
                max_length,
                min_distinct_locations,
                geohash_precision,
            ):
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


def select_random_bases(candidates, num_bases, rng):
    """按照论文描述，从合格候选轨迹中进行固定种子的随机选择。"""
    if len(candidates) < num_bases:
        raise ValueError("candidate pool is smaller than num-base-trajectories")
    selected = rng.choice(len(candidates), size=num_bases, replace=False)
    return [candidates[int(index)] for index in selected]


def route_geohashes(candidate, precision):
    return {
        encode_geohash(lon, lat, precision)
        for lon, lat in candidate["polyline"]
    }


def shared_subpath_ratio(left, right):
    denominator = min(len(left), len(right))
    return len(left & right) / denominator if denominator else 0.0


def select_shared_subpath_bases(
        candidates,
        num_bases,
        precision,
        min_ratio,
        max_ratio,
        rng,
):
    """选择共享中等比例路径的位置轨迹。

    共享段可以位于起点、中段或终点，不要求轨迹具有相同起点。先从候选池
    随机抽取一组锚点进行搜索，再选择拥有最多中等重叠邻居的锚点。最终从
    合格邻居中随机选择，避免恢复成“最相似轨迹”或“最远轨迹”采样。
    """
    if len(candidates) < num_bases:
        raise ValueError("candidate pool is smaller than num-base-trajectories")

    route_sets = [route_geohashes(item, precision) for item in candidates]
    best_anchor = None
    best_neighbors = []
    anchor_count = min(256, len(candidates))
    anchor_indices = rng.choice(
        len(candidates),
        size=anchor_count,
        replace=False,
    )
    for anchor_value in anchor_indices:
        anchor = int(anchor_value)
        neighbors = []
        for index in range(len(candidates)):
            if index == anchor:
                continue
            ratio = shared_subpath_ratio(
                route_sets[anchor],
                route_sets[index],
            )
            if min_ratio <= ratio <= max_ratio:
                neighbors.append((index, ratio))
        if len(neighbors) > len(best_neighbors):
            best_anchor = anchor
            best_neighbors = neighbors

    if best_anchor is None or len(best_neighbors) < num_bases - 1:
        raise ValueError(
            "cannot find enough shared-subpath bases; "
            "lower min-shared-subpath-ratio or enlarge candidate-pool-size"
        )

    selected_positions = rng.choice(
        len(best_neighbors),
        size=num_bases - 1,
        replace=False,
    )
    selected_indices = [best_anchor] + [
        best_neighbors[int(position)][0]
        for position in selected_positions
    ]
    return [candidates[index] for index in selected_indices]


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


def transform_time(absolute_seconds, mode, shift_minutes, speed_scale):
    """在 addTimeNoise 之后执行 timeShift 或 speedScale。"""
    absolute_seconds = absolute_seconds.astype(np.float64)
    if mode == "original":
        return absolute_seconds
    if mode == "time_shift":
        return absolute_seconds + shift_minutes * 60.0
    if mode == "speed_scale":
        if speed_scale <= 0:
            raise ValueError("speed-scale must be positive")
        start = absolute_seconds[0]
        return start + (absolute_seconds - start) / speed_scale
    raise ValueError(f"unknown temporal mode: {mode}")


def add_monotonic_time_noise(elapsed_seconds, max_noise_minutes, rng):
    noise_seconds = rng.uniform(
        -max_noise_minutes * 60.0,
        max_noise_minutes * 60.0,
        len(elapsed_seconds),
    )
    noisy = elapsed_seconds + noise_seconds
    return np.maximum.accumulate(noisy)


def sample_cluster_sizes(num_clusters, min_size, max_size, rng):
    """对应 Algorithm 1 中每个簇独立执行 randint(minN, maxN)。"""
    return rng.integers(
        low=min_size,
        high=max_size + 1,
        size=num_clusters,
    )


def build_rows(bases, cluster_sizes, args, rng):
    modes = ("original", "time_shift", "speed_scale")
    rows = []

    for base_index, base in enumerate(bases):
        base_polyline = base["polyline"]
        start_second = base["timestamp"] % (24 * 60 * 60)
        base_absolute_seconds = (
            start_second
            + np.arange(len(base_polyline)) * SECONDS_PER_RAW_TIME_SLOT
        )

        for mode_index, mode in enumerate(modes):
            label = base_index * len(modes) + mode_index
            for _ in range(int(cluster_sizes[label])):
                polyline, absolute_seconds = random_drop(
                    base_polyline,
                    base_absolute_seconds,
                    args.dropout_rate,
                    rng,
                )
                polyline = add_spatial_noise(
                    polyline,
                    args.spatial_noise_meters,
                    rng,
                )
                # Algorithm 1：先 addTimeNoise，再 timeShift/speedScale。
                absolute_seconds = add_monotonic_time_noise(
                    absolute_seconds,
                    args.time_noise_minutes,
                    rng,
                )
                absolute_seconds = transform_time(
                    absolute_seconds,
                    mode,
                    args.time_shift_minutes,
                    args.speed_scale,
                )
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
                    "duration": float(
                        (absolute_seconds[-1] - absolute_seconds[0]) / 60.0
                    ),
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
    if args.min_cluster_size <= 0:
        raise ValueError("min-cluster-size must be positive")
    if args.max_cluster_size < args.min_cluster_size:
        raise ValueError("max-cluster-size must be no smaller than min-cluster-size")
    if not 0 <= args.dropout_rate < 1:
        raise ValueError("dropout-rate must be in [0, 1)")
    if args.max_base_length > args.max_sequence_length:
        raise ValueError("max-base-length cannot exceed max-sequence-length")
    if args.min_base_distinct_locations <= 0:
        raise ValueError("min-base-distinct-locations must be positive")
    if args.min_base_distinct_locations > args.max_base_length:
        raise ValueError(
            "min-base-distinct-locations cannot exceed max-base-length"
        )
    if not (
        0.0
        <= args.min_shared_subpath_ratio
        <= args.max_shared_subpath_ratio
        <= 1.0
    ):
        raise ValueError(
            "shared-subpath ratios must satisfy 0 <= min <= max <= 1"
        )


def summarize_base_overlap(bases, precision):
    route_sets = [route_geohashes(item, precision) for item in bases]
    ratios = []
    same_origin_pairs = 0
    pair_count = 0
    origins = []
    for base in bases:
        lon, lat = base["polyline"][0]
        origins.append(encode_geohash(lon, lat, precision))
    for left in range(len(bases)):
        for right in range(left + 1, len(bases)):
            ratios.append(shared_subpath_ratio(route_sets[left], route_sets[right]))
            same_origin_pairs += origins[left] == origins[right]
            pair_count += 1
    values = np.asarray(ratios, dtype=np.float64)
    return {
        "same_origin_pair_ratio": (
            same_origin_pairs / pair_count if pair_count else 0.0
        ),
        "shared_subpath_ratio": {
            "min": float(values.min()) if len(values) else 0.0,
            "mean": float(values.mean()) if len(values) else 0.0,
            "max": float(values.max()) if len(values) else 0.0,
        },
    }


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

    base_geohashes = {
        encode_geohash(lon, lat, args.geohash_precision)
        for base in bases
        for lon, lat in base["polyline"]
    }
    raw_location_count = len(location_vocab) - len(SPECIAL_TOKENS)
    metadata = {
        "dataset_revision": 3,
        "source": str(args.input.resolve()),
        "paper_aligned_reconstruction": True,
        "seed": args.seed,
        "num_base_trajectories": len(bases),
        "base_trip_ids": [item["trip_id"] for item in bases],
        "num_clusters": len(bases) * 3,
        "total_samples": len(dataframe),
        "base_selection": args.base_selection,
        "base_overlap": summarize_base_overlap(
            bases,
            args.geohash_precision,
        ),
        "base_candidate_filters": {
            "min_length": args.min_base_length,
            "max_length": args.max_base_length,
            "min_distinct_locations": args.min_base_distinct_locations,
        },
        "cluster_size_range": {
            "min": args.min_cluster_size,
            "max": args.max_cluster_size,
        },
        "cluster_sizes": {
            str(label): int(count)
            for label, count in dataframe["label"].value_counts().sort_index().items()
        },
        "temporal_modes": ["original", "time_shift", "speed_scale"],
        "dropout_rate": args.dropout_rate,
        "paper_spatial_noise_meters": 50.0,
        "effective_spatial_noise_meters": args.spatial_noise_meters,
        "time_noise_minutes": args.time_noise_minutes,
        "time_shift_minutes": args.time_shift_minutes,
        "speed_scale": args.speed_scale,
        "geohash_precision": args.geohash_precision,
        "max_sequence_length": args.max_sequence_length,
        "location_vocab_size": len(location_vocab),
        "base_raw_location_count": len(base_geohashes),
        "raw_location_count": raw_location_count,
        "token_vocab_growth_ratio": (
            raw_location_count / len(base_geohashes)
            if base_geohashes else float("nan")
        ),
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
                "effective spatial noise inferred from public QD token behavior",
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
    seed_sequence = np.random.SeedSequence(args.seed)
    candidate_seed, selection_seed, size_seed, sample_seed = seed_sequence.spawn(4)
    candidate_rng = np.random.default_rng(candidate_seed)
    selection_rng = np.random.default_rng(selection_seed)
    size_rng = np.random.default_rng(size_seed)
    sample_rng = np.random.default_rng(sample_seed)
    candidates = collect_candidate_bases(
        input_path=args.input,
        pool_size=args.candidate_pool_size,
        min_length=args.min_base_length,
        max_length=args.max_base_length,
        min_distinct_locations=args.min_base_distinct_locations,
        geohash_precision=args.geohash_precision,
        rng=candidate_rng,
    )
    if args.base_selection == "random":
        bases = select_random_bases(
            candidates,
            args.num_base_trajectories,
            selection_rng,
        )
    else:
        bases = select_shared_subpath_bases(
            candidates=candidates,
            num_bases=args.num_base_trajectories,
            precision=args.geohash_precision,
            min_ratio=args.min_shared_subpath_ratio,
            max_ratio=args.max_shared_subpath_ratio,
            rng=selection_rng,
        )
    print("selected base trip IDs:")
    for index, base in enumerate(bases):
        print(f"  path {index}: {base['trip_id']} (length={len(base['polyline'])})")

    cluster_sizes = sample_cluster_sizes(
        num_clusters=args.num_base_trajectories * 3,
        min_size=args.min_cluster_size,
        max_size=args.max_cluster_size,
        rng=size_rng,
    )
    rows = build_rows(bases, cluster_sizes, args, sample_rng)
    location_vocab = encode_and_pad_rows(rows, args.max_sequence_length)
    dataframe, metadata = save_dataset(rows, bases, location_vocab, args)
    print("saved dataset:", args.output_dir.resolve())
    print("shape:", dataframe.shape)
    print("clusters:", metadata["cluster_sizes"])
    print("location vocab size:", metadata["location_vocab_size"])
    print("base raw locations:", metadata["base_raw_location_count"])
    print("token vocab growth ratio:", metadata["token_vocab_growth_ratio"])
    print("base overlap:", metadata["base_overlap"])
    print("trajectory length:", metadata["trajectory_length"])


if __name__ == "__main__":
    main()
