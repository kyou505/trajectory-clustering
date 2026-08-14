import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import yaml
from src.experiment.experiment_config import (
    load_experiment_config,
)
from src.evaluate_condtc import evaluate_checkpoint
from src.training.train_condtc import train_condtc


class TeeStream:
    """把写入同时转发到多个流（终端 + 日志文件）。"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def parse_cli_args():
    parser = argparse.ArgumentParser(
        description="Run a Con-DTC experiment"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML file",
    )
    return parser.parse_args()



def create_run_directory(config, project_dir):
    project_dir = Path(project_dir)

    output_root = Path(config.experiment.output_root)
    if not output_root.is_absolute():
        output_root = project_dir / output_root

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"{timestamp}_seed{config.training.seed}"
    )

    run_dir = (
        output_root
        / config.experiment.name
        / run_name
    )

    # 防止意外覆盖已有实验
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoints").mkdir()

    return run_dir


def save_config_snapshot(config, run_dir):
    run_dir = Path(run_dir)
    config_path = run_dir / "config.yaml"

    with config_path.open(
        "x",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            asdict(config),
            file,
            sort_keys=False,
            allow_unicode=True,
        )

    return config_path

def main():
    args = parse_cli_args()
    config = load_experiment_config(args.config)

    project_dir = Path(__file__).resolve().parents[2]

    run_dir = create_run_directory(
        config=config,
        project_dir=project_dir,
    )
    config_snapshot = save_config_snapshot(
        config=config,
        run_dir=run_dir,
    )

    log_path = run_dir / "train.log"
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with log_path.open("w", encoding="utf-8") as log_file:
        sys.stdout = TeeStream(original_stdout, log_file)
        sys.stderr = TeeStream(original_stderr, log_file)
        try:
            print("experiment initialized")
            print("run directory:", run_dir)
            print("config snapshot:", config_snapshot)
            model, history, checkpoint_path = train_condtc(
                num_epochs=config.training.num_epochs,
                batch_size=config.data.batch_size,
                initialization_batch_size=config.data.initialization_batch_size,
                num_clusters=config.model.num_clusters,
                time_loss_weight=config.loss.time_loss_weight,
                clustering_loss_weight=config.loss.clustering_loss_weight,
                instance_temperature=config.loss.instance_temperature,
                cluster_temperature=config.loss.cluster_temperature,
                instance_loss_weight=config.loss.instance_loss_weight,
                cluster_contrastive_loss_weight=config.loss.cluster_contrastive_loss_weight,
                representation_learning_rate=config.optimizer.representation_learning_rate,
                clustering_learning_rate=config.optimizer.clustering_learning_rate,
                weight_decay=config.optimizer.weight_decay,
                seed=config.training.seed,
                max_initialization_batches=config.training.max_initialization_batches,
                max_train_batches=config.training.max_train_batches,
                log_interval=config.training.log_interval,
                output_dir=run_dir,
                pretrain_checkpoint_path=config.checkpoint.pretrain_path,
                target_ema_weight_warmup=config.target_history.weight_warmup,
                target_ema_momentum=config.target_history.momentum,
                target_ema_start_epoch=config.target_history.start_epoch,
                target_ema_minimum_weight=config.target_history.minimum_weight,
                target_ema_weighting_strategy=config.target_history.weighting_strategy,
                target_ema_margin_quantile=config.target_history.margin_quantile,
                target_entropy_mix_enabled=config.target_history.entropy_target_mix_enabled
            )

            print("evaluating:", checkpoint_path)
            metrics = evaluate_checkpoint(
                checkpoint_path=checkpoint_path,
                batch_size=256,
            )
            metrics_path = run_dir / "metrics.json"
            with metrics_path.open("w", encoding="utf-8") as file:
                json.dump(metrics, file, indent=2, ensure_ascii=False)
            print("metrics saved:", metrics_path)
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    main()