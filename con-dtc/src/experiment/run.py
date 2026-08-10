import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import yaml
from src.experiment.experiment_config import (
    load_experiment_config,
)
from src.training.train_condtc import train_condtc


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

    print("experiment initialized")
    print("run directory:", run_dir)
    print("config snapshot:", config_snapshot)
    train_condtc()


if __name__ == "__main__":
    main()