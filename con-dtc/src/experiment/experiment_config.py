from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass(frozen=True)
class ExperimentSettings:
    name: str
    output_root: str


@dataclass(frozen=True)
class DataConfig:
    dataset: str
    batch_size: int
    initialization_batch_size: int


@dataclass(frozen=True)
class ModelConfig:
    num_clusters: int


@dataclass(frozen=True)
class LossConfig:
    time_loss_weight: float
    clustering_loss_weight: float
    instance_temperature: float
    cluster_temperature: float
    instance_loss_weight: float
    cluster_contrastive_loss_weight: float


@dataclass(frozen=True)
class OptimizerConfig:
    representation_learning_rate: float
    clustering_learning_rate: float
    weight_decay: float


@dataclass(frozen=True)
class TrainingConfig:
    num_epochs: int
    seed: int
    log_interval: int
    max_initialization_batches: Optional[int]
    max_train_batches: Optional[int]


@dataclass(frozen=True)
class CheckpointConfig:
    pretrain_path: str
    filename: str


@dataclass(frozen=True)
class ConDTCExperimentConfig:
    experiment: ExperimentSettings
    data: DataConfig
    model: ModelConfig
    loss: LossConfig
    optimizer: OptimizerConfig
    training: TrainingConfig
    checkpoint: CheckpointConfig



def load_experiment_config(config_path):
    config_path = Path(config_path)

    if not config_path.is_file():
        raise FileNotFoundError(
            f"config file does not exist: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file)

    if not isinstance(raw_config, dict):
        raise ValueError("experiment config must be a mapping")

    expected_sections = {
        "experiment",
        "data",
        "model",
        "loss",
        "optimizer",
        "training",
        "checkpoint",
    }

    actual_sections = set(raw_config)
    missing_sections = expected_sections - actual_sections
    unknown_sections = actual_sections - expected_sections

    if missing_sections:
        raise ValueError(
            f"missing config sections: {sorted(missing_sections)}"
        )

    if unknown_sections:
        raise ValueError(
            f"unknown config sections: {sorted(unknown_sections)}"
        )

    config = ConDTCExperimentConfig(
        experiment=ExperimentSettings(
            **raw_config["experiment"]
        ),
        data=DataConfig(**raw_config["data"]),
        model=ModelConfig(**raw_config["model"]),
        loss=LossConfig(**raw_config["loss"]),
        optimizer=OptimizerConfig(
            **raw_config["optimizer"]
        ),
        training=TrainingConfig(
            **raw_config["training"]
        ),
        checkpoint=CheckpointConfig(
            **raw_config["checkpoint"]
        ),
    )
    validate_experiment_config(config)
    return config

def validate_experiment_config(config):
    if not config.experiment.name.strip():
        raise ValueError("experiment name cannot be empty")

    if config.data.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if config.data.initialization_batch_size <= 0:
        raise ValueError(
            "initialization_batch_size must be positive"
        )

    if config.model.num_clusters < 2:
        raise ValueError("num_clusters must be at least 2")

    if config.training.num_epochs <= 0:
        raise ValueError("num_epochs must be positive")

    if config.training.log_interval <= 0:
        raise ValueError("log_interval must be positive")

    if config.loss.instance_temperature <= 0:
        raise ValueError(
            "instance_temperature must be positive"
        )

    if config.loss.cluster_temperature <= 0:
        raise ValueError(
            "cluster_temperature must be positive"
        )

    if config.optimizer.representation_learning_rate <= 0:
        raise ValueError(
            "representation_learning_rate must be positive"
        )

    if config.optimizer.clustering_learning_rate <= 0:
        raise ValueError(
            "clustering_learning_rate must be positive"
        )

    for name in (
        "max_initialization_batches",
        "max_train_batches",
    ):
        value = getattr(config.training, name)
        if value is not None and value <= 0:
            raise ValueError(
                f"{name} must be positive or null"
            )