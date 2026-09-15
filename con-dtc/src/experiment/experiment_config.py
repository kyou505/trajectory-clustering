from dataclasses import dataclass
import math
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
    # 设为 0 时，同时禁用历史编码器推理和额外的对比损失。
    history_instance_loss_weight: float = 0.0
    history_instance_temperature: float = 0.5
    history_instance_start_epoch: int = 3
    # 设为 None 时，沿用直接保存上一轮在线编码器快照的方式。
    history_encoder_ema_momentum: Optional[float] = None


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
    # 仅用于离线诊断，不把历史表示或真实标签用于优化。
    save_representation_history: bool = False
    # 默认保持旧配置按基础损失选 best；固定训练预算的对照可选 last。
    checkpoint_selection: str = "best"


@dataclass(frozen=True)
class CheckpointConfig:
    pretrain_path: str

@dataclass(frozen=True)
class TargetHistoryConfig:
    momentum: float

    save_interval: int = 1
    save_raw_assignments: bool = True
    start_epoch: int = 1
    # false：EMA 启用后立即使用目标 minimum_weight
    # true：从 1.0 线性下降到目标 minimum_weight
    weight_warmup: bool = False
    # 加权方式：stability / margin / stability_margin
    weighting_signal: str = "stability"
    # 是否对高熵样本使用 EMA/current target 插值
    entropy_target_mix_enabled: bool = False
    # 使用当前 epoch 熵分布的分位点划分高熵样本
    entropy_quantile: float = 0.7
    # 连续插值：在entropy_quantile到该分位点之间连续降低EMA比例。为None时直接使用硬阈值
    entropy_upper_quantile: Optional[float] = None
    # 高熵样本目标中 EMA target 的占比
    high_entropy_ema_alpha: float = 0.5
    # 不设置表示不启用DEC样本加权
    minimum_weight: Optional[float] = None


@dataclass(frozen=True)
class LFSSConfig:
    """仅表示层 LS、LI；与 DART 的软分配 EMA 独立，不接入 LC。"""
    enabled: bool = True
    noise_weight: float = 1.0
    instance_weight: float = 0.1
    target_momentum: float = 0.996
    temperature: float = 0.5
    noise_std: float = 0.001
    projection_dim: int = 256
    hidden_dim: int = 4096


def validate_lfss_config(config, history_instance_loss_weight=0.0):
    if config is None:
        return
    if not isinstance(config.enabled, bool):
        raise ValueError("lfss.enabled 必须是布尔值")
    for name in ("noise_weight", "instance_weight", "noise_std"):
        value = getattr(config, name)
        if isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError(f"lfss.{name} 必须是有限非负数")
    if (isinstance(config.target_momentum, bool)
            or not math.isfinite(config.target_momentum)
            or not 0 <= config.target_momentum < 1):
        raise ValueError("lfss.target_momentum 必须属于 [0, 1)")
    if (isinstance(config.temperature, bool)
            or not math.isfinite(config.temperature) or config.temperature <= 0):
        raise ValueError("lfss.temperature 必须是有限正数")
    for name in ("projection_dim", "hidden_dim"):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"lfss.{name} 必须是正整数")
    if config.enabled:
        if config.noise_weight + config.instance_weight <= 0:
            raise ValueError("LFSS 启用时 LS、LI 至少有一项权重大于 0")
        if history_instance_loss_weight > 0:
            raise ValueError("LFSS 表示约束与旧版 history_instance_loss_weight 不能同时启用")


@dataclass(frozen=True)
class ConDTCExperimentConfig:
    experiment: ExperimentSettings
    data: DataConfig
    model: ModelConfig
    loss: LossConfig
    optimizer: OptimizerConfig
    training: TrainingConfig
    checkpoint: CheckpointConfig
    target_history: Optional[TargetHistoryConfig] = None
    lfss: Optional[LFSSConfig] = None



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

    required_sections = {
        "experiment",
        "data",
        "model",
        "loss",
        "optimizer",
        "training",
        "checkpoint",
    }

    optional_sections = {
        "target_history",
        "lfss",
    }

    actual_sections = set(raw_config)
    missing_sections = required_sections - actual_sections
    unknown_sections = actual_sections - required_sections - optional_sections

    target_history_raw = raw_config.get("target_history")
    target_history = (
        TargetHistoryConfig(**target_history_raw)
        if target_history_raw is not None
        else None
    )

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
        target_history=target_history,
        lfss=LFSSConfig(**raw_config["lfss"]) if raw_config.get("lfss") is not None else None,
    )
    validate_experiment_config(config)
    return config

def validate_experiment_config(config):
    validate_lfss_config(config.lfss, config.loss.history_instance_loss_weight)
    if config.training.checkpoint_selection not in ("best", "last"):
        raise ValueError("checkpoint_selection 必须为 best 或 last")
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

    if not isinstance(config.training.save_representation_history, bool):
        raise ValueError("save_representation_history must be a boolean")

    if config.loss.instance_temperature <= 0:
        raise ValueError(
            "instance_temperature must be positive"
        )

    if config.loss.cluster_temperature <= 0:
        raise ValueError(
            "cluster_temperature must be positive"
        )

    validate_history_instance_config(
        config.loss.history_instance_loss_weight,
        config.loss.history_instance_temperature,
        config.loss.history_instance_start_epoch,
        config.training.num_epochs,
        config.loss.history_encoder_ema_momentum,
    )

    if config.optimizer.representation_learning_rate <= 0:
        raise ValueError(
            "representation_learning_rate must be positive"
        )

    if config.optimizer.clustering_learning_rate <= 0:
        raise ValueError(
            "clustering_learning_rate must be positive"
        )

    target_history = config.target_history
    if target_history is not None:
        if not (1 <= target_history.start_epoch <= config.training.num_epochs):
            raise ValueError("target_history.start_epoch must be between 1 and training.num_epochs")
        if target_history.minimum_weight is not None:
            if target_history.minimum_weight <= 0.0:
                raise ValueError("target_history.minimum_weight must be greater than 0")

        if target_history.weight_warmup and target_history.minimum_weight is None:
            raise ValueError("weight_warmup requires minimum_weight")

        if not 0.0 < target_history.entropy_quantile < 1.0:
            raise ValueError(
                "target_history.entropy_quantile must be in (0, 1)"
            )

        if not 0.0 <= target_history.high_entropy_ema_alpha <= 1.0:
            raise ValueError(
                "target_history.high_entropy_ema_alpha must be in [0, 1]"
            )
        if target_history.weighting_signal not in {
            "stability",
            "margin",
            "stability_margin",
        }:
            raise ValueError("invalid target_history.weighting_signal")

        upper_quantile = target_history.entropy_upper_quantile
        if upper_quantile is not None:
            if not target_history.entropy_quantile < upper_quantile <= 1.0:
                raise ValueError("target_history.entropy_upper_quantile must be greater than entropy_quantile and no greater than 1")

    for name in (
        "max_initialization_batches",
        "max_train_batches",
    ):
        value = getattr(config.training, name)
        if value is not None and value <= 0:
            raise ValueError(
                f"{name} must be positive or null"
            )


def validate_history_instance_config(weight, temperature, start_epoch, num_epochs,
                                     encoder_ema_momentum=None):
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("history_instance_loss_weight must be finite and nonnegative")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("history_instance_temperature must be finite and positive")
    if isinstance(start_epoch, bool) or not isinstance(start_epoch, int) or start_epoch < 1:
        raise ValueError("history_instance_start_epoch must be a positive integer")
    if weight > 0 and start_epoch > num_epochs:
        raise ValueError("history_instance_start_epoch must not exceed num_epochs when enabled")
    if encoder_ema_momentum is not None:
        if (isinstance(encoder_ema_momentum, bool)
                or not math.isfinite(encoder_ema_momentum)
                or not 0 <= encoder_ema_momentum < 1):
            raise ValueError("history_encoder_ema_momentum must be null or finite in [0, 1)")
