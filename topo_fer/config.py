"""Configuration dataclasses for TOPO-FER experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModelConfig:
    """Configuration for the Geometric Scaffold Network."""

    backbone: str = "resnet18"
    pretrained_backbone: bool = True
    feature_dim: int = 512
    latent_dim: int = 128
    ode_hidden_dim: int = 128
    ode_num_layers: int = 2
    ode_time_span: List[float] = field(default_factory=lambda: [0.0, 1.0])
    manifold_projection_dim: int = 64
    classifier_hidden_dim: int = 128
    dropout: float = 0.2


@dataclass
class TopologyConfig:
    """Configuration for the Topological Discovery Module."""

    max_dimension: int = 2
    homology_coefficient: int = 2
    persistence_threshold: float = 0.1
    rips_epsilon: Optional[float] = None
    min_cluster_size: int = 10
    clustering_algorithm: str = "hdbscan"
    hdbscan_min_samples: int = 5
    hdbscan_min_cluster_size: int = 15
    topological_regularizer_weight: float = 1.0
    reconstruction_weight: float = 0.1


@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters."""

    batch_size: int = 64
    num_workers: int = 8
    max_epochs: int = 100
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    known_classification_weight: float = 1.0
    manifold_smoothness_weight: float = 0.5
    contrastive_weight: float = 0.2
    discovery_start_epoch: int = 10
    checkpoint_interval: int = 5
    gradient_clip_norm: float = 5.0
    mixed_precision: bool = True


@dataclass
class DatasetConfig:
    """Configuration for dataset paths and options."""

    name: str = "raf-db"
    root: str = "./data/raf-db"
    split_ratio: float = 0.8
    image_size: int = 224
    augmentations: Optional[List[str]] = field(
        default_factory=lambda: ["random_horizontal_flip", "color_jitter"]
    )
    known_class_ids: Optional[List[int]] = None
    unlabeled_ratio: float = 0.3


@dataclass
class ExperimentConfig:
    """Aggregated configuration for an experiment."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    output_dir: str = "./outputs"
    seed: int = 42


def experiment_config_from_dict(config: Dict[str, Any]) -> ExperimentConfig:
    """Create an ExperimentConfig from a nested dictionary."""
    dataset_cfg = DatasetConfig(**config.get("dataset", {}))
    model_cfg = ModelConfig(**config.get("model", {}))
    training_cfg = TrainingConfig(**config.get("training", {}))
    topology_cfg = TopologyConfig(**config.get("topology", {}))
    return ExperimentConfig(
        dataset=dataset_cfg,
        model=model_cfg,
        training=training_cfg,
        topology=topology_cfg,
        output_dir=config.get("output_dir", "./outputs"),
        seed=config.get("seed", 42),
    )


def load_experiment_config(path: str) -> ExperimentConfig:
    """Load configuration from a YAML file."""
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return experiment_config_from_dict(data or {})
