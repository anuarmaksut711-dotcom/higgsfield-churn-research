"""Central lightweight configuration for reproducible experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    observation_days: int = 14
    temporal_block_users: int = 15_000
    temporal_folds: int = 3
    random_seed: int = 42
    iterations: int = 650
    learning_rate: float = 0.06
    depth: int = 7
    l2_leaf_reg: float = 5.0
    random_strength: float = 0.5
    early_stopping_rounds: int = 80
    thread_count: int = -1


DEFAULT_CONFIG = PipelineConfig()
