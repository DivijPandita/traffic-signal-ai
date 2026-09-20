"""
Loads reward_config.yaml and constructs a ready-to-use
StaticMultiObjectiveReward instance. Centralizing this here means
training scripts don't need to know the YAML file's internal
structure -- they just call build_static_reward().
"""

import yaml

from rewards.static_multiobjective import StaticMultiObjectiveReward
from rewards.context_aware_reward import ContextAwareReward


def build_context_aware_reward(config_path: str = "configs/reward_config.yaml") -> ContextAwareReward:
    cfg = load_reward_config(config_path)
    return ContextAwareReward(
        normalization=cfg["normalization"],
        context_weights=cfg["context_weights"],
        context_thresholds=cfg["context_thresholds"],
        emergency_bonus=cfg["emergency_bonus"],
    )


def load_reward_config(path: str = "configs/reward_config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_static_reward(config_path: str = "configs/reward_config.yaml") -> StaticMultiObjectiveReward:
    cfg = load_reward_config(config_path)
    return StaticMultiObjectiveReward(
        normalization=cfg["normalization"],
        weights=cfg["static_weights"],
        emergency_bonus=cfg["emergency_bonus"],
    )