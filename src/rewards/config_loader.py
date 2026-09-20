"""
Loads reward_config.yaml and constructs a ready-to-use
StaticMultiObjectiveReward instance. Centralizing this here means
training scripts don't need to know the YAML file's internal
structure -- they just call build_static_reward().
"""

import yaml

from rewards.static_multiobjective import StaticMultiObjectiveReward


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