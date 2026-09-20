"""
Diagnostic: runs each scenario under RANDOM actions (a reasonable
stand-in for "unoptimized" traffic) and reports how often each
context gets triggered, so we can sanity-check (and if needed retune)
the thresholds in reward_config.yaml BEFORE using them for real
training.
"""

import sys
from collections import Counter
sys.path.insert(0, "src")

from environment.sumo_env import TrafficSignalEnv
from rewards.config_loader import build_context_aware_reward


def check_distribution(sumocfg_path: str, name: str, max_steps: int = 500):
    reward_fn = build_context_aware_reward()
    env = TrafficSignalEnv(sumocfg_path, max_episode_steps=max_steps, reward_fn=reward_fn)
    obs, info = env.reset()

    context_counts = Counter()
    for _ in range(max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        context_counts[info["active_context"]] += 1
        if terminated or truncated:
            break

    total = sum(context_counts.values())
    print(f"\n=== {name} (random actions, {total} steps) ===")
    for ctx, count in context_counts.most_common():
        print(f"  {ctx:<15} {count:>5} ({100*count/total:.1f}%)")

    env.close()


if __name__ == "__main__":
    check_distribution("sumo/configs/scenario_moderate.sumocfg", "moderate")
    check_distribution("sumo/configs/scenario_heavy.sumocfg", "heavy")
    check_distribution("sumo/configs/scenario_peak.sumocfg", "peak")
    check_distribution("sumo/configs/scenario_high_emission.sumocfg", "high_emission")
    check_distribution("sumo/configs/scenario_emergency.sumocfg", "emergency")