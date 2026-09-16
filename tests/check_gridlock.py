"""
Pre-Phase-9 diagnostic: runs heavy/peak scenarios through the Gym
environment with RANDOM actions (worst-case-ish, similar to an
untrained agent's early behavior) and reports whether queues grow
unboundedly or stabilize. This is not a pass/fail test -- it's a
tuning diagnostic to decide if demand_heavy/demand_peak need adjusting
before we start real training in Phase 9.
"""

import sys
sys.path.insert(0, "src")

from environment.sumo_env import TrafficSignalEnv


def check_scenario(sumocfg_path: str, name: str, max_steps: int = 700):
    env = TrafficSignalEnv(sumocfg_path, max_episode_steps=max_steps)
    obs, info = env.reset()

    queue_history = []
    for step in range(max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        queue_history.append(info["total_queue_length"])
        if terminated or truncated:
            break

    first_quarter = queue_history[:len(queue_history)//4] or [0]
    last_quarter = queue_history[-len(queue_history)//4:] or [0]
    avg_first = sum(first_quarter) / len(first_quarter)
    avg_last = sum(last_quarter) / len(last_quarter)
    max_queue = max(queue_history)

    print(f"\n=== {name} ===")
    print(f"  steps completed: {len(queue_history)}")
    print(f"  avg queue (first quarter): {avg_first:.1f}")
    print(f"  avg queue (last quarter):  {avg_last:.1f}")
    print(f"  max queue observed:        {max_queue}")
    if avg_last > avg_first * 2 and avg_last > 15:
        print(f"  ⚠️  WARNING: queue appears to be growing unboundedly (possible gridlock)")
    else:
        print(f"  ✅ queue appears to stabilize or stay manageable")

    env.close()


if __name__ == "__main__":
    check_scenario("sumo/configs/scenario_moderate.sumocfg", "moderate (reference)")
    check_scenario("sumo/configs/scenario_heavy.sumocfg", "heavy")
    check_scenario("sumo/configs/scenario_peak.sumocfg", "peak")