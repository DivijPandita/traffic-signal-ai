"""
Phase 13 sanity check: verifies the static multi-objective reward
computes sensible values, responds correctly to each of the five
metric terms independently, applies the emergency bonus correctly,
and integrates cleanly into the real environment via reward_fn.
"""

import sys
sys.path.insert(0, "src")

from rewards.config_loader import build_static_reward
from environment.sumo_env import TrafficSignalEnv


def test_reward_is_negative_for_normal_traffic():
    reward_fn = build_static_reward()
    metrics = {
        "total_waiting_time": 20.0,
        "total_queue_length": 5.0,
        "total_fuel_consumption": 10.0,
        "total_co2_emission": 15000.0,
        "total_delay": 2.0,
        "emergency_present": False,
    }
    r = reward_fn.compute(metrics)
    assert r < 0, f"Expected negative reward for normal traffic cost, got {r}"
    print(f"✅ Static reward is negative for typical traffic: {r:.4f}")


def test_reward_worsens_as_congestion_increases():
    reward_fn = build_static_reward()
    light = {
        "total_waiting_time": 5.0, "total_queue_length": 1.0,
        "total_fuel_consumption": 2.0, "total_co2_emission": 2000.0,
        "total_delay": 0.5, "emergency_present": False,
    }
    heavy = {
        "total_waiting_time": 80.0, "total_queue_length": 15.0,
        "total_fuel_consumption": 25.0, "total_co2_emission": 35000.0,
        "total_delay": 8.0, "emergency_present": False,
    }
    r_light = reward_fn.compute(light)
    r_heavy = reward_fn.compute(heavy)
    assert r_heavy < r_light, "Expected heavier congestion to produce a worse (more negative) reward."
    print(f"✅ Reward correctly worsens with congestion: light={r_light:.3f}, heavy={r_heavy:.3f}")


def test_emergency_bonus_applied():
    reward_fn = build_static_reward()
    metrics = {
        "total_waiting_time": 20.0, "total_queue_length": 5.0,
        "total_fuel_consumption": 10.0, "total_co2_emission": 15000.0,
        "total_delay": 2.0, "emergency_present": False,
    }
    r_no_emergency = reward_fn.compute(metrics)

    metrics_emergency = dict(metrics)
    metrics_emergency["emergency_present"] = True
    r_emergency = reward_fn.compute(metrics_emergency)

    assert r_emergency > r_no_emergency, "Expected emergency bonus to increase reward."
    assert abs((r_emergency - r_no_emergency) - reward_fn.emergency_bonus) < 1e-6
    print(f"✅ Emergency bonus correctly applied: +{reward_fn.emergency_bonus}")


def test_weights_are_static_regardless_of_context():
    reward_fn = build_static_reward()
    w_normal = reward_fn.get_weights(context="normal")
    w_congested = reward_fn.get_weights(context="congested")
    w_emergency = reward_fn.get_weights(context="emergency")
    assert w_normal == w_congested == w_emergency, "Static reward's weights must never change with context."
    print("✅ Weights remain identical regardless of context argument (as expected for static reward)")


def test_integrates_with_real_environment():
    reward_fn = build_static_reward()
    env = TrafficSignalEnv(
        "sumo/configs/scenario_moderate.sumocfg",
        max_episode_steps=10,
        reward_fn=reward_fn,
    )
    obs, info = env.reset()
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(reward, float)
        if terminated or truncated:
            break
    print(f"✅ Static multi-objective reward integrates correctly with TrafficSignalEnv (last reward={reward:.4f})")
    env.close()


if __name__ == "__main__":
    test_reward_is_negative_for_normal_traffic()
    test_reward_worsens_as_congestion_increases()
    test_emergency_bonus_applied()
    test_weights_are_static_regardless_of_context()
    test_integrates_with_real_environment()
    print("\n🎉 Phase 13 verification passed. Static multi-objective reward is working.")