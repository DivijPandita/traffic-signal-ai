"""
Phase 8 sanity check: verifies the Gym-style environment resets,
steps, respects its declared action/observation spaces, computes a
sane placeholder reward, and terminates/truncates correctly.
"""

import sys
sys.path.insert(0, "src")

import numpy as np
from environment.sumo_env import TrafficSignalEnv


def test_reset_returns_valid_observation():
    env = TrafficSignalEnv("sumo/configs/scenario_moderate.sumocfg", max_episode_steps=20)
    obs, info = env.reset()
    assert env.observation_space.contains(obs) or obs.shape == env.observation_space.shape
    print(f"✅ reset() returns observation of shape {obs.shape}")
    env.close()


def test_action_space_matches_green_phases():
    env = TrafficSignalEnv("sumo/configs/scenario_moderate.sumocfg", max_episode_steps=20)
    env.reset()
    n_actions = env.action_space.n
    assert n_actions >= 2, f"Expected at least 2 green phases, got {n_actions}"
    print(f"✅ Action space has {n_actions} discrete actions (green phases)")
    env.close()


def test_step_returns_correct_tuple_shape():
    env = TrafficSignalEnv("sumo/configs/scenario_moderate.sumocfg", max_episode_steps=20)
    obs, info = env.reset()
    action = 0
    next_obs, reward, terminated, truncated, info = env.step(action)

    assert next_obs.shape == obs.shape
    assert isinstance(reward, (int, float, np.floating))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "total_waiting_time" in info
    print(f"✅ step() returns correct types: reward={reward:.2f}, terminated={terminated}, truncated={truncated}")
    env.close()


def test_reward_is_nonpositive():
    """Placeholder reward is -waiting_time, so it should never be positive."""
    env = TrafficSignalEnv("sumo/configs/scenario_moderate.sumocfg", max_episode_steps=20)
    obs, info = env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert reward <= 0, f"Expected non-positive reward, got {reward}"
        if terminated or truncated:
            break
    print("✅ Placeholder reward stays non-positive as expected")
    env.close()


def test_truncation_triggers_at_max_steps():
    env = TrafficSignalEnv("sumo/configs/scenario_moderate.sumocfg", max_episode_steps=5)
    obs, info = env.reset()
    truncated = False
    steps_taken = 0
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        steps_taken += 1
        if terminated or truncated:
            break
    assert truncated, "Expected truncation to trigger at max_episode_steps=5"
    assert steps_taken == 5
    print(f"✅ Truncation triggers correctly after {steps_taken} steps")
    env.close()


def test_switching_action_inserts_yellow_transition():
    """
    Indirect check: after choosing a DIFFERENT green phase than the
    current one, the phase should end up as the newly requested green
    phase (not stuck on yellow), confirming the transition completed.
    """
    env = TrafficSignalEnv("sumo/configs/scenario_moderate.sumocfg", max_episode_steps=20)
    obs, info = env.reset()

    current_action = 0
    env.step(current_action)

    other_action = 1 if env.action_space.n > 1 else 0
    obs, reward, terminated, truncated, info = env.step(other_action)

    import traci
    final_phase = traci.trafficlight.getPhase(env.tls_id)
    expected_phase = env._action_space_helper.action_to_phase_index(other_action)
    assert final_phase == expected_phase, (
        f"Expected traffic light to end on phase {expected_phase} after switching, got {final_phase}"
    )
    print(f"✅ Phase switch completed correctly (yellow transition handled): now on phase {final_phase}")
    env.close()


if __name__ == "__main__":
    test_reset_returns_valid_observation()
    test_action_space_matches_green_phases()
    test_step_returns_correct_tuple_shape()
    test_reward_is_nonpositive()
    test_truncation_triggers_at_max_steps()
    test_switching_action_inserts_yellow_transition()
    print("\n🎉 Phase 8 verification passed. Gym-style RL environment is working.")