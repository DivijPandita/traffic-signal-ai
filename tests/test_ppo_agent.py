"""
Phase 11 sanity check: verifies PPO's core mechanics (action
sampling, GAE computation, the clipped update step, save/load) work
correctly in isolation, plus a short integration run against the
real environment.
"""

import sys
import os
sys.path.insert(0, "src")

import numpy as np
from agents.ppo_agent import PPOAgent, RolloutBuffer


def test_action_sampling_shapes():
    agent = PPOAgent(state_dim=19, action_dim=2)
    state = np.random.rand(19).astype(np.float32)

    action, log_prob, value = agent.select_action(state)
    assert 0 <= action < 2
    assert isinstance(log_prob, float)
    assert isinstance(value, float)

    greedy_action = agent.select_action_greedy(state)
    assert 0 <= greedy_action < 2
    print(f"✅ Action sampling/greedy both return valid actions (sampled={action}, greedy={greedy_action})")

def test_gae_computation_basic_shape():
    agent = PPOAgent(state_dim=4, action_dim=2)
    rewards = [1.0, 1.0, -1.0, 1.0, 1.0]
    values = [0.5, 0.5, 0.5, 0.5, 0.5]
    dones = [0.0, 0.0, 0.0, 0.0, 1.0]

    advantages, returns = agent._compute_gae(rewards, values, dones, last_value=0.0)
    assert len(advantages) == 5
    assert len(returns) == 5
    # For a terminal step (done=1), there is no bootstrapped future value,
    # so the return at that step should simply equal its own reward.
    assert abs(returns[-1] - rewards[-1]) < 1e-3, (
        f"Expected terminal return ({returns[-1]}) to equal terminal reward ({rewards[-1]})"
    )
    print(f"✅ GAE computes advantages/returns of correct shape: advantages={advantages}")


def test_update_runs_without_error_and_clears_buffer():
    agent = PPOAgent(state_dim=4, action_dim=2, minibatch_size=8)

    for _ in range(32):
        state = np.random.rand(4).astype(np.float32)
        action, log_prob, value = agent.select_action(state)
        reward = np.random.randn()
        agent.store_transition(state, action, reward, log_prob, value, 0.0)

    assert len(agent.buffer) == 32
    loss = agent.update(last_state=np.random.rand(4).astype(np.float32), last_done=False)
    assert isinstance(loss, float)
    assert len(agent.buffer) == 0, "Buffer should be cleared after update()."
    print(f"✅ PPO update runs correctly, loss={loss:.4f}, buffer cleared after update")


def test_save_and_load_preserves_behavior():
    agent = PPOAgent(state_dim=19, action_dim=2)
    state = np.random.rand(19).astype(np.float32)
    action_before = agent.select_action_greedy(state)

    os.makedirs("models/ppo", exist_ok=True)
    save_path = "models/ppo/test_checkpoint.pt"
    agent.save(save_path)

    new_agent = PPOAgent(state_dim=19, action_dim=2)
    new_agent.load(save_path)
    action_after = new_agent.select_action_greedy(state)

    assert action_before == action_after
    print("✅ Save/load preserves agent behavior")
    os.remove(save_path)


def test_short_training_run_against_real_env():
    from training.train_ppo import train

    agent = train(
        "sumo/configs/scenario_moderate.sumocfg",
        "test_run",
        total_updates=2,
        rollout_length=32,
        max_episode_steps=40,
        model_out_dir="models/ppo",
        log_out_dir="results/logs",
    )
    assert os.path.isfile("models/ppo/ppo_test_run.pt")
    assert os.path.isfile("results/logs/ppo_train_test_run.csv")
    print("✅ Short end-to-end PPO training run against real SUMO env completed without errors")

    os.remove("models/ppo/ppo_test_run.pt")
    os.remove("results/logs/ppo_train_test_run.csv")


if __name__ == "__main__":
    test_action_sampling_shapes()
    test_gae_computation_basic_shape()
    test_update_runs_without_error_and_clears_buffer()
    test_save_and_load_preserves_behavior()
    test_short_training_run_against_real_env()
    print("\n🎉 Phase 11 verification passed. PPO agent is working.")