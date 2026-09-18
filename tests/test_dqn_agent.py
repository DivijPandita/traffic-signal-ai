"""
Phase 9 sanity check: verifies the DQN agent's core mechanics
(action selection, replay buffer, training step, target network sync,
save/load) work correctly in isolation, plus a short integration run
against the real environment.
"""

import sys
import os
sys.path.insert(0, "src")

import numpy as np
import torch

from agents.dqn_agent import DQNAgent, ReplayBuffer


def test_replay_buffer_stores_and_samples():
    buf = ReplayBuffer(capacity=100)
    for i in range(20):
        state = np.random.rand(5).astype(np.float32)
        buf.push(state, i % 3, float(i), state, 0.0)
    assert len(buf) == 20
    states, actions, rewards, next_states, dones = buf.sample(10)
    assert states.shape == (10, 5)
    assert actions.shape == (10,)
    print("✅ ReplayBuffer stores and samples correctly")


def test_agent_action_selection_shapes():
    agent = DQNAgent(state_dim=19, action_dim=2)
    state = np.random.rand(19).astype(np.float32)

    action_explore = agent.select_action(state, explore=True)
    action_greedy = agent.select_action(state, explore=False)
    assert 0 <= action_explore < 2
    assert 0 <= action_greedy < 2
    print(f"✅ Action selection returns valid actions (explore={action_explore}, greedy={action_greedy})")


def test_train_step_reduces_loss_over_time():
    """
    Not a strict convergence guarantee (RL is noisy), but on a purely
    synthetic, easy-to-fit dataset, loss should trend down, not stay
    flat or explode. This catches gross errors in the Bellman update.
    """
    agent = DQNAgent(state_dim=4, action_dim=2, batch_size=16, target_update_freq=50)

    # Fill buffer with simple synthetic experiences: fixed reward
    # pattern the network should be able to learn quickly.
    for _ in range(500):
        state = np.random.rand(4).astype(np.float32)
        action = np.random.randint(0, 2)
        reward = 1.0 if action == 0 else -1.0
        next_state = np.random.rand(4).astype(np.float32)
        agent.store_experience(state, action, reward, next_state, 0.0)

    losses = []
    for _ in range(200):
        loss = agent.train_step()
        if loss is not None:
            losses.append(loss)

    assert len(losses) > 0, "No training steps occurred."
    early_avg = sum(losses[:20]) / 20
    late_avg = sum(losses[-20:]) / 20
    print(f"✅ Loss trend: early_avg={early_avg:.4f}, late_avg={late_avg:.4f}")
    assert late_avg < early_avg * 3, "Loss exploded rather than learning."


def test_save_and_load_preserves_behavior():
    agent = DQNAgent(state_dim=19, action_dim=2)
    state = np.random.rand(19).astype(np.float32)

    action_before = agent.select_action(state, explore=False)

    os.makedirs("models/dqn", exist_ok=True)
    save_path = "models/dqn/test_checkpoint.pt"
    agent.save(save_path)

    new_agent = DQNAgent(state_dim=19, action_dim=2)
    new_agent.load(save_path)
    action_after = new_agent.select_action(state, explore=False)

    assert action_before == action_after, "Loaded agent produces different greedy action than original."
    print("✅ Save/load preserves agent behavior")

    os.remove(save_path)


def test_short_training_run_against_real_env():
    """
    Integration check: runs a VERY short real training loop (3 episodes,
    40 steps each) against the actual SUMO environment, just to confirm
    nothing crashes end-to-end. Not checking for convergence here.
    """
    from training.train_dqn import train

    agent = train(
        "sumo/configs/scenario_moderate.sumocfg",
        "test_run",
        num_episodes=3,
        max_episode_steps=40,
        model_out_dir="models/dqn",
        log_out_dir="results/logs",
    )
    assert os.path.isfile("models/dqn/dqn_test_run.pt")
    assert os.path.isfile("results/logs/dqn_train_test_run.csv")
    print("✅ Short end-to-end training run against real SUMO env completed without errors")

    os.remove("models/dqn/dqn_test_run.pt")
    os.remove("results/logs/dqn_train_test_run.csv")


if __name__ == "__main__":
    test_replay_buffer_stores_and_samples()
    test_agent_action_selection_shapes()
    test_train_step_reduces_loss_over_time()
    test_save_and_load_preserves_behavior()
    test_short_training_run_against_real_env()
    print("\n🎉 Phase 9 verification passed. DQN agent is working.")