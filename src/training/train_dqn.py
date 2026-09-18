"""
Training loop for the DQN agent against TrafficSignalEnv, using the
simple placeholder waiting-time reward from Phase 8.

This script trains on ONE scenario at a time. Later evaluation
scripts (Phase 10) will load the saved model and test it across
multiple scenarios.
"""

import os
import sys
import csv
import argparse

sys.path.insert(0, "src")

from environment.sumo_env import TrafficSignalEnv
from agents.dqn_agent import DQNAgent
from environment.state import get_state_dim


def train(
    sumocfg_path: str,
    scenario_name: str,
    num_episodes: int = 50,
    max_episode_steps: int = 500,
    seed: int = 42,
    model_out_dir: str = "models/dqn",
    log_out_dir: str = "results/logs",
):
    os.makedirs(model_out_dir, exist_ok=True)
    os.makedirs(log_out_dir, exist_ok=True)

    env = TrafficSignalEnv(
        sumocfg_path,
        max_episode_steps=max_episode_steps,
        seed=seed,
    )

    # One reset() call needed to discover action_dim/state_dim from
    # the live traffic light program before constructing the agent.
    obs, _ = env.reset()
    action_dim = env.action_space.n
    state_dim = obs.shape[0]

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        epsilon_decay_steps=num_episodes * max_episode_steps * 0.6,
    )

    log_path = os.path.join(log_out_dir, f"dqn_train_{scenario_name}.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "total_reward", "avg_waiting_time", "epsilon", "avg_loss"])

    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        episode_waiting_times = []
        losses = []

        done = False
        while not done:
            action = agent.select_action(obs, explore=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.store_experience(obs, action, reward, next_obs, float(done))
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

            episode_reward += reward
            episode_waiting_times.append(info["total_waiting_time"])
            obs = next_obs

        avg_wait = sum(episode_waiting_times) / len(episode_waiting_times)
        avg_loss = sum(losses) / len(losses) if losses else 0.0

        print(
            f"[{scenario_name}] Episode {episode+1}/{num_episodes} | "
            f"reward={episode_reward:.1f} | avg_wait={avg_wait:.2f} | "
            f"epsilon={agent.epsilon:.3f} | avg_loss={avg_loss:.4f}"
        )

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([episode + 1, episode_reward, avg_wait, agent.epsilon, avg_loss])

    model_path = os.path.join(model_out_dir, f"dqn_{scenario_name}.pt")
    agent.save(model_path)
    print(f"\nSaved trained model to {model_path}")
    print(f"Training log saved to {log_path}")

    env.close()
    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="moderate")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=500)
    args = parser.parse_args()

    sumocfg = f"sumo/configs/scenario_{args.scenario}.sumocfg"
    train(sumocfg, args.scenario, num_episodes=args.episodes, max_episode_steps=args.max_steps)