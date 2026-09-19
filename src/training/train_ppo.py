"""
Training loop for the PPO agent against TrafficSignalEnv -- same
environment, same simple waiting-time reward as DQN (Phase 9/10), so
the two algorithms are directly comparable on identical scenarios.

Structural difference from train_dqn.py: PPO collects a full ROLLOUT
(a fixed number of steps, possibly spanning multiple episodes) before
each policy update, rather than updating after every single step.
"""

import os
import sys
import csv
import argparse

sys.path.insert(0, "src")

from environment.sumo_env import TrafficSignalEnv
from agents.ppo_agent import PPOAgent


def train(
    sumocfg_path: str,
    scenario_name: str,
    total_updates: int = 100,
    rollout_length: int = 256,
    max_episode_steps: int = 700,
    train_seeds: list = None,
    model_out_dir: str = "models/ppo",
    log_out_dir: str = "results/logs",
):
    os.makedirs(model_out_dir, exist_ok=True)
    os.makedirs(log_out_dir, exist_ok=True)

    if train_seeds is None:
        train_seeds = list(range(10))  # same convention as DQN: 0-9, eval uses 100+

    env = TrafficSignalEnv(sumocfg_path, max_episode_steps=max_episode_steps, seed=train_seeds[0])
    obs, _ = env.reset()
    action_dim = env.action_space.n
    state_dim = obs.shape[0]

    agent = PPOAgent(state_dim=state_dim, action_dim=action_dim)

    log_path = os.path.join(log_out_dir, f"ppo_train_{scenario_name}.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["update", "avg_reward_per_step", "avg_waiting_time", "loss"])

    episode_count = 0
    episode_seed = train_seeds[episode_count % len(train_seeds)]
    obs, info = env.reset(seed=episode_seed)
    episode_step = 0

    for update_idx in range(total_updates):
        rollout_rewards = []
        rollout_waiting_times = []

        for _ in range(rollout_length):
            action, log_prob, value = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.store_transition(obs, action, reward, log_prob, value, float(done))
            rollout_rewards.append(reward)
            rollout_waiting_times.append(info["total_waiting_time"])

            obs = next_obs
            episode_step += 1

            if done:
                episode_count += 1
                episode_seed = train_seeds[episode_count % len(train_seeds)]
                obs, info = env.reset(seed=episode_seed)
                episode_step = 0

        loss = agent.update(last_state=obs, last_done=done)

        avg_reward = sum(rollout_rewards) / len(rollout_rewards)
        avg_wait = sum(rollout_waiting_times) / len(rollout_waiting_times)

        print(
            f"[{scenario_name}] Update {update_idx+1}/{total_updates} | "
            f"avg_reward/step={avg_reward:.2f} | avg_wait={avg_wait:.2f} | loss={loss:.4f}"
        )

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([update_idx + 1, avg_reward, avg_wait, loss])

    model_path = os.path.join(model_out_dir, f"ppo_{scenario_name}.pt")
    agent.save(model_path)
    print(f"\nSaved trained model to {model_path}")
    print(f"Training log saved to {log_path}")

    env.close()
    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="moderate")
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--rollout-length", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=700)
    args = parser.parse_args()

    sumocfg = f"sumo/configs/scenario_{args.scenario}.sumocfg"
    train(
        sumocfg, args.scenario,
        total_updates=args.updates,
        rollout_length=args.rollout_length,
        max_episode_steps=args.max_steps,
    )