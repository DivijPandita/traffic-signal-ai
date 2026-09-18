"""
Fair evaluation: compares a trained DQN agent against the fixed-time
baseline, using the SAME live-metrics measurement method for both
(see Phase 10 rationale -- Phase 6's tripinfo-based baseline is not
directly comparable to live-metric-based agent evaluation).

Both controllers are evaluated across the SAME set of held-out seeds
that were never used during DQN training, so the comparison reflects
generalization, not memorization of one traffic realization.
"""

import os
import sys
import numpy as np

sys.path.insert(0, "src")

from environment.sumo_env import TrafficSignalEnv
from environment.sumo_connection import SumoConnection
from environment.metrics import get_network_totals
from agents.dqn_agent import DQNAgent
import traci


def evaluate_dqn_agent(
    sumocfg_path: str,
    model_path: str,
    eval_seeds: list,
    max_episode_steps: int = 700,
) -> dict:
    """
    Runs the trained DQN agent (epsilon=0, pure exploitation) across
    each seed in eval_seeds, collecting live per-step metrics, then
    returns mean/std across seeds.
    """
    env = TrafficSignalEnv(sumocfg_path, max_episode_steps=max_episode_steps)
    obs, _ = env.reset(seed=eval_seeds[0])
    action_dim = env.action_space.n
    state_dim = obs.shape[0]

    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
    agent.load(model_path)

    per_seed_results = []
    for seed in eval_seeds:
        obs, info = env.reset(seed=seed)
        waiting_times, queue_lengths, co2_list, fuel_list = [], [], [], []
        total_arrived = 0

        done = False
        while not done:
            action = agent.select_action(obs, explore=False)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            waiting_times.append(info["total_waiting_time"])
            queue_lengths.append(info["total_queue_length"])
            co2_list.append(info["total_co2_emission"])
            fuel_list.append(info["total_fuel_consumption"])
            total_arrived += info["arrived_count"]

        per_seed_results.append({
            "avg_waiting_time": np.mean(waiting_times),
            "avg_queue_length": np.mean(queue_lengths),
            "avg_co2": np.mean(co2_list),
            "avg_fuel": np.mean(fuel_list),
            "throughput": total_arrived,
        })

    env.close()
    return _aggregate(per_seed_results)


def evaluate_fixed_time_live(
    sumocfg_path: str,
    eval_seeds: list,
    max_episode_steps: int = 700,
    decision_interval: int = 5,
) -> dict:
    """
    Runs the fixed-time (default program) controller across the same
    seeds, using the SAME live-metrics collection as evaluate_dqn_agent
    so the comparison is apples-to-apples. Crucially, we never call
    setPhase() here -- the traffic light follows its own program
    autonomously, exactly like Phase 6, just measured differently.
    """
    per_seed_results = []

    for seed in eval_seeds:
        conn = SumoConnection(sumocfg_path, seed=seed)
        conn.start()

        waiting_times, queue_lengths, co2_list, fuel_list = [], [], [], []
        total_arrived = 0
        steps = 0

        while conn.is_simulation_running() and steps < max_episode_steps * decision_interval:
            conn.step()
            totals = get_network_totals()
            waiting_times.append(totals["total_waiting_time"])
            queue_lengths.append(totals["total_queue_length"])
            co2_list.append(totals["total_co2_emission"])
            fuel_list.append(totals["total_fuel_consumption"])
            total_arrived += traci.simulation.getArrivedNumber()
            steps += 1

        per_seed_results.append({
            "avg_waiting_time": np.mean(waiting_times),
            "avg_queue_length": np.mean(queue_lengths),
            "avg_co2": np.mean(co2_list),
            "avg_fuel": np.mean(fuel_list),
            "throughput": total_arrived,
        })
        conn.close()

    return _aggregate(per_seed_results)


def _aggregate(per_seed_results: list) -> dict:
    """Mean and std across seeds for each metric -- required by the
    project spec's experimental rigor section."""
    keys = per_seed_results[0].keys()
    aggregated = {}
    for k in keys:
        values = [r[k] for r in per_seed_results]
        aggregated[f"{k}_mean"] = float(np.mean(values))
        aggregated[f"{k}_std"] = float(np.std(values))
    return aggregated


def compare(scenario_name: str, sumocfg_path: str, model_path: str, eval_seeds: list):
    print(f"\n{'='*60}")
    print(f"Scenario: {scenario_name} | Eval seeds: {eval_seeds}")
    print(f"{'='*60}")

    fixed = evaluate_fixed_time_live(sumocfg_path, eval_seeds)
    dqn = evaluate_dqn_agent(sumocfg_path, model_path, eval_seeds)

    print(f"\n{'Metric':<20}{'Fixed-Time':<25}{'DQN':<25}")
    for metric in ["avg_waiting_time", "avg_queue_length", "avg_co2", "avg_fuel", "throughput"]:
        f_str = f"{fixed[metric+'_mean']:.2f} ± {fixed[metric+'_std']:.2f}"
        d_str = f"{dqn[metric+'_mean']:.2f} ± {dqn[metric+'_std']:.2f}"
        print(f"{metric:<20}{f_str:<25}{d_str:<25}")

    return {"fixed_time": fixed, "dqn": dqn}


if __name__ == "__main__":
    EVAL_SEEDS = [100, 101, 102, 103, 104]  # held out, never used in training
    compare(
        "moderate",
        "sumo/configs/scenario_moderate.sumocfg",
        "models/dqn/dqn_moderate.pt",
        EVAL_SEEDS,
    )