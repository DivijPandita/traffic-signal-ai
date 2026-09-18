"""
Diagnostic: checks whether the trained DQN agent is causing vehicle
insertion starvation (vehicles waiting to enter the network, not yet
spawned) compared to the fixed-time controller, and separately
measures the true cumulative arrived-vehicle count using per-substep
sampling (fixing the suspected undercount bug) for a fair comparison.
"""

import sys
sys.path.insert(0, "src")

import traci
from environment.sumo_env import TrafficSignalEnv
from environment.sumo_connection import SumoConnection
from agents.dqn_agent import DQNAgent


def run_dqn_with_full_diagnostics(sumocfg_path, model_path, seed, max_rl_steps=200):
    env = TrafficSignalEnv(sumocfg_path, max_episode_steps=max_rl_steps)
    obs, _ = env.reset(seed=seed)
    agent = DQNAgent(state_dim=obs.shape[0], action_dim=env.action_space.n)
    agent.load(model_path)

    true_arrived_total = 0
    max_vehicles_in_net = 0

    done = False
    step_count = 0
    while not done and step_count < max_rl_steps:
        action = agent.select_action(obs, explore=False)

        # Manually replicate what step() does internally, but sample
        # arrived count on EVERY substep to get the true cumulative total.
        green_phases = env._action_space_helper.green_phases
        desired_phase = env._action_space_helper.action_to_phase_index(action)
        current_phase = traci.trafficlight.getPhase(env.tls_id)

        if desired_phase != current_phase:
            from environment.actions import get_yellow_phase_for
            yellow_phase = get_yellow_phase_for(env.tls_id, current_phase)
            if yellow_phase is not None:
                traci.trafficlight.setPhase(env.tls_id, yellow_phase)
                for _ in range(env.yellow_duration):
                    env._conn.step()
                    true_arrived_total += traci.simulation.getArrivedNumber()
            traci.trafficlight.setPhase(env.tls_id, desired_phase)

        for _ in range(env.decision_interval):
            env._conn.step()
            true_arrived_total += traci.simulation.getArrivedNumber()

        n_in_net = len(traci.vehicle.getIDList())
        max_vehicles_in_net = max(max_vehicles_in_net, n_in_net)

        from environment.state import build_state_vector
        obs = build_state_vector(env.tls_id, green_phases)
        step_count += 1
        done = not env._conn.is_simulation_running()

    print(f"DQN  | true cumulative arrived: {true_arrived_total} | "
          f"max concurrent vehicles in net: {max_vehicles_in_net}")
    env.close()
    return true_arrived_total


def run_fixed_time_with_diagnostics(sumocfg_path, seed, max_sim_steps=1000):
    conn = SumoConnection(sumocfg_path, seed=seed)
    conn.start()

    true_arrived_total = 0
    max_vehicles_in_net = 0
    steps = 0
    while conn.is_simulation_running() and steps < max_sim_steps:
        conn.step()
        true_arrived_total += traci.simulation.getArrivedNumber()
        n_in_net = len(traci.vehicle.getIDList())
        max_vehicles_in_net = max(max_vehicles_in_net, n_in_net)
        steps += 1

    print(f"Fixed| true cumulative arrived: {true_arrived_total} | "
          f"max concurrent vehicles in net: {max_vehicles_in_net}")
    conn.close()
    return true_arrived_total


if __name__ == "__main__":
    SUMOCFG = "sumo/configs/scenario_moderate.sumocfg"
    MODEL = "models/dqn/dqn_moderate.pt"
    SEED = 100

    print("=== Fixed-Time (per-substep sampling, ~1000s) ===")
    run_fixed_time_with_diagnostics(SUMOCFG, SEED, max_sim_steps=1000)

    print("\n=== DQN (per-substep sampling, ~200 RL steps ≈ up to 1600s) ===")
    run_dqn_with_full_diagnostics(SUMOCFG, MODEL, SEED, max_rl_steps=200)