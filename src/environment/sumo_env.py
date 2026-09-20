"""
Gymnasium-style RL environment wrapping SUMO via TraCI.

This ties together:
  - sumo_connection.py (Phase 5): process lifecycle
  - metrics.py (Phase 7): live traffic measurements
  - state.py (this phase): observation construction
  - actions.py (this phase): action space and phase-switching logic

Reward in THIS phase is an intentionally simple placeholder (negative
total waiting time). It will be replaced by the modular reward classes
in Phase 13 (static multi-objective) and Phase 14 (context-aware) --
by design, that swap will only touch the reward computation inside
step(), not this file's overall structure.
"""

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import traci
import yaml
from rewards.base_reward import BaseReward

sys.path.insert(0, os.path.dirname(__file__))

from sumo_connection import SumoConnection
from actions import ActionSpace, get_yellow_phase_for
from state import build_state_vector, get_state_dim
from metrics import get_network_totals


class TrafficSignalEnv(gym.Env):
    """
    One episode = one full run of a given .sumocfg scenario.
    One RL step = the agent picks a green phase, the environment holds
    it (after any mandatory yellow transition) for `decision_interval`
    seconds of simulated time, then returns the resulting state/reward.
    """

    def __init__(
        self,
        sumocfg_path: str,
        tls_id: str = "C",
        decision_interval: int = 5,
        yellow_duration: int = 3,
        max_episode_steps: int = 700,
        seed: int = 42,
        use_gui: bool = False,
        reward_fn: BaseReward = None,
    ):
        super().__init__()
        self.sumocfg_path = sumocfg_path
        self.tls_id = tls_id
        self.decision_interval = decision_interval
        self.yellow_duration = yellow_duration
        self.max_episode_steps = max_episode_steps
        self.seed_value = seed
        self.use_gui = use_gui
        self.reward_fn = reward_fn  # None => fall back to placeholder reward

        self._conn = None
        self._action_space_helper = None
        self._episode_step_count = 0

        self.action_space = None
        self.observation_space = None


    def reset(self, seed=None, options=None):
        if self._conn is not None:
            self._conn.close()

        if seed is not None:
            self.seed_value = seed

        self._conn = SumoConnection(
            self.sumocfg_path, use_gui=self.use_gui, seed=self.seed_value
        )
        self._conn.start()
        self._episode_step_count = 0

        self._action_space_helper = ActionSpace(self.tls_id)
        if self.action_space is None:
            self.action_space = spaces.Discrete(self._action_space_helper.n)
            obs_dim = get_state_dim(self._action_space_helper.n)
            self.observation_space = spaces.Box(
                low=0.0, high=10.0, shape=(obs_dim,), dtype=np.float32
            )

        obs = build_state_vector(self.tls_id, self._action_space_helper.green_phases)
        info = {}
        return obs, info       

    def step(self, action: int):
        green_phases = self._action_space_helper.green_phases
        desired_phase = self._action_space_helper.action_to_phase_index(action)
        current_phase = traci.trafficlight.getPhase(self.tls_id)

        arrived_this_rl_step = 0

        # If switching to a different green phase, insert the mandatory
        # yellow transition first.
        if desired_phase != current_phase:
            yellow_phase = get_yellow_phase_for(self.tls_id, current_phase)
            if yellow_phase is not None:
                traci.trafficlight.setPhase(self.tls_id, yellow_phase)
                for _ in range(self.yellow_duration):
                    self._conn.step()
                    arrived_this_rl_step += traci.simulation.getArrivedNumber()
            traci.trafficlight.setPhase(self.tls_id, desired_phase)

        # Hold the chosen green phase for decision_interval seconds.
        for _ in range(self.decision_interval):
            self._conn.step()
            arrived_this_rl_step += traci.simulation.getArrivedNumber()

        obs = build_state_vector(self.tls_id, green_phases)

        totals = get_network_totals()

        if self.reward_fn is not None:
            # context defaults to "normal" until Phase 14 wires in a
            # real context-detection step here.
            reward = self.reward_fn.compute(totals, context="normal")
        else:
            # Fallback placeholder reward, preserved for backward
            # compatibility with Phase 8-12 code/tests that don't pass
            # a reward_fn explicitly.
            reward = -totals["total_waiting_time"]
        

        self._episode_step_count += 1
        terminated = not self._conn.is_simulation_running()
        truncated = self._episode_step_count >= self.max_episode_steps

        info = {
            "total_queue_length": totals["total_queue_length"],
            "total_waiting_time": totals["total_waiting_time"],
            "total_co2_emission": totals["total_co2_emission"],
            "total_fuel_consumption": totals["total_fuel_consumption"],
            "emergency_present": totals["emergency_present"],
            "arrived_count": arrived_this_rl_step,
        }

        return obs, reward, terminated, truncated, info
    

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None