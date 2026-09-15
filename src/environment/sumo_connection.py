"""
Low-level wrapper around a TraCI connection to SUMO.

This module owns the *lifecycle* of a SUMO process: starting it, stepping
it forward, reading raw traffic-light/vehicle state, and shutting it down
cleanly. It intentionally does NOT know anything about RL concepts
(state vectors, rewards, actions) -- that logic belongs in sumo_env.py
(Phase 8). This separation lets us test "can we talk to SUMO at all"
independently of "does the RL environment work."
"""

import os
import sys
import traci
import sumolib


class SumoConnection:
    """
    Manages a single SUMO simulation process via TraCI.

    Typical usage:
        conn = SumoConnection(sumocfg_path="sumo/configs/scenario_low.sumocfg")
        conn.start()
        for _ in range(100):
            conn.step()
            print(conn.get_current_time())
        conn.close()
    """

    def __init__(self, sumocfg_path: str, use_gui: bool = False, seed: int = 42):
        """
        Args:
            sumocfg_path: path to a .sumocfg file (relative paths inside
                the file, like net-file/route-files, are resolved relative
                to the .sumocfg's own directory by SUMO itself).
            use_gui: if True, launches sumo-gui instead of headless sumo.
                Useful for visual debugging; keep False for training speed.
            seed: random seed passed to SUMO for reproducibility.
        """
        self._check_sumo_home()

        self.sumocfg_path = sumocfg_path
        self.use_gui = use_gui
        self.seed = seed
        self._connected = False

    def _check_sumo_home(self):
        """
        TraCI needs SUMO_HOME to be set correctly (Phase 2). We check this
        explicitly here, rather than letting a cryptic traci error surface
        later, so failures are easy to diagnose.
        """
        if "SUMO_HOME" not in os.environ:
            raise EnvironmentError(
                "SUMO_HOME is not set. Revisit Phase 2 setup."
            )
        tools_path = os.path.join(os.environ["SUMO_HOME"], "tools")
        if tools_path not in sys.path:
            sys.path.append(tools_path)

    def start(self):
        """
        Launches SUMO as a subprocess and establishes the TraCI connection.
        Must be called before step()/close().
        """
        if self._connected:
            raise RuntimeError("SumoConnection.start() called while already connected.")

        binary = "sumo-gui" if self.use_gui else "sumo"
        sumo_cmd = [
            binary,
            "-c", self.sumocfg_path,
            "--seed", str(self.seed),
            "--no-step-log", "true",
            "--time-to-teleport", "300",
        ]
        traci.start(sumo_cmd)
        self._connected = True

    def step(self):
        """
        Advances the simulation by exactly one simulation step
        (default 1 second of simulated time, as configured in the network).
        This is the fundamental unit of interaction: in the RL loop
        (Phase 8+), one call to step() corresponds to one environment
        step, though we may call it multiple times per RL action to
        simulate "holding" a signal phase for several seconds.
        """
        if not self._connected:
            raise RuntimeError("Cannot step(): connection not started. Call start() first.")
        traci.simulationStep()

    def get_current_time(self) -> float:
        """Current simulation time in seconds since episode start."""
        return traci.simulation.getTime()

    def get_vehicle_ids(self) -> list:
        """IDs of all vehicles currently in the simulation."""
        return list(traci.vehicle.getIDList())

    def get_traffic_light_ids(self) -> list:
        """IDs of all traffic lights in the network. Should be ['C'] for now."""
        return list(traci.trafficlight.getIDList())

    def get_traffic_light_phase(self, tls_id: str) -> int:
        """Current phase index of the given traffic light."""
        return traci.trafficlight.getPhase(tls_id)

    def set_traffic_light_phase(self, tls_id: str, phase_index: int):
        """
        Forces the traffic light to a specific phase index immediately.
        This is the core "action" primitive the RL agent will eventually
        call (Phase 8+), though action design there will be more nuanced
        (e.g., minimum green time enforcement).
        """
        traci.trafficlight.setPhase(tls_id, phase_index)

    def get_lane_ids_for_edge(self, edge_id: str) -> list:
        """
        Lane IDs belonging to a given edge, e.g. edge 'N2C' -> ['N2C_0', 'N2C_1'].
        Needed because TraCI's per-lane queries (queue length, waiting time)
        operate on lane IDs, not edge IDs.
        """
        n_lanes = traci.edge.getLaneNumber(edge_id)
        return [f"{edge_id}_{i}" for i in range(n_lanes)]

    def get_edge_vehicle_count(self, edge_id: str) -> int:
        """Number of vehicles currently on a given edge."""
        return traci.edge.getLastStepVehicleNumber(edge_id)

    def get_edge_waiting_time(self, edge_id: str) -> float:
        """
        Sum of waiting time (seconds stopped, speed < 0.1 m/s) of all
        vehicles currently on the edge. This is a core Phase 8 state
        variable and Phase 13+ reward ingredient.
        """
        return traci.edge.getWaitingTime(edge_id)

    def is_simulation_running(self) -> bool:
        """
        True if there are still vehicles expected (either currently present
        or scheduled to depart later in the route file).
        """
        return traci.simulation.getMinExpectedNumber() > 0

    def close(self):
        """Cleanly shuts down the TraCI connection and the SUMO process."""
        if self._connected:
            traci.close()
            self._connected = False