"""
Baseline controllers for comparison against the RL agents (Phase 9+).

This file currently implements only the Fixed-Time baseline: SUMO's
default signal program, generated automatically by netconvert in
Phase 3, run without any interference. Later baselines (e.g. a simple
threshold-based adaptive controller) will be added to this same file
so all baselines share a consistent evaluation interface.
"""

import os
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, "src")

from environment.sumo_connection import SumoConnection


class FixedTimeController:
    """
    Runs a full episode letting SUMO's default traffic light program
    operate unmodified. We never call set_traffic_light_phase() here --
    that's the whole point: this measures "do nothing extra" performance.
    """

    def __init__(self, sumocfg_path: str, seed: int = 42, use_gui: bool = False):
        self.sumocfg_path = sumocfg_path
        self.seed = seed
        self.use_gui = use_gui

    def run_episode(self, tripinfo_output_path: str, max_steps: int = 4000) -> dict:
        """
        Runs one full episode and writes per-vehicle trip statistics to
        tripinfo_output_path. Returns an aggregated metrics dict.

        Note: tripinfo output requires passing --tripinfo-output directly
        to the sumo command line, so we construct the connection's
        underlying command manually here rather than reusing
        SumoConnection.start() as-is.
        """
        conn = SumoConnection(self.sumocfg_path, use_gui=self.use_gui, seed=self.seed)

        # We need --tripinfo-output on the SUMO command line itself, so we
        # patch traci.start() call by temporarily extending the command.
        # SumoConnection doesn't expose this yet, so we do a minimal,
        # explicit workaround here rather than modifying Phase 5's file.
        import traci
        conn._check_sumo_home()
        binary = "sumo-gui" if self.use_gui else "sumo"
        sumo_cmd = [
            binary,
            "-c", self.sumocfg_path,
            "--seed", str(self.seed),
            "--no-step-log", "true",
            "--time-to-teleport", "300",
            "--tripinfo-output", tripinfo_output_path,
            "--device.emissions.probability", "1.0",
        ]
        traci.start(sumo_cmd)
        conn._connected = True

        steps = 0
        while conn.is_simulation_running() and steps < max_steps:
            conn.step()  # No action taken -- default program runs on its own.
            steps += 1

        conn.close()

        return self._parse_tripinfo(tripinfo_output_path, steps)

    @staticmethod
    def _parse_tripinfo(tripinfo_path: str, total_steps: int) -> dict:
        """
        Parses SUMO's tripinfo XML output into aggregate metrics.
        Each <tripinfo> element represents one completed vehicle trip
        and already contains SUMO-computed waitingTime, timeLoss,
        CO2 emissions (mg), fuel consumption (ml), etc.
        """
        tree = ET.parse(tripinfo_path)
        trips = tree.getroot().findall("tripinfo")

        n = len(trips)
        if n == 0:
            return {"num_vehicles": 0, "episode_steps": total_steps}

        def avg(attr, emission_tag=None):
            if emission_tag:
                values = []
                for t in trips:
                    emissions_elem = t.find("emissions")
                    if emissions_elem is not None:
                        values.append(float(emissions_elem.get(emission_tag)))
                return sum(values) / len(values) if values else 0.0
            return sum(float(t.get(attr)) for t in trips) / n

        return {
            "num_vehicles": n,
            "episode_steps": total_steps,
            "avg_waiting_time": avg("waitingTime"),
            "avg_time_loss": avg("timeLoss"),
            "avg_duration": avg("duration"),
            "avg_route_length": avg("routeLength"),
            "avg_co2_mg": avg(None, emission_tag="CO2_abs"),
            "avg_fuel_ml": avg(None, emission_tag="fuel_abs"),
            "throughput": n,  # vehicles that completed their trip this episode
        }


def run_fixed_time_baseline(scenario_name: str, sumocfg_path: str, seed: int = 42) -> dict:
    """
    Convenience entry point: runs the fixed-time baseline on a given
    scenario and prints + returns the resulting metrics.
    """
    os.makedirs("results/logs", exist_ok=True)
    tripinfo_path = f"results/logs/fixed_time_{scenario_name}_tripinfo.xml"

    controller = FixedTimeController(sumocfg_path, seed=seed)
    metrics = controller.run_episode(tripinfo_path)

    print(f"\n=== Fixed-Time Baseline: {scenario_name} ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    return metrics


if __name__ == "__main__":
    # Quick manual run across all six scenarios.
    import yaml

    with open("configs/scenarios.yaml") as f:
        scenarios = yaml.safe_load(f)["scenarios"]

    all_results = {}
    for sc in scenarios:
        all_results[sc["name"]] = run_fixed_time_baseline(sc["name"], sc["sumocfg"])