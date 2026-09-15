"""
Phase 6 sanity check: verifies the Fixed-Time baseline controller runs
a full episode, produces a non-empty tripinfo file, parses it into
sensible aggregate metrics, and that demand ordering (low < peak)
holds for waiting time under this baseline specifically.
"""

import sys
import os
sys.path.insert(0, "src")

from evaluation.baselines import run_fixed_time_baseline


def test_fixed_time_runs_low_scenario():
    metrics = run_fixed_time_baseline("low", "sumo/configs/scenario_low.sumocfg")
    assert metrics["num_vehicles"] > 0, "No vehicles completed trips."
    assert metrics["avg_waiting_time"] >= 0
    assert metrics["avg_co2_mg"] > 0, "Expected non-zero CO2 emissions."
    assert metrics["avg_fuel_ml"] > 0, "Expected non-zero fuel consumption."
    print(f"✅ Fixed-time baseline ran on 'low' scenario: {metrics}")


def test_tripinfo_file_created():
    path = "results/logs/fixed_time_low_tripinfo.xml"
    assert os.path.isfile(path), f"Expected tripinfo file at {path}"
    assert os.path.getsize(path) > 0, "tripinfo file is empty."
    print("✅ tripinfo XML file created and non-empty")


def test_waiting_time_increases_with_demand():
    low = run_fixed_time_baseline("low", "sumo/configs/scenario_low.sumocfg")
    peak = run_fixed_time_baseline("peak", "sumo/configs/scenario_peak.sumocfg")
    assert peak["avg_waiting_time"] > low["avg_waiting_time"], (
        "Expected peak scenario to have higher waiting time than low under fixed-time control."
    )
    print(f"✅ Waiting time under fixed-time control: low={low['avg_waiting_time']:.2f}, "
          f"peak={peak['avg_waiting_time']:.2f}")


if __name__ == "__main__":
    test_fixed_time_runs_low_scenario()
    test_tripinfo_file_created()
    test_waiting_time_increases_with_demand()
    print("\n🎉 Phase 6 verification passed. Fixed-time baseline is working.")