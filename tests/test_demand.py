"""
Phase 4 sanity check: runs all six scenarios headless, collects trip
info, and checks that (a) vehicles are actually generated per scenario,
(b) demand ordering (low < moderate < heavy < peak) holds for waiting
time, and (c) the emergency scenario contains exactly one ambulance.
"""

import subprocess
import xml.etree.ElementTree as ET
import tempfile
import os

SCENARIOS = ["low", "moderate", "heavy", "peak", "emergency", "high_emission"]


def run_scenario_and_get_tripinfo(name):
    cfg = f"sumo/configs/scenario_{name}.sumocfg"
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp:
        tripinfo_path = tmp.name

    result = subprocess.run(
        ["sumo", "-c", cfg, "--no-step-log", "true",
         "--tripinfo-output", tripinfo_path],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"SUMO failed on {name}: {result.stderr}"

    tree = ET.parse(tripinfo_path)
    trips = tree.getroot().findall("tripinfo")
    os.remove(tripinfo_path)
    return trips


def test_all_scenarios_generate_vehicles():
    for name in SCENARIOS:
        trips = run_scenario_and_get_tripinfo(name)
        assert len(trips) > 0, f"Scenario '{name}' produced zero vehicles."
        print(f"✅ Scenario '{name}': {len(trips)} vehicles completed")


def test_demand_ordering_by_waiting_time():
    avg_wait = {}
    for name in ["low", "moderate", "heavy", "peak"]:
        trips = run_scenario_and_get_tripinfo(name)
        waits = [float(t.get("waitingTime")) for t in trips]
        avg_wait[name] = sum(waits) / len(waits)

    print(f"Average waiting times: {avg_wait}")
    assert avg_wait["low"] <= avg_wait["moderate"] <= avg_wait["heavy"] <= avg_wait["peak"], (
        "Expected waiting time to increase with demand level."
    )
    print("✅ Waiting time increases with demand level as expected")


def test_emergency_scenario_has_one_ambulance():
    trips = run_scenario_and_get_tripinfo("emergency")
    ambulances = [t for t in trips if t.get("vType") == "ambulance"]
    assert len(ambulances) == 1, f"Expected exactly 1 ambulance, found {len(ambulances)}"
    print("✅ Emergency scenario contains exactly one ambulance trip")


def test_high_emission_scenario_has_trucks_and_buses():
    trips = run_scenario_and_get_tripinfo("high_emission")
    types_present = {t.get("vType") for t in trips}
    assert "truck" in types_present and "bus" in types_present, (
        f"Expected truck and bus in high_emission scenario, found types: {types_present}"
    )
    print(f"✅ high_emission scenario contains vehicle types: {types_present}")


if __name__ == "__main__":
    test_all_scenarios_generate_vehicles()
    test_demand_ordering_by_waiting_time()
    test_emergency_scenario_has_one_ambulance()
    test_high_emission_scenario_has_trucks_and_buses()
    print("\n🎉 Phase 4 verification passed. Traffic demand and routes are ready.")