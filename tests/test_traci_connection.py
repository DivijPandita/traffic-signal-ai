"""
Phase 5 sanity check: verifies we can start a SUMO process via TraCI,
step it forward, read live state (time, vehicles, traffic light phase,
edge metrics), force a phase change, and shut down cleanly.
"""

import sys
sys.path.insert(0, "src")

from environment.sumo_connection import SumoConnection


SUMOCFG = "sumo/configs/scenario_low.sumocfg"


def test_connection_lifecycle():
    conn = SumoConnection(SUMOCFG, use_gui=False, seed=42)
    conn.start()
    assert conn.get_current_time() == 0.0
    conn.close()
    print("✅ Connection starts and closes cleanly")


def test_simulation_steps_forward():
    conn = SumoConnection(SUMOCFG, use_gui=False, seed=42)
    conn.start()
    for _ in range(10):
        conn.step()
    t = conn.get_current_time()
    assert t == 10.0, f"Expected time=10.0 after 10 steps, got {t}"
    conn.close()
    print(f"✅ Simulation steps forward correctly (time={t})")


def test_traffic_light_present_and_readable():
    conn = SumoConnection(SUMOCFG, use_gui=False, seed=42)
    conn.start()
    tls_ids = conn.get_traffic_light_ids()
    assert tls_ids == ["C"], f"Expected traffic light ['C'], got {tls_ids}"
    phase = conn.get_traffic_light_phase("C")
    assert isinstance(phase, int)
    print(f"✅ Traffic light 'C' found, current phase index = {phase}")
    conn.close()


def test_vehicles_appear_over_time():
    conn = SumoConnection(SUMOCFG, use_gui=False, seed=42)
    conn.start()
    max_vehicles_seen = 0
    for _ in range(200):
        conn.step()
        max_vehicles_seen = max(max_vehicles_seen, len(conn.get_vehicle_ids()))
    assert max_vehicles_seen > 0, "No vehicles appeared in 200 steps."
    print(f"✅ Vehicles appear over time (max concurrent seen: {max_vehicles_seen})")
    conn.close()


def test_edge_metrics_readable():
    conn = SumoConnection(SUMOCFG, use_gui=False, seed=42)
    conn.start()
    for _ in range(100):
        conn.step()
    count = conn.get_edge_vehicle_count("N2C")
    wait = conn.get_edge_waiting_time("N2C")
    assert isinstance(count, int)
    assert isinstance(wait, float)
    print(f"✅ Edge metrics readable: N2C vehicle_count={count}, waiting_time={wait}")
    conn.close()


def test_set_traffic_light_phase():
    conn = SumoConnection(SUMOCFG, use_gui=False, seed=42)
    conn.start()
    conn.set_traffic_light_phase("C", 0)
    conn.step()
    phase = conn.get_traffic_light_phase("C")
    assert phase == 0, f"Expected phase 0 after setting it, got {phase}"
    print("✅ Traffic light phase can be set via TraCI")
    conn.close()


def test_full_episode_runs_to_completion():
    conn = SumoConnection(SUMOCFG, use_gui=False, seed=42)
    conn.start()
    steps = 0
    while conn.is_simulation_running() and steps < 4000:
        conn.step()
        steps += 1
    assert steps < 4000, "Simulation did not finish within expected step budget."
    print(f"✅ Full episode completed in {steps} steps")
    conn.close()


if __name__ == "__main__":
    test_connection_lifecycle()
    test_simulation_steps_forward()
    test_traffic_light_present_and_readable()
    test_vehicles_appear_over_time()
    test_edge_metrics_readable()
    test_set_traffic_light_phase()
    test_full_episode_runs_to_completion()
    print("\n🎉 Phase 5 verification passed. Python <-> SUMO TraCI connection is working.")