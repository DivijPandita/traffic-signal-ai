"""
Phase 7 sanity check: verifies live metrics can be computed mid-simulation
(not just at episode end), that values behave sensibly as traffic builds
up, and that emergency vehicle detection fires at the correct time in
the emergency scenario.
"""

import sys
sys.path.insert(0, "src")

from environment.sumo_connection import SumoConnection
from environment.metrics import get_all_approach_metrics, get_network_totals


def test_metrics_readable_at_step_zero():
    conn = SumoConnection("sumo/configs/scenario_low.sumocfg", seed=42)
    conn.start()
    metrics = get_network_totals()
    assert metrics["total_vehicles"] == 0, "Expected 0 vehicles at t=0."
    assert metrics["total_queue_length"] == 0
    print(f"✅ Metrics readable at step 0: {metrics['total_vehicles']} vehicles")
    conn.close()


def test_metrics_grow_as_traffic_builds():
    conn = SumoConnection("sumo/configs/scenario_heavy.sumocfg", seed=42)
    conn.start()

    for _ in range(30):
        conn.step()
    early = get_network_totals()

    for _ in range(300):
        conn.step()
    later = get_network_totals()

    print(f"Early (t=30): vehicles={early['total_vehicles']}, queue={early['total_queue_length']}")
    print(f"Later (t=330): vehicles={later['total_vehicles']}, queue={later['total_queue_length']}")

    assert later["total_vehicles"] >= early["total_vehicles"], (
        "Expected vehicle count to grow or stay stable, not shrink, in heavy traffic early on."
    )
    print("✅ Metrics reflect traffic building up over time")
    conn.close()


def test_per_approach_metrics_have_all_four_directions():
    conn = SumoConnection("sumo/configs/scenario_moderate.sumocfg", seed=42)
    conn.start()
    for _ in range(100):
        conn.step()
    metrics = get_all_approach_metrics()
    assert set(metrics.keys()) == {"N", "S", "E", "W"}
    for direction, m in metrics.items():
        assert "queue_length" in m
        assert "waiting_time" in m
        assert "fuel_consumption" in m
        assert "co2_emission" in m
        assert "delay" in m
        assert "emergency_present" in m
    print(f"✅ Per-approach metrics present for all 4 directions: {list(metrics.keys())}")
    conn.close()

def test_emergency_detection_fires_at_correct_time():
    """
    The emergency scenario (Phase 4) inserts a single ambulance at
    depart=300s on route N_S. The ambulance is fast (speedFactor=1.3)
    and covers the ~200m N2C approach edge in roughly 10-12 seconds,
    so we must check EVERY step in a window around t=300, not skip
    ahead in large blocks, or we can miss the entire detection window.
    """
    conn = SumoConnection("sumo/configs/scenario_emergency.sumocfg", seed=42)
    conn.start()

    # Step up to just before the scheduled departure.
    for _ in range(298):
        conn.step()

    detected = False
    for _ in range(60):  # scan one step at a time from ~t=298 to ~t=358
        conn.step()
        if get_network_totals()["emergency_present"]:
            detected = True
            break

    assert detected, "Ambulance was never detected on any approach edge after departure."
    print(f"✅ Emergency vehicle detected on an approach edge at t={conn.get_current_time():.0f}")
    conn.close()


def test_fuel_and_co2_are_nonnegative():
    conn = SumoConnection("sumo/configs/scenario_high_emission.sumocfg", seed=42)
    conn.start()
    for _ in range(200):
        conn.step()
    metrics = get_network_totals()
    assert metrics["total_fuel_consumption"] >= 0
    assert metrics["total_co2_emission"] >= 0
    print(f"✅ Live fuel/CO2 metrics: fuel={metrics['total_fuel_consumption']:.2f} ml/s, "
          f"co2={metrics['total_co2_emission']:.2f} mg/s")
    conn.close()


if __name__ == "__main__":
    test_metrics_readable_at_step_zero()
    test_metrics_grow_as_traffic_builds()
    test_per_approach_metrics_have_all_four_directions()
    test_emergency_detection_fires_at_correct_time()
    test_fuel_and_co2_are_nonnegative()
    print("\n🎉 Phase 7 verification passed. Live metrics collection is working.")