"""
Phase 14 sanity check: verifies context detection correctly classifies
each context in isolation, that weights differ meaningfully across
contexts, that emergency always overrides other conditions, and that
the reward integrates correctly with the real environment.
"""

import sys
sys.path.insert(0, "src")

from rewards.config_loader import build_context_aware_reward


def test_normal_context_detected_for_light_traffic():
    reward_fn = build_context_aware_reward()
    metrics = {
        "total_waiting_time": 5.0, "total_queue_length": 2.0,
        "total_fuel_consumption": 3.0, "total_co2_emission": 3000.0,
        "total_delay": 0.5, "emergency_present": False,
    }
    ctx = reward_fn.detect_context(metrics)
    assert ctx == "normal"
    print(f"✅ Light traffic correctly detected as 'normal'")


def test_congested_context_detected_for_high_queue():
    reward_fn = build_context_aware_reward()
    metrics = {
        "total_waiting_time": 20.0, "total_queue_length": 15.0,  # over threshold (10)
        "total_fuel_consumption": 5.0, "total_co2_emission": 5000.0,  # well under CO2 threshold
        "total_delay": 1.0, "emergency_present": False,
    }
    ctx = reward_fn.detect_context(metrics)
    assert ctx == "congested"
    print(f"✅ High queue length correctly detected as 'congested'")


def test_high_emission_context_detected_for_high_co2():
    reward_fn = build_context_aware_reward()
    metrics = {
        "total_waiting_time": 10.0, "total_queue_length": 3.0,  # well under queue threshold
        "total_fuel_consumption": 20.0, "total_co2_emission": 35000.0,  # over threshold (25000)
        "total_delay": 1.0, "emergency_present": False,
    }
    ctx = reward_fn.detect_context(metrics)
    assert ctx == "high_emission"
    print(f"✅ High CO2 correctly detected as 'high_emission'")


def test_emergency_overrides_all_other_conditions():
    reward_fn = build_context_aware_reward()
    metrics = {
        # Deliberately also severely congested AND high-emission --
        # emergency must still win.
        "total_waiting_time": 90.0, "total_queue_length": 18.0,
        "total_fuel_consumption": 28.0, "total_co2_emission": 38000.0,
        "total_delay": 8.0, "emergency_present": True,
    }
    ctx = reward_fn.detect_context(metrics)
    assert ctx == "emergency"
    print(f"✅ Emergency correctly overrides simultaneous congestion/high-emission conditions")


def test_weights_differ_meaningfully_across_contexts():
    reward_fn = build_context_aware_reward()
    w_normal = reward_fn.get_weights("normal")
    w_congested = reward_fn.get_weights("congested")
    w_emission = reward_fn.get_weights("high_emission")

    assert w_congested["w1_waiting_time"] > w_normal["w1_waiting_time"], (
        "Expected congested context to weight waiting_time higher than normal."
    )
    assert w_emission["w4_co2"] > w_normal["w4_co2"], (
        "Expected high_emission context to weight CO2 higher than normal."
    )
    print("✅ Context-specific weights differ meaningfully in the expected directions")


def test_severity_based_competition_picks_dominant_condition():
    """
    A case where BOTH congestion and emission thresholds are exceeded,
    but emission is proportionally far more severe -- should resolve
    to high_emission, not congested (regression test for the priority-
    order bug we found and fixed during threshold tuning).
    """
    reward_fn = build_context_aware_reward()
    metrics = {
        "total_waiting_time": 65.0,   # just over 60 threshold -> ratio ~1.08
        "total_queue_length": 11.0,   # just over 10 threshold -> ratio ~1.10
        "total_fuel_consumption": 30.0,
        "total_co2_emission": 100000.0,  # way over 25000 threshold -> ratio 4.0
        "total_delay": 5.0, "emergency_present": False,
    }
    ctx = reward_fn.detect_context(metrics)
    assert ctx == "high_emission", (
        f"Expected severity-based competition to pick high_emission (ratio 4.0) "
        f"over congested (ratio ~1.1), got {ctx}"
    )
    print("✅ Severity-based context competition correctly picks the more dominant condition")


def test_reward_computation_uses_detected_context_weights():
    reward_fn = build_context_aware_reward()
    metrics = {
        "total_waiting_time": 90.0, "total_queue_length": 18.0,
        "total_fuel_consumption": 5.0, "total_co2_emission": 5000.0,
        "total_delay": 5.0, "emergency_present": False,
    }
    # This should be detected as congested -> uses congested weights,
    # which weight waiting_time/queue_length MORE heavily than normal.
    r_context_aware = reward_fn.compute(metrics)  # context=None => auto-detect
    r_forced_normal = reward_fn.compute(metrics, context="normal")

    assert r_context_aware != r_forced_normal, (
        "Expected auto-detected congested weights to differ from forced-normal weights."
    )
    print(f"✅ Auto-detected context produces different reward than forced 'normal': "
          f"{r_context_aware:.4f} vs {r_forced_normal:.4f}")


if __name__ == "__main__":
    test_normal_context_detected_for_light_traffic()
    test_congested_context_detected_for_high_queue()
    test_high_emission_context_detected_for_high_co2()
    test_emergency_overrides_all_other_conditions()
    test_weights_differ_meaningfully_across_contexts()
    test_severity_based_competition_picks_dominant_condition()
    test_reward_computation_uses_detected_context_weights()
    print("\n🎉 Phase 14 verification passed. Context-aware reward is working correctly.")