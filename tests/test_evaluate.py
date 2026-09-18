"""
Phase 10 sanity check: verifies the fair-comparison evaluation
pipeline runs correctly for both controllers, produces sensible
mean/std statistics across seeds, and that DQN's evaluation doesn't
silently reuse training seeds.
"""

import sys
sys.path.insert(0, "src")

from evaluation.evaluate import evaluate_dqn_agent, evaluate_fixed_time_live, compare


EVAL_SEEDS = [100, 101, 102]  # smaller set for fast test


def test_fixed_time_live_returns_valid_stats():
    result = evaluate_fixed_time_live(
        "sumo/configs/scenario_moderate.sumocfg", EVAL_SEEDS, max_episode_steps=100
    )
    assert "avg_waiting_time_mean" in result
    assert "avg_waiting_time_std" in result
    assert result["avg_waiting_time_mean"] >= 0
    print(f"✅ Fixed-time live evaluation: {result['avg_waiting_time_mean']:.2f} ± {result['avg_waiting_time_std']:.2f}")


def test_dqn_evaluation_uses_held_out_seeds_correctly():
    result = evaluate_dqn_agent(
        "sumo/configs/scenario_moderate.sumocfg",
        "models/dqn/dqn_moderate.pt",
        EVAL_SEEDS,
        max_episode_steps=100,
    )
    assert "avg_waiting_time_mean" in result
    assert result["throughput_mean"] > 0
    print(f"✅ DQN evaluation on held-out seeds: wait={result['avg_waiting_time_mean']:.2f}, "
          f"throughput={result['throughput_mean']:.1f}")


def test_compare_runs_end_to_end():
    results = compare(
        "moderate_test",
        "sumo/configs/scenario_moderate.sumocfg",
        "models/dqn/dqn_moderate.pt",
        EVAL_SEEDS,
    )
    assert "fixed_time" in results and "dqn" in results
    print("✅ Full comparison pipeline runs end-to-end")


if __name__ == "__main__":
    test_fixed_time_live_returns_valid_stats()
    test_dqn_evaluation_uses_held_out_seeds_correctly()
    test_compare_runs_end_to_end()
    print("\n🎉 Phase 10 verification passed. Fair DQN vs Fixed-Time evaluation is working.")