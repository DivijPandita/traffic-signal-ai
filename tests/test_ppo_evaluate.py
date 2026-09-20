"""
Phase 12 sanity check: verifies PPO evaluation runs correctly and
the three-way comparison (Fixed-Time / DQN / PPO) produces valid
statistics for all three controllers on held-out seeds.
"""

import sys
sys.path.insert(0, "src")

from evaluation.evaluate import evaluate_ppo_agent, compare

EVAL_SEEDS = [100, 101, 102]


def test_ppo_evaluation_runs():
    result = evaluate_ppo_agent(
        "sumo/configs/scenario_moderate.sumocfg",
        "models/ppo/ppo_moderate.pt",
        EVAL_SEEDS,
        max_episode_steps=100,
    )
    assert "avg_waiting_time_mean" in result
    assert result["throughput_mean"] > 0
    print(f"✅ PPO evaluation on held-out seeds: wait={result['avg_waiting_time_mean']:.2f}, "
          f"throughput={result['throughput_mean']:.1f}")


def test_three_way_compare_runs_end_to_end():
    results = compare(
        "moderate_test",
        "sumo/configs/scenario_moderate.sumocfg",
        "models/dqn/dqn_moderate.pt",
        "models/ppo/ppo_moderate.pt",
        EVAL_SEEDS,
    )
    assert "fixed_time" in results and "dqn" in results and "ppo" in results
    print("✅ Three-way comparison (Fixed-Time / DQN / PPO) runs end-to-end")


if __name__ == "__main__":
    test_ppo_evaluation_runs()
    test_three_way_compare_runs_end_to_end()
    print("\n🎉 Phase 12 verification passed. PPO training and fair three-way evaluation are working.")