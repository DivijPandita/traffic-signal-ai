"""
Context-Aware Multi-Objective Reward Function -- the project's
primary research contribution.

Structurally identical to StaticMultiObjectiveReward (same base class,
same normalization, same five-term formula, same emergency bonus
mechanism). The ONLY difference: instead of one fixed weight dict,
this class selects a weight dict based on the currently detected
traffic context, via ContextDetector. This is what makes Phase 16's
ablation study ("does context-awareness itself help?") a controlled,
single-variable comparison.
"""

from rewards.base_reward import BaseReward
from rewards.context_detector import ContextDetector


class ContextAwareReward(BaseReward):
    def __init__(
        self,
        normalization: dict,
        context_weights: dict,
        context_thresholds: dict,
        emergency_bonus: float = 50.0,
    ):
        super().__init__(normalization, emergency_bonus)
        self.context_weights = context_weights
        self.detector = ContextDetector(context_thresholds)

    def compute(self, metrics: dict, context: str = None) -> float:
        """
        If context is not explicitly provided, it is detected live
        from the metrics themselves -- this is the normal usage
        pattern in sumo_env.py. An explicit context can still be
        passed (useful for testing specific contexts in isolation,
        as the verification tests below do).
        """
        if context is None:
            context = self.detector.detect(metrics)

        weights = self.context_weights[context]
        n = self._normalize(metrics)

        weighted_cost = (
            weights["w1_waiting_time"] * n["waiting_time"]
            + weights["w2_queue_length"] * n["queue_length"]
            + weights["w3_fuel"] * n["fuel"]
            + weights["w4_co2"] * n["co2"]
            + weights["w5_delay"] * n["delay"]
        )

        reward = -weighted_cost

        if metrics.get("emergency_present", False):
            reward += self.emergency_bonus

        return reward

    def get_weights(self, context: str = "normal") -> dict:
        return dict(self.context_weights[context])

    def detect_context(self, metrics: dict) -> str:
        """Exposed separately so training/logging code can record which context was active at each step."""
        return self.detector.detect(metrics)