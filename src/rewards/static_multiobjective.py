"""
Static multi-objective reward: implements

    R_t = -(w1*WaitingTime + w2*QueueLength + w3*Fuel + w4*CO2 + w5*Delay)
          + EmergencyBonus

with FIXED weights that never change based on traffic context. This
is the ablation study's control condition -- see Phase 13 rationale.
"""

from rewards.base_reward import BaseReward


class StaticMultiObjectiveReward(BaseReward):
    def __init__(self, normalization: dict, weights: dict, emergency_bonus: float = 50.0):
        super().__init__(normalization, emergency_bonus)
        self.weights = weights

    def compute(self, metrics: dict, context: str = "normal") -> float:
        # context is intentionally ignored -- weights never adapt.
        n = self._normalize(metrics)

        weighted_cost = (
            self.weights["w1_waiting_time"] * n["waiting_time"]
            + self.weights["w2_queue_length"] * n["queue_length"]
            + self.weights["w3_fuel"] * n["fuel"]
            + self.weights["w4_co2"] * n["co2"]
            + self.weights["w5_delay"] * n["delay"]
        )

        reward = -weighted_cost

        if metrics.get("emergency_present", False):
            reward += self.emergency_bonus

        return reward

    def get_weights(self, context: str = "normal") -> dict:
        return dict(self.weights)  # same weights regardless of context