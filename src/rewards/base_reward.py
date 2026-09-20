"""
Abstract interface all reward functions implement.

Enforcing a shared compute(metrics, context) signature is what lets
sumo_env.py (Phase 8) swap between the placeholder, static
multi-objective, and context-aware rewards via a single constructor
argument -- no branching logic needed inside the environment itself.
This is the structural foundation of Phase 16's ablation study: only
the reward object changes between experimental conditions, everything
else (environment, agent, evaluation) stays byte-for-byte identical.
"""

from abc import ABC, abstractmethod


class BaseReward(ABC):
    """
    All reward implementations receive:
      - metrics: the dict returned by metrics.get_network_totals()
        (Phase 7), containing total_waiting_time, total_queue_length,
        total_fuel_consumption, total_co2_emission, total_delay,
        emergency_present.
      - context: a string describing the current traffic context
        (e.g. "normal", "congested", "high_emission", "emergency").
        StaticMultiObjectiveReward ignores this argument entirely --
        that's the whole point of it being the static control
        condition. ContextAwareReward (Phase 14) uses it to select
        weights.

    Both must return a single float: the scalar reward for this step.
    """

    def __init__(self, normalization: dict, emergency_bonus: float = 50.0):
        self.norm = normalization
        self.emergency_bonus = emergency_bonus

    def _normalize(self, metrics: dict) -> dict:
        """
        Shared normalization logic used by BOTH static and
        context-aware rewards, so any difference in the final reward
        value is attributable ONLY to weight selection, not to
        differing normalization schemes.
        """
        return {
            "waiting_time": metrics["total_waiting_time"] / self.norm["waiting_time_ref"],
            "queue_length": metrics["total_queue_length"] / self.norm["queue_length_ref"],
            "fuel": metrics["total_fuel_consumption"] / self.norm["fuel_ref"],
            "co2": metrics["total_co2_emission"] / self.norm["co2_ref"],
            "delay": metrics["total_delay"] / self.norm["delay_ref"],
        }

    @abstractmethod
    def compute(self, metrics: dict, context: str = "normal") -> float:
        raise NotImplementedError

    @abstractmethod
    def get_weights(self, context: str = "normal") -> dict:
        """
        Returns the weight dict actually used for a given context.
        Useful for logging/debugging -- e.g. Phase 16's ablation study
        will want to record exactly which weights were active at each
        step for context-aware reward analysis.
        """
        raise NotImplementedError