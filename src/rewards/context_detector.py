"""
Rule-based traffic context classifier.

Given live network metrics, decides which of four contexts is
currently active. Emergency is an absolute override -- if an
emergency vehicle is present, that always wins regardless of other
conditions. Otherwise, "congested" and "high_emission" compete based
on SEVERITY (how far each metric exceeds its threshold, as a ratio),
not a fixed checking order -- this avoids one condition always
masking the other when both are true simultaneously, which happened
with naive priority-list ordering (see Phase 14 diagnostic history).
"""


class ContextDetector:
    def __init__(self, thresholds: dict):
        self.thresholds = thresholds

    def detect(self, metrics: dict) -> str:
        if metrics.get("emergency_present", False):
            return "emergency"

        congestion_ratio = max(
            metrics["total_queue_length"] / self.thresholds["congested_queue_length"],
            metrics["total_waiting_time"] / self.thresholds["congested_waiting_time"],
        )
        emission_ratio = metrics["total_co2_emission"] / self.thresholds["high_emission_co2"]

        if congestion_ratio < 1.0 and emission_ratio < 1.0:
            return "normal"

        # Whichever condition is exceeded more severely wins.
        return "congested" if congestion_ratio >= emission_ratio else "high_emission"