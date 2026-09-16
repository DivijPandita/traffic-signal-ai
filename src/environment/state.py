"""
Builds the RL state (observation) vector from live traffic metrics
(metrics.py) and the current traffic light phase.

State layout (all values normalized to roughly [0, 1] or [0, a few]
so no single feature dominates the neural network's input by scale):

  For each of the 4 approaches (N, S, E, W), in that fixed order:
    - normalized queue length
    - normalized waiting time
    - normalized average speed (relative to free-flow speed)
    - normalized vehicle count

  Followed by:
    - one-hot encoding of the CURRENTLY ACTIVE green phase
      (length = number of green phases)
    - emergency vehicle presence flag (0.0 or 1.0)

Total length = 4*4 + num_green_phases + 1
"""

import numpy as np
import traci

from environment.metrics import get_all_approach_metrics

# Normalization constants. These are rough scale references, not hard
# caps -- values can exceed 1.0 in extreme congestion, which is fine
# for a neural network input (it does not need strict [0,1] bounding).
MAX_QUEUE_REFERENCE = 20.0       # vehicles
MAX_WAITING_TIME_REFERENCE = 60.0  # seconds
FREE_FLOW_SPEED = 13.89          # m/s, matches Phase 3 road type
MAX_VEHICLE_COUNT_REFERENCE = 20.0  # vehicles

DIRECTIONS = ["N", "S", "E", "W"]


def build_state_vector(tls_id: str, green_phases: list) -> np.ndarray:
    """
    Constructs the full observation vector for the current simulation
    instant. `green_phases` is the ActionSpace's list of green phase
    indices (Phase 8, actions.py), used to build the one-hot encoding.
    """
    metrics = get_all_approach_metrics()

    features = []
    for direction in DIRECTIONS:
        m = metrics[direction]
        features.append(min(m["queue_length"] / MAX_QUEUE_REFERENCE, 3.0))
        features.append(min(m["waiting_time"] / MAX_WAITING_TIME_REFERENCE, 3.0))
        features.append(m["average_speed"] / FREE_FLOW_SPEED if m["vehicle_count"] > 0 else 1.0)
        features.append(min(m["vehicle_count"] / MAX_VEHICLE_COUNT_REFERENCE, 3.0))

    # One-hot encoding of the currently active green phase.
    current_phase = traci.trafficlight.getPhase(tls_id)
    one_hot = [0.0] * len(green_phases)
    if current_phase in green_phases:
        one_hot[green_phases.index(current_phase)] = 1.0
    features.extend(one_hot)

    # Network-wide emergency presence flag.
    emergency = 1.0 if any(m["emergency_present"] for m in metrics.values()) else 0.0
    features.append(emergency)

    return np.array(features, dtype=np.float32)


def get_state_dim(num_green_phases: int) -> int:
    """Total length of the state vector, needed to size the neural network's input layer (Phase 9)."""
    return 4 * 4 + num_green_phases + 1