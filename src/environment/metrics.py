"""
Live traffic metrics computed step-by-step from TraCI, for vehicles
currently present in the simulation. Unlike SUMO's tripinfo output
(Phase 6), which only reports a vehicle's stats once it has fully
exited the network, these metrics can be queried at ANY point during
a running episode -- which is exactly what the RL state (Phase 8) and
reward function (Phase 13+) need.

All functions here take a `SumoConnection` (or raw `traci` module) and
a set of approach edge IDs, and return per-approach or aggregate values.
"""

import traci


# The four incoming approach edges for our single intersection (Phase 3/4).
# Defined here as the default for this network; multi-intersection
# scenarios (Phase 22) will need a per-junction version of this.
APPROACH_EDGES = {
    "N": "N2C",
    "S": "S2C",
    "E": "E2C",
    "W": "W2C",
}


def get_queue_length(edge_id: str, speed_threshold: float = 0.1) -> int:
    """
    Number of vehicles on the edge currently considered "queued"
    (speed below threshold, i.e. effectively stopped). This is a
    standard proxy for queue length in traffic-signal RL literature --
    we don't need lane geometry to define "queued," just vehicle speed.
    """
    vehicle_ids = traci.edge.getLastStepVehicleIDs(edge_id)
    queued = 0
    for vid in vehicle_ids:
        if traci.vehicle.getSpeed(vid) < speed_threshold:
            queued += 1
    return queued


def get_waiting_time(edge_id: str) -> float:
    """
    Sum of accumulated waiting time (seconds spent at speed < 0.1 m/s)
    across all vehicles currently on this edge. TraCI's edge-level
    getWaitingTime() already aggregates this for us.
    """
    return traci.edge.getWaitingTime(edge_id)


def get_average_speed(edge_id: str) -> float:
    """
    Mean speed (m/s) of vehicles currently on the edge. Used as a proxy
    for "delay" -- lower average speed relative to free-flow speed
    indicates more delay.
    """
    return traci.edge.getLastStepMeanSpeed(edge_id)


def get_vehicle_count(edge_id: str) -> int:
    """Total vehicles currently on the edge (queued or moving)."""
    return traci.edge.getLastStepVehicleNumber(edge_id)


def get_fuel_consumption(edge_id: str) -> float:
    """
    Sum of instantaneous fuel consumption (ml/s) across all vehicles
    currently on the edge, from SUMO's HBEFA3 emission model (the
    same model underlying the vType emissionClass values set in
    Phase 4). This is a live per-second rate, not a cumulative total --
    the reward function (Phase 13+) will use this rate directly, since
    reward is computed per RL step, not per full trip.
    """
    vehicle_ids = traci.edge.getLastStepVehicleIDs(edge_id)
    return sum(traci.vehicle.getFuelConsumption(vid) for vid in vehicle_ids)


def get_co2_emission(edge_id: str) -> float:
    """
    Sum of instantaneous CO2 emission (mg/s) across all vehicles
    currently on the edge. Same live-rate reasoning as fuel above.
    """
    vehicle_ids = traci.edge.getLastStepVehicleIDs(edge_id)
    return sum(traci.vehicle.getCO2Emission(vid) for vid in vehicle_ids)


def get_delay(edge_id: str, free_flow_speed: float = 13.89) -> float:
    """
    Estimated total delay (vehicle-seconds) on this edge this step,
    defined as: for each vehicle, how much slower it's going than
    free-flow speed, summed across vehicles. This is a different
    signal from waiting_time -- a vehicle crawling at 3 m/s is
    "delayed" even if it's not technically "waiting" (speed > 0.1).

    free_flow_speed defaults to 13.89 m/s (50 km/h), matching the
    'road' type's speed attribute set in Phase 3's types.typ.xml.
    """
    vehicle_ids = traci.edge.getLastStepVehicleIDs(edge_id)
    total_delay = 0.0
    for vid in vehicle_ids:
        speed = traci.vehicle.getSpeed(vid)
        speed_deficit = max(0.0, free_flow_speed - speed)
        # Delay this step (1 second) proportional to how far below
        # free-flow speed the vehicle currently is.
        total_delay += speed_deficit / free_flow_speed
    return total_delay


def get_emergency_vehicle_presence(edge_id: str) -> bool:
    """
    True if any vehicle currently on this edge is of vClass "emergency"
    (i.e. an ambulance, per our Phase 4 vType definition). This directly
    feeds the ambulance_detected state variable and the reward
    function's EmergencyBonus term (Phase 13+).
    """
    vehicle_ids = traci.edge.getLastStepVehicleIDs(edge_id)
    for vid in vehicle_ids:
        if traci.vehicle.getVehicleClass(vid) == "emergency":
            return True
    return False


def get_all_approach_metrics(approach_edges: dict = None) -> dict:
    """
    Convenience function: computes every metric above for all four
    approaches in one call, returning a nested dict:

        {
            "N": {"queue_length": 3, "waiting_time": 12.5, ...},
            "S": {...}, "E": {...}, "W": {...}
        }

    This is the primary function state.py (Phase 8) and the reward
    functions (Phase 13+) will call each step.
    """
    if approach_edges is None:
        approach_edges = APPROACH_EDGES

    metrics = {}
    for direction, edge_id in approach_edges.items():
        metrics[direction] = {
            "queue_length": get_queue_length(edge_id),
            "waiting_time": get_waiting_time(edge_id),
            "average_speed": get_average_speed(edge_id),
            "vehicle_count": get_vehicle_count(edge_id),
            "fuel_consumption": get_fuel_consumption(edge_id),
            "co2_emission": get_co2_emission(edge_id),
            "delay": get_delay(edge_id),
            "emergency_present": get_emergency_vehicle_presence(edge_id),
        }
    return metrics


def get_network_totals(approach_edges: dict = None) -> dict:
    """
    Aggregates the per-approach metrics into network-wide totals/averages.
    Useful for logging and for a first, simple version of the reward
    function that doesn't need per-direction granularity.
    """
    per_approach = get_all_approach_metrics(approach_edges)

    total_queue = sum(m["queue_length"] for m in per_approach.values())
    total_waiting = sum(m["waiting_time"] for m in per_approach.values())
    total_vehicles = sum(m["vehicle_count"] for m in per_approach.values())
    total_fuel = sum(m["fuel_consumption"] for m in per_approach.values())
    total_co2 = sum(m["co2_emission"] for m in per_approach.values())
    total_delay = sum(m["delay"] for m in per_approach.values())
    any_emergency = any(m["emergency_present"] for m in per_approach.values())

    avg_speed_values = [m["average_speed"] for m in per_approach.values() if m["vehicle_count"] > 0]
    avg_speed = sum(avg_speed_values) / len(avg_speed_values) if avg_speed_values else 0.0

    return {
        "total_queue_length": total_queue,
        "total_waiting_time": total_waiting,
        "total_vehicles": total_vehicles,
        "total_fuel_consumption": total_fuel,
        "total_co2_emission": total_co2,
        "total_delay": total_delay,
        "average_speed": avg_speed,
        "emergency_present": any_emergency,
        "per_approach": per_approach,
    }