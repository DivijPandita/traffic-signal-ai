"""
Diagnostic script (Phase 7 debugging): tracks the ambulance vehicle
second-by-second to find out exactly when/where it appears, to
diagnose why emergency detection failed in test_metrics.py.
"""

import sys
sys.path.insert(0, "src")

import traci
from environment.sumo_connection import SumoConnection

conn = SumoConnection("sumo/configs/scenario_emergency.sumocfg", seed=42)
conn.start()

seen_departed = False
for step in range(0, 400):
    conn.step()
    t = conn.get_current_time()

    all_ids = traci.vehicle.getIDList()
    if "ambulance_1" in all_ids:
        seen_departed = True
        edge = traci.vehicle.getRoadID("ambulance_1")
        speed = traci.vehicle.getSpeed("ambulance_1")
        vclass = traci.vehicle.getVehicleClass("ambulance_1")
        print(f"t={t:.0f}  ambulance_1 on edge='{edge}' speed={speed:.2f} vClass={vclass}")
    elif seen_departed:
        # It departed before but is no longer in the vehicle list --
        # either it finished its trip or something went wrong.
        print(f"t={t:.0f}  ambulance_1 no longer present (trip finished or removed)")
        break

if not seen_departed:
    print("Ambulance never appeared in traci.vehicle.getIDList() at all.")
    print("Departed IDs seen so far this run:", traci.simulation.getDepartedIDList())

conn.close()