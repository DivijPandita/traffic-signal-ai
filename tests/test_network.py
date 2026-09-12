"""
Phase 3 sanity check: verifies the single-intersection network compiled
correctly — right junction ID, right number of edges, traffic light
present — and that SUMO can load the full .sumocfg without errors.
"""

import os
import subprocess
import sumolib


NET_PATH = "sumo/networks/single_intersection/single_intersection.net.xml"
SUMOCFG_PATH = "sumo/configs/single_intersection.sumocfg"


def test_net_file_exists():
    assert os.path.isfile(NET_PATH), (
        f"{NET_PATH} not found. Did you run netconvert (Section 6)?"
    )
    print("✅ single_intersection.net.xml exists")


def test_junction_c_is_traffic_light():
    net = sumolib.net.readNet(NET_PATH)
    junction = net.getNode("C")
    assert junction is not None, "Junction 'C' not found in network."
    assert junction.getType() == "traffic_light", (
        f"Expected junction 'C' to be a traffic_light, got '{junction.getType()}'"
    )
    print("✅ Junction 'C' exists and is a traffic_light")


def test_four_approaches_exist():
    net = sumolib.net.readNet(NET_PATH)
    incoming_ids = {"N2C", "S2C", "E2C", "W2C"}
    outgoing_ids = {"C2N", "C2S", "C2E", "C2W"}
    all_edge_ids = {e.getID() for e in net.getEdges()}

    missing = (incoming_ids | outgoing_ids) - all_edge_ids
    assert not missing, f"Missing expected edges: {missing}"
    print("✅ All 8 approach/exit edges present (4 incoming, 4 outgoing)")


def test_edges_have_two_lanes():
    net = sumolib.net.readNet(NET_PATH)
    edge = net.getEdge("N2C")
    assert edge.getLaneNumber() == 2, (
        f"Expected 2 lanes on N2C, got {edge.getLaneNumber()}"
    )
    print("✅ Edges have the expected 2 lanes")


def test_sumocfg_runs_headless():
    result = subprocess.run(
        ["sumo", "-c", SUMOCFG_PATH, "--no-step-log", "true"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"SUMO failed to run the config. stderr:\n{result.stderr}"
    )
    print("✅ SUMO successfully loaded and ran the single_intersection config")


if __name__ == "__main__":
    test_net_file_exists()
    test_junction_c_is_traffic_light()
    test_four_approaches_exist()
    test_edges_have_two_lanes()
    test_sumocfg_runs_headless()
    print("\n🎉 Phase 3 verification passed. Single-intersection network is ready.")