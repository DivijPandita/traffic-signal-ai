"""
Phase 2 sanity check (WSL/Linux): verifies SUMO is installed, SUMO_HOME
is set, TraCI/sumolib are importable, and the sumo binary is locatable —
the full chain the RL environment will depend on.
"""

import os
import subprocess
import shutil


def test_sumo_home_set():
    sumo_home = os.environ.get("SUMO_HOME")
    assert sumo_home, "SUMO_HOME is not set. Did you source ~/.bashrc? See Phase 2 Step B."
    assert os.path.isdir(sumo_home), f"SUMO_HOME points to a non-existent path: {sumo_home}"
    print(f"✅ SUMO_HOME is set: {sumo_home}")


def test_sumo_binary_runs():
    result = subprocess.run(["sumo", "--version"], capture_output=True, text=True)
    assert result.returncode == 0, "Could not run 'sumo --version'. Check apt install (Step A)."
    print("✅ sumo binary runs. Version info:")
    print(result.stdout.splitlines()[0])


def test_traci_importable():
    import traci  # noqa: F401
    import sumolib  # noqa: F401
    print("✅ traci and sumolib import correctly")


def test_sumo_binary_locatable():
    sumo_path = shutil.which("sumo")
    assert sumo_path, "sumo not found on PATH."
    print(f"✅ sumo located at {sumo_path}")
    print("   (Full TraCI connection test with a real network happens in Phase 5.)")


if __name__ == "__main__":
    test_sumo_home_set()
    test_sumo_binary_runs()
    test_traci_importable()
    test_sumo_binary_locatable()
    print("\n🎉 Phase 2 verification passed. SUMO + TraCI are ready.")