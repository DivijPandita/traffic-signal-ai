"""
Phase 1 sanity check: verifies the project structure and that
core dependencies are importable. This is NOT a real unit test yet
(no application logic exists) — it's a scaffolding check.
"""

import os
import importlib


def test_folder_structure():
    required_dirs = [
        "configs", "src/environment", "src/agents", "src/rewards",
        "src/perception", "src/iot", "src/training", "src/evaluation",
        "src/visualization", "sumo/networks", "sumo/routes",
        "sumo/scenarios", "sumo/configs", "models", "data",
        "experiments", "results/logs", "results/metrics",
        "results/plots", "results/tables", "tests",
    ]
    missing = [d for d in required_dirs if not os.path.isdir(d)]
    assert not missing, f"Missing directories: {missing}"
    print("✅ Folder structure OK")


def test_dependencies_importable():
    packages = ["numpy", "pandas", "yaml", "matplotlib", "torch"]
    for pkg in packages:
        importlib.import_module(pkg)
    print("✅ All Phase 1 dependencies import correctly")


def test_config_loads():
    import yaml
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    assert cfg["project"]["name"] == "traffic-signal-ai"
    print("✅ config.yaml loads and parses correctly")


if __name__ == "__main__":
    test_folder_structure()
    test_dependencies_importable()
    test_config_loads()
    print("\n🎉 Phase 1 verification passed.")