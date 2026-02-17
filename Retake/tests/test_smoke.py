from __future__ import annotations

import subprocess



def run_scenario(name: str) -> str:
    cmd = ["python", "main.py", "--scenario", name, "--config", "configs/default.json"]
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return completed.stdout



def test_single_smoke() -> None:
    output = run_scenario("single")
    assert "Saved:" in output
    assert "mean_final_goal_error" in output



def test_swarm_smoke() -> None:
    output = run_scenario("swarm")
    assert "Saved:" in output
    assert "min_pairwise_distance" in output
