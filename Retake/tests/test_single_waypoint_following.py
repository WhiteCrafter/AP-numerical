from __future__ import annotations

import numpy as np

from src.scenarios import setup_single_scenario
from src.simulation import run_simulation
from src.types import ScenarioSetup, SimulationParams


def test_setup_single_scenario_stores_full_waypoints_and_explicit_endpoints(
    monkeypatch, tmp_path
) -> None:
    map_path = tmp_path / "dummy_map.png"
    map_path.write_bytes(b"not_used")

    extracted = np.array(
        [
            [0.2, 0.2],
            [0.5, 0.8],
            [0.9, 1.1],
        ],
        dtype=np.float64,
    )

    def fake_extract_waypoints(**_: object) -> np.ndarray:
        return extracted.copy()

    monkeypatch.setattr("src.scenarios.extract_waypoints", fake_extract_waypoints)

    config = {
        "single": {"start": [0.0, 0.0], "goal": [1.0, 1.0]},
        "path_extraction": {
            "map_path": str(map_path),
            "point_a": [0.0, 0.0],
            "point_b": [1.0, 1.0],
            "world_origin": [0.0, 0.0],
            "meters_per_pixel": [0.05, 0.05],
        },
    }

    setup = setup_single_scenario(config)

    assert setup.waypoints is not None
    assert setup.waypoint_indices is not None
    np.testing.assert_allclose(setup.waypoints[0], np.array([0.0, 0.0]))
    np.testing.assert_allclose(setup.waypoints[-1], np.array([1.0, 1.0]))
    np.testing.assert_allclose(setup.targets[0], np.array([0.0, 0.0]))
    np.testing.assert_array_equal(setup.waypoint_indices, np.array([0]))


def test_waypoint_following_changes_motion_vs_final_goal_only() -> None:
    params = SimulationParams(
        dt=0.01,
        t_end=4.0,
        v_max=10.0,
        mass=1.0,
        k_p=6.0,
        k_d=3.0,
        k_rep=0.0,
        r_safe=0.25,
        k_wall=0.0,
        wall_margin=0.0,
    )

    setup_waypoints = ScenarioSetup(
        positions=np.array([[0.0, 0.0]], dtype=np.float64),
        velocities=np.zeros((1, 2), dtype=np.float64),
        targets=np.array([[0.0, 0.0]], dtype=np.float64),
        waypoints=np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64),
        waypoint_indices=np.array([0], dtype=np.int64),
    )
    setup_direct = ScenarioSetup(
        positions=np.array([[0.0, 0.0]], dtype=np.float64),
        velocities=np.zeros((1, 2), dtype=np.float64),
        targets=np.array([[1.0, 1.0]], dtype=np.float64),
    )

    result_waypoints = run_simulation(setup_waypoints, params)
    result_direct = run_simulation(setup_direct, params)

    traj_waypoints = result_waypoints.positions[:, 0, :]
    traj_direct = result_direct.positions[:, 0, :]

    first_high_y_idx = int(np.argmax(traj_waypoints[:, 1] >= 0.8))
    assert traj_waypoints[first_high_y_idx, 0] < 0.2

    direct_high_y_idx = int(np.argmax(traj_direct[:, 1] >= 0.8))
    assert traj_direct[direct_high_y_idx, 0] > 0.7
