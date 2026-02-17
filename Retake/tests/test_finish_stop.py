from __future__ import annotations

import numpy as np

from src.simulation import run_simulation
from src.types import ScenarioSetup, SimulationParams


def _params() -> SimulationParams:
    return SimulationParams(
        dt=0.02,
        t_end=4.0,
        v_max=1.2,
        mass=1.0,
        k_p=4.0,
        k_d=1.0,
        k_rep=0.0,
        r_safe=0.2,
        k_wall=0.0,
        wall_margin=0.0,
    )


def test_single_agent_stops_near_finish_radius() -> None:
    setup = ScenarioSetup(
        positions=np.array([[0.0, 0.0]], dtype=np.float64),
        velocities=np.zeros((1, 2), dtype=np.float64),
        targets=np.array([[2.0, 0.0]], dtype=np.float64),
        terminal_targets=np.array([[2.0, 0.0]], dtype=np.float64),
        stop_radius=0.15,
    )

    result = run_simulation(setup, _params())

    final_pos = result.positions[-1, 0]
    final_vel = result.velocities[-1, 0]
    assert np.linalg.norm(final_pos - np.array([2.0, 0.0])) <= 0.15 + 1e-6
    np.testing.assert_allclose(final_vel, np.zeros(2), atol=1e-8)


def test_swarm_agents_stop_at_respective_finish_lines() -> None:
    positions = np.array([[0.0, -0.2], [2.0, 0.2]], dtype=np.float64)
    terminal_targets = np.array([[2.0, -0.2], [0.0, 0.2]], dtype=np.float64)

    setup = ScenarioSetup(
        positions=positions,
        velocities=np.zeros_like(positions),
        targets=terminal_targets.copy(),
        terminal_targets=terminal_targets,
        stop_radius=0.2,
    )

    result = run_simulation(setup, _params())

    final_pos = result.positions[-1]
    final_vel = result.velocities[-1]
    distances = np.linalg.norm(final_pos - terminal_targets, axis=1)
    assert np.all(distances <= 0.2 + 1e-6)
    np.testing.assert_allclose(final_vel, np.zeros_like(final_vel), atol=1e-8)
