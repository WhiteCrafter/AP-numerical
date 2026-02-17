from __future__ import annotations

import numpy as np

from src.path_curve import build_parametric_spline
from src.simulation import run_simulation
from src.types import ScenarioSetup, SimulationParams


def test_vt_path_following_tracks_curve_within_radius() -> None:
    curve = build_parametric_spline(np.array([[0.0, 0.0], [0.0, 1.5], [1.5, 1.5]], dtype=np.float64))
    params = SimulationParams(
        dt=0.02,
        t_end=4.0,
        v_max=1.0,
        mass=1.0,
        k_p=2.5,
        k_d=1.0,
        k_rep=0.0,
        r_safe=0.2,
        k_wall=0.0,
        wall_margin=0.0,
    )
    setup = ScenarioSetup(
        positions=np.array([[0.0, -0.6]], dtype=np.float64),
        velocities=np.zeros((1, 2), dtype=np.float64),
        targets=np.array([[1.5, 1.5]], dtype=np.float64),
        path_curve=curve,
        path_radius=0.4,
    )

    result = run_simulation(setup, params)
    final = result.positions[-1, 0]

    s_proj, closest = curve.project(np.array([final], dtype=np.float64))
    cross_track = float(np.linalg.norm(final - closest[0]))

    assert s_proj[0] > 0.4 * curve.length
    assert cross_track < 0.5
