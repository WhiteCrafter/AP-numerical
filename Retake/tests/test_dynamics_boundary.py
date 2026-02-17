from __future__ import annotations

import numpy as np

from src.dynamics import accelerations, boundary_forces
from src.types import CorridorGeometry, SimulationParams


def _corridor() -> CorridorGeometry:
    return CorridorGeometry(
        origin=np.array([0.0, 0.0], dtype=np.float64),
        tangent=np.array([1.0, 0.0], dtype=np.float64),
        normal=np.array([0.0, 1.0], dtype=np.float64),
        length=5.0,
        half_width=1.0,
    )


def _params() -> SimulationParams:
    return SimulationParams(
        dt=0.02,
        t_end=1.0,
        v_max=1.0,
        mass=1.0,
        k_p=0.0,
        k_d=0.0,
        k_rep=0.0,
        r_safe=0.25,
        k_wall=10.0,
        wall_margin=0.2,
    )


def test_boundary_force_pushes_down_near_right_wall() -> None:
    geom = _corridor()
    x = np.array([[0.0, 0.9]], dtype=np.float64)

    forces = boundary_forces(x, geom, k_wall=10.0, wall_margin=0.2)

    np.testing.assert_allclose(forces[0, 0], 0.0)
    np.testing.assert_allclose(forces[0, 1], -1.0)


def test_boundary_force_pushes_up_near_left_wall_with_linear_strength() -> None:
    geom = _corridor()
    x = np.array([[0.0, -0.95]], dtype=np.float64)

    forces = boundary_forces(x, geom, k_wall=10.0, wall_margin=0.2)

    np.testing.assert_allclose(forces[0, 0], 0.0)
    np.testing.assert_allclose(forces[0, 1], 1.5)


def test_accelerations_include_boundary_force_term() -> None:
    geom = _corridor()
    params = _params()
    x = np.array([[0.0, 0.9]], dtype=np.float64)
    v = np.zeros_like(x)
    targets = x.copy()

    acc = accelerations(x, v, targets, params, geom)

    np.testing.assert_allclose(acc, np.array([[0.0, -1.0]], dtype=np.float64))
