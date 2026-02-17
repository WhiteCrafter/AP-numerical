from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
VelocityField = Callable[[FloatArray, float], FloatArray]

if TYPE_CHECKING:
    from .path_curve import ParametricSpline2D


@dataclass
class SimulationParams:
    dt: float
    t_end: float
    v_max: float
    mass: float
    k_p: float
    k_d: float
    k_rep: float
    r_safe: float
    k_wall: float
    wall_margin: float


@dataclass
class CorridorGeometry:
    origin: FloatArray
    tangent: FloatArray
    normal: FloatArray
    length: float
    half_width: float


@dataclass
class ScenarioSetup:
    """Initial simulation setup.

    positions: NumPy array with shape ``(N, 2)``.
    velocities: NumPy array with shape ``(N, 2)``.
    targets: NumPy array with shape ``(N, 2)``.
    waypoints: Optional global waypoint sequence with shape ``(M, 2)``.
    waypoint_indices: Per-agent waypoint cursor with shape ``(N,)``.
    path_curve: Optional continuous path model usable by waypoint and VT guidance.
    terminal_targets: Per-agent finish points used for stopping behavior.
    """

    positions: FloatArray
    velocities: FloatArray
    targets: FloatArray
    waypoints: FloatArray | None = None
    waypoint_indices: NDArray[np.int_] | None = None
    path_curve: "ParametricSpline2D" | None = None
    path_radius: float | None = None
    terminal_targets: FloatArray | None = None
    stop_radius: float | None = None
    corridor_geometry: CorridorGeometry | None = None
    velocity_field: VelocityField | None = None


@dataclass
class SimulationResult:
    times: FloatArray
    positions: FloatArray
    velocities: FloatArray
