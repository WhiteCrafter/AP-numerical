from __future__ import annotations

import numpy as np

from .types import CorridorGeometry, FloatArray, SimulationParams


def saturate_velocity(v: FloatArray, v_max: float) -> FloatArray:
    """Clamp each agent velocity to ``v_max`` while preserving direction.

    Formula mapping:
    1. ``s_i = ||v_i||`` is implemented by ``np.linalg.norm(..., axis=1, keepdims=True)``.
    2. ``\alpha_i = v_max / s_i`` when ``s_i > v_max`` else ``1`` is implemented with ``np.divide(..., where=...)``.
    3. ``v_i^{sat} = \alpha_i v_i`` is implemented by elementwise multiplication ``v * scale``.
    """
    speeds = np.linalg.norm(v, axis=1, keepdims=True)
    scale = np.ones_like(speeds)
    np.divide(v_max, speeds, out=scale, where=speeds > v_max)
    return v * scale


def repulsion_forces(x: FloatArray, k_rep: float, r_safe: float) -> FloatArray:
    """Compute pairwise short-range repulsion using explicit ``i``/``j`` loops.

    Formula mapping (for each ordered pair ``i != j``):
    1. Relative displacement ``d_ij = x_i - x_j`` maps to ``diff = x[i] - x[j]``.
    2. Distance ``r_ij = ||d_ij||`` maps to ``dist = np.linalg.norm(diff)``.
    3. Pair force
       ``f_ij = (k_rep / (r_ij^3 + eps)) * d_ij`` when ``0 < r_ij < r_safe``
       (otherwise ``0``) maps to the ``if`` block and ``forces[i] += ...`` update.
    4. Net repulsion ``f_i^{rep} = sum_{j!=i} f_ij`` is accumulated in ``forces[i]``.
    """
    n_agents = x.shape[0]
    eps = 1e-9
    forces = np.zeros_like(x)

    for i in range(n_agents):
        for j in range(n_agents):
            if i == j:
                continue

            diff = x[i] - x[j]
            dist = np.linalg.norm(diff)
            if eps < dist < r_safe:
                forces[i] += (k_rep / (dist**3 + eps)) * diff

    return forces


def boundary_forces(
    x: FloatArray,
    corridor_geometry: CorridorGeometry | None,
    k_wall: float,
    wall_margin: float,
) -> FloatArray:
    """Compute corridor wall repulsion near left/right boundaries.

    The signed lateral coordinate is ``s = (x - origin) Â· normal``.
    The corridor walls are at ``s = -half_width`` and ``s = +half_width``.
    For a margin ``m`` from each wall, repulsive force is applied along ``normal``:

    * Left wall distance: ``d_left = s + half_width`` and push ``+normal``.
    * Right wall distance: ``d_right = half_width - s`` and push ``-normal``.

    Force magnitude is linear in margin penetration,
    ``k_wall * (wall_margin - d)``, and zero when ``d >= wall_margin``.
    """
    forces = np.zeros_like(x)
    if corridor_geometry is None or k_wall <= 0.0 or wall_margin <= 0.0:
        return forces

    rel = x - corridor_geometry.origin
    lateral = rel @ corridor_geometry.normal

    d_left = lateral + corridor_geometry.half_width
    d_right = corridor_geometry.half_width - lateral

    left_strength = np.maximum(0.0, wall_margin - d_left)
    right_strength = np.maximum(0.0, wall_margin - d_right)

    forces += (k_wall * left_strength)[:, np.newaxis] * corridor_geometry.normal
    forces -= (k_wall * right_strength)[:, np.newaxis] * corridor_geometry.normal
    return forces


def accelerations(
    x: FloatArray,
    v: FloatArray,
    targets: FloatArray,
    params: SimulationParams,
    corridor_geometry: CorridorGeometry | None = None,
) -> FloatArray:
    """Compute agent accelerations from tracking, damping, repulsion and wall terms."""
    reps = repulsion_forces(x, params.k_rep, params.r_safe)
    walls = boundary_forces(x, corridor_geometry, params.k_wall, params.wall_margin)
    track = params.k_p * (targets - x)
    damp = -params.k_d * v
    total = track + damp + reps + walls
    return total / params.mass


