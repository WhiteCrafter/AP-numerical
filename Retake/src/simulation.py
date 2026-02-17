from __future__ import annotations

import numpy as np

from .dynamics import accelerations, boundary_forces, repulsion_forces, saturate_velocity
from .types import CorridorGeometry, FloatArray, ScenarioSetup, SimulationParams, SimulationResult, VelocityField

State = tuple[FloatArray, FloatArray]


def _desired_velocity_from_path(
    x: FloatArray,
    targets: FloatArray,
    setup: ScenarioSetup,
    params: SimulationParams,
) -> FloatArray | None:
    curve = setup.path_curve
    if curve is None:
        return None

    s_proj, centers = curve.project(x)
    desired = np.zeros_like(x)
    v_tangent = max(0.1, params.v_max)
    k_normal = float(params.k_p)
    for i in range(x.shape[0]):
        tangent = curve.tangent(float(s_proj[i]))
        to_target = targets[i] - x[i]
        if float(to_target @ tangent) < 0.0:
            tangent = -tangent
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
        err = x[i] - centers[i]
        e_n = float(err @ normal)
        radial_term = -k_normal * e_n * normal
        if setup.path_radius is not None and abs(e_n) > setup.path_radius:
            radial_term += -k_normal * (abs(e_n) - setup.path_radius) * np.sign(e_n) * normal
        desired[i] = v_tangent * tangent + radial_term
    return saturate_velocity(desired, params.v_max)


def _rhs(
    state: State,
    targets: FloatArray,
    params: SimulationParams,
    corridor_geometry: CorridorGeometry | None,
    velocity_field: VelocityField | None,
    time: float,
    setup: ScenarioSetup,
    arrived: np.ndarray | None = None,
) -> State:
    x, v = state
    flow = np.zeros_like(x) if velocity_field is None else np.asarray(velocity_field(x, time), dtype=np.float64)
    if arrived is not None and np.any(arrived):
        flow = flow.copy()
        flow[arrived] = 0.0
    x_dot = saturate_velocity(v + flow, params.v_max)

    desired_v = _desired_velocity_from_path(x, targets, setup, params)
    if desired_v is None:
        v_dot = accelerations(x, v, targets, params, corridor_geometry)
    else:
        reps = repulsion_forces(x, params.k_rep, params.r_safe)
        walls = boundary_forces(x, corridor_geometry, params.k_wall, params.wall_margin)
        track_v = params.k_p * (desired_v - v)
        damp = -params.k_d * v
        v_dot = (track_v + reps + walls + damp) / params.mass
    if arrived is not None and np.any(arrived):
        x_dot = x_dot.copy()
        v_dot = v_dot.copy()
        x_dot[arrived] = 0.0
        v_dot[arrived] = 0.0
    return x_dot, v_dot


def _add_state(state: State, delta: State, scale: float) -> State:
    x, v = state
    dx, dv = delta
    return x + dx * scale, v + dv * scale


def _rk4_step(
    state: State,
    targets: FloatArray,
    dt: float,
    params: SimulationParams,
    corridor_geometry: CorridorGeometry | None,
    velocity_field: VelocityField | None,
    time: float,
    setup: ScenarioSetup,
    arrived: np.ndarray | None = None,
) -> State:
    k1 = _rhs(state, targets, params, corridor_geometry, velocity_field, time, setup, arrived)
    k2 = _rhs(_add_state(state, k1, dt / 2.0), targets, params, corridor_geometry, velocity_field, time + dt / 2.0, setup, arrived)
    k3 = _rhs(_add_state(state, k2, dt / 2.0), targets, params, corridor_geometry, velocity_field, time + dt / 2.0, setup, arrived)
    k4 = _rhs(_add_state(state, k3, dt), targets, params, corridor_geometry, velocity_field, time + dt, setup, arrived)

    x, v = state
    x_next = x + (dt / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
    v_next = v + (dt / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])

    return x_next, v_next



def _arrival_mask(x: FloatArray, terminal_targets: FloatArray | None, stop_radius: float | None) -> np.ndarray:
    if terminal_targets is None or stop_radius is None or stop_radius <= 0.0:
        return np.zeros(x.shape[0], dtype=bool)
    d = np.linalg.norm(x - terminal_targets, axis=1)
    return d <= float(stop_radius)

def _advance_waypoint_targets(
    x: FloatArray,
    targets: FloatArray,
    waypoints: FloatArray | None,
    waypoint_indices: np.ndarray | None,
    epsilon: float,
    arrived: np.ndarray | None = None,
) -> None:
    if waypoints is None or waypoint_indices is None or waypoints.shape[0] == 0:
        return

    last_idx = waypoints.shape[0] - 1
    for i in range(x.shape[0]):
        idx = int(np.clip(waypoint_indices[i], 0, last_idx))
        while idx < last_idx and float(np.linalg.norm(x[i] - waypoints[idx])) <= epsilon:
            idx += 1
        waypoint_indices[i] = idx
        targets[i] = waypoints[idx]


def run_simulation(setup: ScenarioSetup, params: SimulationParams) -> SimulationResult:
    steps = int(params.t_end / params.dt) + 1
    times = np.arange(steps, dtype=np.float64) * params.dt

    x = np.array(setup.positions, dtype=np.float64, copy=True)
    v = np.array(setup.velocities, dtype=np.float64, copy=True)
    targets = np.array(setup.targets, dtype=np.float64, copy=True)
    waypoints = None if setup.waypoints is None else np.array(setup.waypoints, dtype=np.float64, copy=False)
    waypoint_indices = None
    if setup.waypoint_indices is not None:
        waypoint_indices = np.array(setup.waypoint_indices, dtype=np.int64, copy=True)

    waypoint_epsilon = max(1e-6, params.v_max * params.dt)
    terminal_targets = None if setup.terminal_targets is None else np.array(setup.terminal_targets, dtype=np.float64, copy=False)

    arrived = _arrival_mask(x, terminal_targets, setup.stop_radius)
    if np.any(arrived):
        v[arrived] = 0.0

    _advance_waypoint_targets(x, targets, waypoints, waypoint_indices, waypoint_epsilon, arrived)

    positions = [x.copy()]
    velocities = [v.copy()]

    for _ in range(1, steps):
        arrived = _arrival_mask(x, terminal_targets, setup.stop_radius)
        if np.any(arrived):
            v[arrived] = 0.0
            if terminal_targets is not None:
                targets[arrived] = terminal_targets[arrived]

        _advance_waypoint_targets(x, targets, waypoints, waypoint_indices, waypoint_epsilon, arrived)
        x, v = _rk4_step((x, v), targets, params.dt, params, setup.corridor_geometry, setup.velocity_field, times[_ - 1], setup, arrived)

        arrived_next = _arrival_mask(x, terminal_targets, setup.stop_radius)
        if np.any(arrived_next):
            v[arrived_next] = 0.0

        positions.append(x.copy())
        velocities.append(v.copy())

    return SimulationResult(times=times, positions=np.stack(positions), velocities=np.stack(velocities))
