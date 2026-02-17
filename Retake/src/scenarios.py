from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import numpy as np

from .path_extraction import build_spline_from_waypoints, extract_waypoints
from .types import CorridorGeometry, FloatArray, ScenarioSetup, VelocityField

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - tested through fallback behavior
    cv2 = None


# -------------------------------
# setup_single_scenario
# -------------------------------
def _build_straight_line_targets(start: FloatArray, goal: FloatArray, n: int, config: dict) -> tuple[FloatArray, FloatArray | None, np.ndarray | None]:
    """Build map-driven targets and optional waypoint-following metadata for A->B."""
    path_cfg = config.get("path_extraction")
    if path_cfg is None:
        return np.repeat(goal[np.newaxis, :], n, axis=0), None, None

    map_path = path_cfg.get("map_path")
    if not map_path or not Path(map_path).exists():
        return np.repeat(goal[np.newaxis, :], n, axis=0), None, None

    point_a = np.array(path_cfg.get("point_a", start.tolist()), dtype=np.float64)
    point_b = np.array(path_cfg.get("point_b", goal.tolist()), dtype=np.float64)
    origin_world = np.array(path_cfg["world_origin"], dtype=np.float64)
    meters_per_pixel = np.array(path_cfg["meters_per_pixel"], dtype=np.float64)

    try:
        waypoints = extract_waypoints(
            map_path=map_path,
            point_a_world=point_a,
            point_b_world=point_b,
            origin_world=origin_world,
            meters_per_pixel=meters_per_pixel,
            traversable_threshold=float(path_cfg.get("traversable_threshold", 0.5)),
            waypoint_stride=int(path_cfg.get("waypoint_stride", 4)),
            debug_output_dir=path_cfg.get("debug_output_dir"),
            debug_prefix=str(path_cfg.get("debug_prefix", "path_debug")),
            map_resize_factor=float(path_cfg.get("map_resize_factor", 1.0)),
        )
    except ValueError as exc:
        warnings.warn(
            f"Path extraction failed ({exc}); falling back to direct goal target.",
            RuntimeWarning,
            stacklevel=2,
        )
        return np.repeat(goal[np.newaxis, :], n, axis=0), None, None

    waypoints = np.asarray(waypoints, dtype=np.float64)
    if waypoints.shape[0] == 0:
        return np.repeat(goal[np.newaxis, :], n, axis=0), None, None

    waypoints[0] = point_a
    waypoints[-1] = point_b
    waypoint_indices = np.zeros(n, dtype=np.int64)
    first_waypoint = waypoints[0]
    return np.repeat(first_waypoint[np.newaxis, :], n, axis=0), waypoints, waypoint_indices


def setup_single_scenario(config: dict) -> ScenarioSetup:
    single = config["single"]
    start = np.array(single["start"], dtype=np.float64)
    goal = np.array(single["goal"], dtype=np.float64)

    positions = start[np.newaxis, :]
    velocities = np.zeros_like(positions)
    targets, waypoints, waypoint_indices = _build_straight_line_targets(start, goal, n=1, config=config)
    path_curve = None if waypoints is None else build_spline_from_waypoints(waypoints)
    path_radius = float(config.get("single", {}).get("path_radius", 0.25)) if path_curve is not None else None
    return ScenarioSetup(
        positions=positions,
        velocities=velocities,
        targets=targets,
        waypoints=waypoints,
        waypoint_indices=waypoint_indices,
        path_curve=path_curve,
        path_radius=path_radius,
    )


# -------------------------------
# setup_swarm_scenario
# -------------------------------
def setup_swarm_scenario(config: dict) -> ScenarioSetup:
    """Create initial robot states for two groups moving opposite directions."""
    swarm_cfg = config["swarm"]
    sim_cfg = config.get("simulation", {})
    n_a = int(swarm_cfg["n_a"])
    n_b = int(swarm_cfg["n_b"])
    r_safe = float(sim_cfg.get("r_safe", 0.25))

    path_cfg = config.get("path_extraction", {})
    start_a_cfg = swarm_cfg.get("start_a", path_cfg.get("point_a"))
    start_b_cfg = swarm_cfg.get("start_b", path_cfg.get("point_b"))
    if start_a_cfg is None or start_b_cfg is None:
        raise ValueError(
            "Swarm requires start/goal points. Provide swarm.start_a/start_b or select Point A/B in the CLI prompt."
        )

    start_a = np.array(start_a_cfg, dtype=np.float64)
    start_b = np.array(start_b_cfg, dtype=np.float64)
    centerline = start_b - start_a
    corridor_length = float(np.linalg.norm(centerline))
    if corridor_length <= 1e-9:
        raise ValueError("Swarm start_a and start_b must not be identical.")

    lane_cfg = swarm_cfg.get("lane_center_offsets", [-0.2, 0.2])
    if len(lane_cfg) < 2:
        raise ValueError("swarm.lane_center_offsets must contain at least two values.")
    lane_a_offset = float(lane_cfg[0])
    lane_b_offset = float(lane_cfg[1])
    corridor_width = float(
        swarm_cfg.get("corridor_width", 2.0 * max(abs(lane_a_offset), abs(lane_b_offset), r_safe))
    )
    half_width = 0.5 * corridor_width
    path_radius = float(swarm_cfg.get("path_radius", half_width))
    preferred_spacing = float(swarm_cfg.get("preferred_spawn_spacing", r_safe * 1.2))
    spawn_spacing = max(preferred_spacing, r_safe * 1.05)
    seed = swarm_cfg.get("spawn_seed")
    rng = np.random.default_rng(None if seed is None else int(seed))
    jitter_scale = float(swarm_cfg.get("spawn_jitter", 0.01))

    tangent = centerline / corridor_length
    normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)

    def clip_to_corridor(points: np.ndarray) -> np.ndarray:
        rel = points - start_a
        longitudinal = np.clip(rel @ tangent, 0.0, corridor_length)
        lateral = np.clip(rel @ normal, -half_width, half_width)
        return start_a + np.outer(longitudinal, tangent) + np.outer(lateral, normal)

    def make_group(anchor: np.ndarray, n: int, lane_offset: float, direction_sign: float) -> np.ndarray:
        if n <= 0:
            return np.empty((0, 2), dtype=np.float64)
        lane_offset = float(np.clip(lane_offset, -half_width, half_width))
        distances = np.arange(n, dtype=np.float64) * spawn_spacing
        distances += rng.uniform(-jitter_scale, jitter_scale, size=n)
        distances = np.clip(distances, 0.0, corridor_length)
        distances.sort()
        distances = np.maximum.accumulate(distances)

        if direction_sign > 0.0:
            centers = anchor + np.outer(distances, tangent)
        else:
            centers = anchor - np.outer(distances, tangent)

        lateral_jitter = rng.uniform(-jitter_scale, jitter_scale, size=n)
        points = centers + np.outer(lane_offset + lateral_jitter, normal)
        points = clip_to_corridor(points)

        for i in range(1, n):
            delta = points[i] - points[i - 1]
            dist = float(np.linalg.norm(delta))
            if dist < r_safe:
                needed = (r_safe - dist) + 1e-3
                points[i] += direction_sign * needed * tangent

        return clip_to_corridor(points)

    def enforce_pairwise_distance(points: np.ndarray) -> np.ndarray:
        resolved = points.copy()
        max_iters = 200
        eps = 1e-9
        for _ in range(max_iters):
            moved = False
            for i in range(len(resolved)):
                for j in range(i + 1, len(resolved)):
                    d = resolved[i] - resolved[j]
                    dist = float(np.linalg.norm(d))
                    if dist + eps >= r_safe:
                        continue
                    moved = True
                    if dist <= eps:
                        unit = tangent if (i + j) % 2 == 0 else normal
                    else:
                        unit = d / dist
                    correction = 0.5 * (r_safe - dist + 1e-3)
                    resolved[i] += correction * unit
                    resolved[j] -= correction * unit
            resolved = clip_to_corridor(resolved)
            if not moved:
                break

        diffs = resolved[:, np.newaxis, :] - resolved[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        np.fill_diagonal(dists, np.inf)
        min_dist = float(np.min(dists)) if resolved.shape[0] > 1 else np.inf
        if min_dist + 1e-9 < r_safe:
            raise ValueError(
                "Could not initialize swarm with pairwise distances above r_safe. "
                "Increase corridor length/width or reduce robot count."
            )
        return resolved

    pos_a = make_group(start_a, n_a, lane_a_offset, direction_sign=1.0)
    pos_b = make_group(start_b, n_b, lane_b_offset, direction_sign=-1.0)

    positions = enforce_pairwise_distance(np.vstack((pos_a, pos_b)))
    pos_a = positions[:n_a]
    pos_b = positions[n_a:]
    velocities = np.zeros_like(positions)

    def make_targets(group_positions: np.ndarray, direction_sign: float) -> np.ndarray:
        rel = group_positions - start_a
        lateral = rel @ normal
        if direction_sign > 0.0:
            target_longitudinal = np.full(group_positions.shape[0], corridor_length)
        else:
            target_longitudinal = np.zeros(group_positions.shape[0])
        target_points = (
            start_a
            + np.outer(target_longitudinal, tangent)
            + np.outer(np.clip(lateral, -half_width, half_width), normal)
        )
        return clip_to_corridor(target_points)

    targets = np.vstack((make_targets(pos_a, 1.0), make_targets(pos_b, -1.0)))
    corridor_geometry = CorridorGeometry(
        origin=start_a,
        tangent=tangent,
        normal=normal,
        length=corridor_length,
        half_width=half_width,
    )

    swarm_path = build_spline_from_waypoints(np.vstack((start_a, start_b)))
    return ScenarioSetup(
        positions=positions,
        velocities=velocities,
        targets=targets,
        path_curve=swarm_path,
        path_radius=path_radius,
        corridor_geometry=corridor_geometry,
    )


# -------------------------------
# setup_pedestrian_scenario
# -------------------------------
def _video_frames_to_velocity_grids(
    frames_gray: list[np.ndarray], fps: float, meters_per_pixel: np.ndarray, invert_y_axis: bool
) -> np.ndarray:
    if len(frames_gray) < 2:
        raise ValueError("At least two frames are required for optical flow estimation.")

    pixel_flows: list[np.ndarray] = []
    for idx in range(len(frames_gray) - 1):
        prev = frames_gray[idx]
        nxt = frames_gray[idx + 1]
        flow = cv2.calcOpticalFlowFarneback(
            prev,
            nxt,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        pixel_flows.append(flow.astype(np.float64, copy=False))

    flow_grid = np.stack(pixel_flows, axis=0)
    sign_y = -1.0 if invert_y_axis else 1.0
    scale = np.array([meters_per_pixel[0] * fps, sign_y * meters_per_pixel[1] * fps], dtype=np.float64)
    return flow_grid * scale


def _build_interpolated_velocity_field(
    velocity_grids_world: np.ndarray,
    fps: float,
    world_origin: np.ndarray,
    meters_per_pixel: np.ndarray,
    invert_y_axis: bool,
    interpolation: str = "bilinear",
) -> VelocityField:
    if velocity_grids_world.ndim != 4 or velocity_grids_world.shape[-1] != 2:
        raise ValueError("velocity_grids_world must have shape (T, H, W, 2).")
    if fps <= 0.0:
        raise ValueError("fps must be positive.")

    n_t, height, width, _ = velocity_grids_world.shape
    dt_frame = 1.0 / fps
    mode = interpolation.lower()
    if mode not in {"nearest", "bilinear"}:
        raise ValueError("interpolation must be either 'nearest' or 'bilinear'.")

    def _world_to_pixel(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        px = (points[:, 0] - world_origin[0]) / meters_per_pixel[0]
        if invert_y_axis:
            py = (world_origin[1] - points[:, 1]) / meters_per_pixel[1]
        else:
            py = (points[:, 1] - world_origin[1]) / meters_per_pixel[1]
        return px, py

    def _sample_spatial(grid: np.ndarray, points: np.ndarray) -> np.ndarray:
        px, py = _world_to_pixel(points)
        if mode == "nearest":
            x_idx = np.clip(np.rint(px).astype(np.int64), 0, width - 1)
            y_idx = np.clip(np.rint(py).astype(np.int64), 0, height - 1)
            return grid[y_idx, x_idx]

        x0 = np.floor(px).astype(np.int64)
        y0 = np.floor(py).astype(np.int64)
        x1 = x0 + 1
        y1 = y0 + 1

        x0c = np.clip(x0, 0, width - 1)
        x1c = np.clip(x1, 0, width - 1)
        y0c = np.clip(y0, 0, height - 1)
        y1c = np.clip(y1, 0, height - 1)

        wx = np.clip(px - x0, 0.0, 1.0)
        wy = np.clip(py - y0, 0.0, 1.0)

        v00 = grid[y0c, x0c]
        v10 = grid[y0c, x1c]
        v01 = grid[y1c, x0c]
        v11 = grid[y1c, x1c]

        top = (1.0 - wx)[:, None] * v00 + wx[:, None] * v10
        bottom = (1.0 - wx)[:, None] * v01 + wx[:, None] * v11
        return (1.0 - wy)[:, None] * top + wy[:, None] * bottom

    def field(position: FloatArray, time: float) -> FloatArray:
        points = np.asarray(position, dtype=np.float64)
        squeeze = points.ndim == 1
        if squeeze:
            points = points[np.newaxis, :]

        t_idx = np.clip(time / dt_frame, 0.0, max(0.0, n_t - 1.0))
        t0 = int(np.floor(t_idx))
        t1 = min(t0 + 1, n_t - 1)
        wt = float(t_idx - t0)

        flow_0 = _sample_spatial(velocity_grids_world[t0], points)
        if t1 == t0:
            flow = flow_0
        else:
            flow_1 = _sample_spatial(velocity_grids_world[t1], points)
            flow = (1.0 - wt) * flow_0 + wt * flow_1

        return flow[0] if squeeze else flow

    return field


def estimate_velocity_field_placeholder(video_cfg: dict[str, Any] | None = None) -> VelocityField:
    """Estimate dense world-frame velocity field from video or fallback to placeholder mode."""
    cfg = dict(video_cfg or {})
    placeholder_velocity = np.array(cfg.get("placeholder_velocity", [0.4, 0.0]), dtype=np.float64)

    def placeholder_field(position: FloatArray, _: float) -> FloatArray:
        points = np.asarray(position, dtype=np.float64)
        if points.ndim == 1:
            return placeholder_velocity.copy()
        return np.repeat(placeholder_velocity[np.newaxis, :], points.shape[0], axis=0)

    mode = str(cfg.get("mode", "placeholder")).lower()
    if mode == "placeholder":
        return placeholder_field

    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is not available; cannot estimate optical flow from video.")

    video_path = cfg.get("video_path")
    if not video_path:
        raise ValueError("pedestrian.flow_estimation.video_path must be set when mode='video'.")
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0.0:
        fps = float(cfg.get("fps", 30.0))

    frame_start = int(cfg.get("frame_start", 0))
    frame_end_cfg = cfg.get("frame_end")
    frame_end = None if frame_end_cfg is None else int(frame_end_cfg)

    idx = 0
    frames_gray: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if idx >= frame_start and (frame_end is None or idx < frame_end):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames_gray.append(gray)
            idx += 1
            if frame_end is not None and idx >= frame_end:
                break
    finally:
        capture.release()

    meters_per_pixel = np.array(cfg.get("meters_per_pixel", [0.05, 0.05]), dtype=np.float64)
    world_origin = np.array(cfg.get("world_origin", [0.0, 0.0]), dtype=np.float64)
    invert_y_axis = bool(cfg.get("invert_y_axis", True))
    interpolation = str(cfg.get("interpolation", "bilinear"))

    velocity_grids_world = _video_frames_to_velocity_grids(frames_gray, fps, meters_per_pixel, invert_y_axis)
    return _build_interpolated_velocity_field(
        velocity_grids_world=velocity_grids_world,
        fps=fps,
        world_origin=world_origin,
        meters_per_pixel=meters_per_pixel,
        invert_y_axis=invert_y_axis,
        interpolation=interpolation,
    )


def setup_pedestrian_scenario(config: dict) -> ScenarioSetup:
    ped_cfg = config["pedestrian"]
    start = np.array(ped_cfg["start"], dtype=np.float64)
    goal = np.array(ped_cfg["goal"], dtype=np.float64)

    flow_cfg = dict(ped_cfg.get("flow_estimation", {}))
    on_error = str(flow_cfg.get("on_error", "raise")).lower()
    try:
        velocity_field = estimate_velocity_field_placeholder(flow_cfg)
    except Exception:
        if on_error != "placeholder":
            raise
        fallback_cfg = dict(flow_cfg)
        fallback_cfg["mode"] = "placeholder"
        velocity_field = estimate_velocity_field_placeholder(fallback_cfg)

    positions = start[np.newaxis, :]
    velocities = np.zeros_like(positions)
    targets = goal[np.newaxis, :]
    return ScenarioSetup(positions=positions, velocities=velocities, targets=targets, velocity_field=velocity_field)


def build_scenario(scenario: str, config: dict) -> ScenarioSetup:
    if scenario == "single":
        return setup_single_scenario(config)
    if scenario == "swarm":
        return setup_swarm_scenario(config)
    if scenario == "pedestrian":
        return setup_pedestrian_scenario(config)
    raise ValueError(f"Unknown scenario: {scenario}")
