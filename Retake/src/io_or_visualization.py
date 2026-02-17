from __future__ import annotations

import csv
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from .path_extraction import load_map_image
from .types import FloatArray, ScenarioSetup, SimulationResult


def export_trajectories_csv(result: SimulationResult, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", "robot_id", "x", "y", "vx", "vy"])
        for k, t in enumerate(result.times):
            for rid, (pos, vel) in enumerate(zip(result.positions[k], result.velocities[k])):
                writer.writerow([f"{t:.4f}", rid, f"{pos[0]:.6f}", f"{pos[1]:.6f}", f"{vel[0]:.6f}", f"{vel[1]:.6f}"])

    return output


def _world_extent(origin_world: FloatArray, meters_per_pixel: FloatArray, image_shape: tuple[int, int]) -> tuple[float, float, float, float]:
    h, w = image_shape
    x0 = float(origin_world[0])
    y0 = float(origin_world[1])
    x1 = x0 + float((w - 1) * meters_per_pixel[0])
    y1 = y0 + float((h - 1) * meters_per_pixel[1])
    return x0, x1, y0, y1


def _plot_path_with_width(ax: plt.Axes, setup: ScenarioSetup) -> None:
    curve = setup.path_curve
    if curve is None or curve.samples.shape[0] == 0:
        return

    pts = curve.samples
    ax.plot(pts[:, 0], pts[:, 1], color="cyan", lw=2.0, label="path centerline")

    if setup.path_radius is None or setup.path_radius <= 0.0:
        return

    radius = float(setup.path_radius)
    tangents = np.gradient(pts, axis=0)
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms = np.where(norms <= 1e-9, 1.0, norms)
    tangents = tangents / norms
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))

    upper = pts + radius * normals
    lower = pts - radius * normals
    tube = np.vstack((upper, lower[::-1]))
    ax.fill(tube[:, 0], tube[:, 1], color="cyan", alpha=0.15, linewidth=0, label="path width")


def render_simulation_video(
    result: SimulationResult,
    setup: ScenarioSetup,
    output_path: str | Path,
    map_path: str | Path | None = None,
    origin_world: FloatArray | None = None,
    meters_per_pixel: FloatArray | None = None,
    fps: int = 20,
    show_popup: bool = False,
) -> Path | None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    map_extent: tuple[float, float, float, float] | None = None
    if map_path is not None and origin_world is not None and meters_per_pixel is not None and Path(map_path).exists():
        image = load_map_image(map_path)
        extent = _world_extent(np.asarray(origin_world), np.asarray(meters_per_pixel), image.shape)
        map_extent = extent
        ax.imshow(image, cmap="gray", origin="lower", extent=extent, alpha=0.9)

    _plot_path_with_width(ax, setup)

    n_agents = result.positions.shape[1]
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(10, n_agents)))
    scat = ax.scatter(result.positions[0, :, 0], result.positions[0, :, 1], s=35, c=colors[:n_agents], zorder=5)
    trails = [ax.plot([], [], color=colors[i], lw=1.25, alpha=0.9)[0] for i in range(n_agents)]

    all_pos = result.positions.reshape(-1, 2)
    x_min, y_min = np.min(all_pos, axis=0) - 0.5
    x_max, y_max = np.max(all_pos, axis=0) + 0.5
    if map_extent is not None:
        x0, x1, y0, y1 = map_extent
        x_min = min(x_min, x0)
        x_max = max(x_max, x1)
        y_min = min(y_min, y0)
        y_max = max(y_max, y1)
    ax.set_xlim(float(x_min), float(x_max))
    ax.set_ylim(float(y_min), float(y_max))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Simulation: agents + path + path width")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    def _update(frame_idx: int):
        frame_pos = result.positions[frame_idx]
        scat.set_offsets(frame_pos)
        for agent_idx, trail in enumerate(trails):
            trail.set_data(result.positions[: frame_idx + 1, agent_idx, 0], result.positions[: frame_idx + 1, agent_idx, 1])
        return [scat, *trails]

    ani = animation.FuncAnimation(fig, _update, frames=result.positions.shape[0], interval=1000.0 / max(1, fps), blit=False)

    saved_path: Path | None = None
    try:
        writer = animation.PillowWriter(fps=max(1, fps))
        ani.save(output, writer=writer)
        saved_path = output
    except Exception as exc:
        warnings.warn(f"Video export failed ({exc}).", RuntimeWarning, stacklevel=2)

    if show_popup:
        try:
            plt.show()
        except Exception as exc:
            warnings.warn(f"Popup visualization unavailable ({exc}).", RuntimeWarning, stacklevel=2)

    plt.close(fig)
    return saved_path


def maybe_visualize(result: SimulationResult, setup: ScenarioSetup, config: dict) -> Path | None:
    out_cfg = config.get("output", {})
    save_video = bool(out_cfg.get("save_video", True))
    show_popup = bool(out_cfg.get("show_popup", True))
    if not save_video and not show_popup:
        return None

    path_cfg = config.get("path_extraction", {})
    output_dir = Path(out_cfg.get("output_dir", "outputs/plots"))
    video_file = output_dir / str(out_cfg.get("video_name", "simulation.gif"))
    fps = int(out_cfg.get("video_fps", 20))

    return render_simulation_video(
        result=result,
        setup=setup,
        output_path=video_file,
        map_path=path_cfg.get("map_path"),
        origin_world=np.array(path_cfg.get("world_origin", [0.0, 0.0]), dtype=np.float64),
        meters_per_pixel=np.array(path_cfg.get("meters_per_pixel", [0.05, 0.05]), dtype=np.float64),
        fps=fps,
        show_popup=show_popup,
    )
