
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline, splprep, splev


def sample_segment(p0: np.ndarray, p1: np.ndarray, n: int, include_endpoint: bool) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n, endpoint=include_endpoint)
    return (1.0 - t)[:, None] * p0 + t[:, None] * p1


def sample_polyline(knots: np.ndarray, n_per_segment: int) -> np.ndarray:
    points = []
    for i in range(len(knots) - 1):
        segment = sample_segment(knots[i], knots[i + 1], n_per_segment, include_endpoint=True)
        if i > 0:
            segment = segment[1:]
        points.append(segment)
    return np.vstack(points)


def build_shapes() -> dict[str, np.ndarray]:
    shapes = {}

    # Arc segment for a simple "C" outline.
    theta_c = np.linspace(np.deg2rad(45), np.deg2rad(315), 180)
    c_points = np.column_stack((np.cos(theta_c), np.sin(theta_c)))
    shapes["C"] = c_points

    # Loop plus a curved tail to suggest a "6".
    center = np.array([0.0, -0.4])
    theta_loop = np.linspace(0.0, 2.0 * np.pi, 260, endpoint=False)
    loop = np.column_stack((np.cos(theta_loop), 0.9 * np.sin(theta_loop))) + center
    target = center + np.array([0.7, 0.5])
    tail_start = loop[np.argmin(np.linalg.norm(loop - target, axis=1))]
    tail_knots = np.array([tail_start, [0.3, 1.1], [-0.4, 1.4]])
    tail = sample_polyline(tail_knots, n_per_segment=25)
    six_points = np.vstack((loop, tail[1:]))
    six_points[:, 0] *= -1.0
    shapes["6"] = six_points

    # Polyline for a "Z".
    z_knots = np.array([[0.0, 1.0], [1.2, 1.0], [0.0, -0.2], [1.2, -0.2]])
    z_points = sample_polyline(z_knots, n_per_segment=80)
    shapes["Z"] = z_points

    return shapes


def arc_length_param(points: np.ndarray) -> np.ndarray:
    diffs = np.diff(points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    s = np.concatenate(([0.0], np.cumsum(seg_lengths)))
    if s[-1] == 0.0:
        return np.zeros_like(s)
    return s / s[-1]


def select_nodes_uniform(points: np.ndarray, n: int) -> np.ndarray:
    if n >= len(points):
        return np.arange(len(points))
    idx = np.linspace(0, len(points) - 1, n).round().astype(int)
    return np.unique(idx)


def select_nodes_corner(points: np.ndarray, n: int) -> np.ndarray:
    if n >= len(points):
        return np.arange(len(points))

    interior = np.arange(1, len(points) - 1)
    # Turning angle (0 for straight, larger for corners) acts as a corner strength score.
    v1 = points[1:-1] - points[:-2]
    v2 = points[2:] - points[1:-1]
    v1_norm = np.linalg.norm(v1, axis=1)
    v2_norm = np.linalg.norm(v2, axis=1)
    denom = np.clip(v1_norm * v2_norm, 1e-12, None)
    cos_angle = np.sum(v1 * v2, axis=1) / denom
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    turning = angle

    k = max(0, n - 2)
    top = interior[np.argsort(turning)[::-1][:k]]
    idx = np.unique(np.concatenate(([0, len(points) - 1], top)))

    if idx.size < n:
        uniform = np.linspace(0, len(points) - 1, n).round().astype(int)
        idx = np.unique(np.concatenate((idx, uniform)))

    if idx.size > n:
        priorities = np.zeros(len(points))
        priorities[1:-1] = turning
        priorities[0] = np.inf
        priorities[-1] = np.inf
        idx = idx[np.argsort(priorities[idx])[::-1][:n]]

    return np.sort(idx)


def fit_cubic_spline(points: np.ndarray, t_all: np.ndarray, nodes_idx: np.ndarray) -> np.ndarray:
    t_nodes = t_all[nodes_idx]
    x_nodes = points[nodes_idx, 0]
    y_nodes = points[nodes_idx, 1]
    cs_x = CubicSpline(t_nodes, x_nodes, bc_type="natural")
    cs_y = CubicSpline(t_nodes, y_nodes, bc_type="natural")
    t_dense = np.linspace(t_nodes[0], t_nodes[-1], 400)
    return np.column_stack((cs_x(t_dense), cs_y(t_dense)))


def fit_bspline(points: np.ndarray, t_all: np.ndarray, nodes_idx: np.ndarray) -> np.ndarray:
    t_nodes = t_all[nodes_idx]
    x_nodes = points[nodes_idx, 0]
    y_nodes = points[nodes_idx, 1]
    k = min(3, len(nodes_idx) - 1)
    tck, _ = splprep([x_nodes, y_nodes], u=t_nodes, s=0, k=k)
    u_dense = np.linspace(t_nodes[0], t_nodes[-1], 400)
    x_dense, y_dense = splev(u_dense, tck)
    return np.column_stack((x_dense, y_dense))


def mean_nearest_distance(points: np.ndarray, curve: np.ndarray) -> float:
    diff = points[:, None, :] - curve[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    return float(dist.min(axis=1).mean())


def plot_shape_experiments(name: str, points: np.ndarray, output_dir: Path) -> None:
    t_all = arc_length_param(points)
    configs = [
        {
            "title": "Cubic + uniform nodes (n=12)",
            "selector": select_nodes_uniform,
            "n": 12,
            "spline": "cubic",
        },
        {
            "title": "Cubic + corner-aware nodes (n=12)",
            "selector": select_nodes_corner,
            "n": 12,
            "spline": "cubic",
        },
        {
            "title": "B-spline + uniform nodes (n=12)",
            "selector": select_nodes_uniform,
            "n": 12,
            "spline": "bspline",
        },
        {
            "title": "Cubic + uniform nodes (n=20)",
            "selector": select_nodes_uniform,
            "n": 20,
            "spline": "cubic",
        },
    ]

    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    for ax, cfg in zip(axes.ravel(), configs):
        nodes_idx = cfg["selector"](points, cfg["n"])
        if cfg["spline"] == "cubic":
            curve = fit_cubic_spline(points, t_all, nodes_idx)
        else:
            curve = fit_bspline(points, t_all, nodes_idx)
        # Mean nearest distance acts as a simple, comparable error proxy.
        error = mean_nearest_distance(points, curve)

        ax.plot(points[:, 0], points[:, 1], ".", color="#b0b0b0", markersize=3, label="original")
        ax.plot(curve[:, 0], curve[:, 1], "-", color="#1f77b4", linewidth=2, label="spline")
        ax.plot(
            points[nodes_idx, 0],
            points[nodes_idx, 1],
            "o",
            color="#d62728",
            markersize=4,
            label="nodes",
        )
        ax.set_title(f"{cfg['title']}\nmean dist = {error:.3f}")
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        if cfg is configs[0]:
            ax.legend(loc="upper right", frameon=False)

    fig.suptitle(f"Shape {name}: spline fitting experiments", fontsize=14)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    output_path = output_dir / f"{name}_spline_experiments.png"
    fig.savefig(output_path, dpi=160)


def main() -> None:
    # Store figures alongside the script for report-ready output.
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(exist_ok=True)
    shapes = build_shapes()
    for name, points in shapes.items():
        plot_shape_experiments(name, points, output_dir)
    plt.show()


if __name__ == "__main__":
    main()
