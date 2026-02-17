from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .path_curve import ParametricSpline2D, build_parametric_spline
from .types import FloatArray


def load_map_image(path: str | Path) -> FloatArray:
    """Load a map image (e.g., PNG) into [0, 1] grayscale intensity values."""
    map_path = Path(path)
    image = Image.open(map_path).convert("L")
    image_array = np.asarray(image, dtype=np.float64)
    return image_array / 255.0





def _resize_grayscale_image(image: FloatArray, resize_factor: float) -> FloatArray:
    factor = float(resize_factor)
    if abs(factor - 1.0) <= 1e-9:
        return image
    if factor <= 0.0:
        raise ValueError("resize_factor must be positive.")
    height, width = image.shape
    out_w = max(1, int(round(width * factor)))
    out_h = max(1, int(round(height * factor)))
    pil_image = Image.fromarray((np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
    resample = Image.Resampling.BILINEAR if factor < 1.0 else Image.Resampling.BICUBIC
    resized = pil_image.resize((out_w, out_h), resample=resample)
    return np.asarray(resized, dtype=np.float64) / 255.0

def world_to_pixel(point: FloatArray, origin: FloatArray, meters_per_pixel: FloatArray) -> tuple[int, int]:
    x_idx = int(round((point[0] - origin[0]) / meters_per_pixel[0]))
    y_idx = int(round((point[1] - origin[1]) / meters_per_pixel[1]))
    return y_idx, x_idx


def pixel_to_world(row: int, col: int, origin: FloatArray, meters_per_pixel: FloatArray) -> FloatArray:
    x = origin[0] + col * meters_per_pixel[0]
    y = origin[1] + row * meters_per_pixel[1]
    return np.array([x, y], dtype=np.float64)


def _shortest_path(occupancy: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    neighbors = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ]

    queue: deque[tuple[int, int]] = deque([start])
    parents: dict[tuple[int, int], tuple[int, int] | None] = {start: None}

    rows, cols = occupancy.shape
    while queue:
        cur = queue.popleft()
        if cur == goal:
            break

        for dr, dc in neighbors:
            nxt = (cur[0] + dr, cur[1] + dc)
            r, c = nxt
            if r < 0 or r >= rows or c < 0 or c >= cols:
                continue
            if not occupancy[r, c] or nxt in parents:
                continue
            parents[nxt] = cur
            queue.append(nxt)

    if goal not in parents:
        return []

    path: list[tuple[int, int]] = []
    cursor: tuple[int, int] | None = goal
    while cursor is not None:
        path.append(cursor)
        cursor = parents[cursor]
    path.reverse()
    return path


def _save_debug_artifacts(
    image: np.ndarray,
    occupancy: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    pixel_path: list[tuple[int, int]],
    output_dir: Path,
    prefix: str,
    spline_path: np.ndarray | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    grayscale = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(grayscale, mode="L").save(output_dir / f"{prefix}_grayscale.png")

    traversable_vis = np.repeat((grayscale * 0.35).astype(np.uint8)[..., np.newaxis], 3, axis=2)
    traversable_vis[occupancy] = np.array([220, 220, 220], dtype=np.uint8)
    traversable_image = Image.fromarray(traversable_vis, mode="RGB")
    draw = ImageDraw.Draw(traversable_image)

    marker_radius = 3
    start_x, start_y = start[1], start[0]
    goal_x, goal_y = goal[1], goal[0]
    draw.ellipse(
        (start_x - marker_radius, start_y - marker_radius, start_x + marker_radius, start_y + marker_radius),
        fill=(0, 255, 0),
    )
    draw.ellipse(
        (goal_x - marker_radius, goal_y - marker_radius, goal_x + marker_radius, goal_y + marker_radius),
        fill=(255, 0, 0),
    )
    traversable_image.save(output_dir / f"{prefix}_traversable_with_points.png")

    path_only = np.zeros_like(grayscale, dtype=np.uint8)
    for row, col in pixel_path:
        row_min = max(0, row - 1)
        row_max = min(path_only.shape[0], row + 2)
        col_min = max(0, col - 1)
        col_max = min(path_only.shape[1], col + 2)
        path_only[row_min:row_max, col_min:col_max] = 255
    Image.fromarray(path_only, mode="L").save(output_dir / f"{prefix}_path_only.png")

    if spline_path is not None and spline_path.size > 0:
        spline_only = np.zeros_like(grayscale, dtype=np.uint8)
        for row_f, col_f in spline_path:
            row = int(round(float(row_f)))
            col = int(round(float(col_f)))
            if row < 0 or row >= spline_only.shape[0] or col < 0 or col >= spline_only.shape[1]:
                continue
            row_min = max(0, row - 1)
            row_max = min(spline_only.shape[0], row + 2)
            col_min = max(0, col - 1)
            col_max = min(spline_only.shape[1], col + 2)
            spline_only[row_min:row_max, col_min:col_max] = 255
        Image.fromarray(spline_only, mode="L").save(output_dir / f"{prefix}_spline_only.png")


def extract_waypoints(
    map_path: str | Path,
    point_a_world: FloatArray,
    point_b_world: FloatArray,
    origin_world: FloatArray,
    meters_per_pixel: FloatArray,
    traversable_threshold: float = 0.5,
    waypoint_stride: int = 4,
    debug_output_dir: str | Path | None = None,
    debug_prefix: str = "path_debug",
    map_resize_factor: float = 1.0,
) -> FloatArray:
    """Extract ordered world-coordinate waypoints between A and B."""
    image = load_map_image(map_path)
    image = _resize_grayscale_image(image, map_resize_factor)
    occupancy = image >= traversable_threshold

    effective_mpp = np.asarray(meters_per_pixel, dtype=np.float64) / float(map_resize_factor)
    start = world_to_pixel(point_a_world, origin_world, effective_mpp)
    goal = world_to_pixel(point_b_world, origin_world, effective_mpp)

    rows, cols = occupancy.shape
    for node, name in ((start, "A"), (goal, "B")):
        if node[0] < 0 or node[0] >= rows or node[1] < 0 or node[1] >= cols:
            raise ValueError(f"Point {name} is outside map bounds: {node}")
        if not occupancy[node]:
            raise ValueError(f"Point {name} lies on a non-traversable cell: {node}")

    pixel_path = _shortest_path(occupancy, start, goal)
    if not pixel_path:
        raise ValueError("No traversable path found between points A and B")

    stride = max(1, int(waypoint_stride))
    sampled_pixels = pixel_path[::stride]
    if sampled_pixels[-1] != pixel_path[-1]:
        sampled_pixels.append(pixel_path[-1])

    # Keep the current shortest path as baseline, then recenter sampled points
    # toward corridor middle using local normal probing.
    recenter_probe_px = 40
    sampled_arr = np.asarray(sampled_pixels, dtype=np.float64)
    recentered = sampled_arr.copy()
    n_pts = recentered.shape[0]
    rows, cols = occupancy.shape

    def nearest_traversable(row_f: float, col_f: float, max_radius: int = 3) -> tuple[float, float]:
        row0 = int(np.clip(round(row_f), 0, rows - 1))
        col0 = int(np.clip(round(col_f), 0, cols - 1))
        if occupancy[row0, col0]:
            return float(row0), float(col0)

        best: tuple[int, int] | None = None
        best_d2 = float("inf")
        for rad in range(1, max_radius + 1):
            r_min = max(0, row0 - rad)
            r_max = min(rows - 1, row0 + rad)
            c_min = max(0, col0 - rad)
            c_max = min(cols - 1, col0 + rad)
            for rr in range(r_min, r_max + 1):
                for cc in range(c_min, c_max + 1):
                    if not occupancy[rr, cc]:
                        continue
                    d2 = float((rr - row_f) ** 2 + (cc - col_f) ** 2)
                    if d2 < best_d2:
                        best_d2 = d2
                        best = (rr, cc)
            if best is not None:
                return float(best[0]), float(best[1])
        return float(row0), float(col0)

    def furthest_traversable_along(
        center_rc: np.ndarray,
        direction_rc: np.ndarray,
    ) -> np.ndarray:
        last = center_rc.copy()
        for step in range(1, recenter_probe_px + 1):
            probe = center_rc + direction_rc * float(step)
            rr = int(round(probe[0]))
            cc = int(round(probe[1]))
            if rr < 0 or rr >= rows or cc < 0 or cc >= cols or not occupancy[rr, cc]:
                break
            last = probe
        return last

    if n_pts >= 3:
        for i in range(1, n_pts - 1):
            prev_rc = sampled_arr[i - 1]
            curr_rc = sampled_arr[i]
            next_rc = sampled_arr[i + 1]
            tangent = next_rc - prev_rc
            t_norm = float(np.linalg.norm(tangent))
            if t_norm <= 1e-9:
                continue
            tangent = tangent / t_norm
            normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
            n_norm = float(np.linalg.norm(normal))
            if n_norm <= 1e-9:
                continue
            normal = normal / n_norm

            side_a = furthest_traversable_along(curr_rc, normal)
            side_b = furthest_traversable_along(curr_rc, -normal)
            mid = 0.5 * (side_a + side_b)
            rr_mid, cc_mid = nearest_traversable(mid[0], mid[1])
            recentered[i, 0] = rr_mid
            recentered[i, 1] = cc_mid

    # Preserve endpoints exactly as selected A and B.
    recentered[0] = sampled_arr[0]
    recentered[-1] = sampled_arr[-1]

    # Catmull-Rom cubic spline in pixel space, then arc-length re-sampling.
    spline_density = 4
    if n_pts < 2:
        spline_points = recentered.copy()
    elif n_pts == 2:
        spline_points = recentered.copy()
    else:
        per_segment = max(2, int(spline_density))
        curve_chunks: list[np.ndarray] = []
        for i in range(n_pts - 1):
            p0 = recentered[max(i - 1, 0)]
            p1 = recentered[i]
            p2 = recentered[i + 1]
            p3 = recentered[min(i + 2, n_pts - 1)]

            t_vals = np.linspace(0.0, 1.0, per_segment, endpoint=(i == n_pts - 2), dtype=np.float64)
            t = t_vals[:, np.newaxis]
            t2 = t * t
            t3 = t2 * t
            seg = 0.5 * (
                (2.0 * p1)
                + (-p0 + p2) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
            )
            curve_chunks.append(seg)

        spline_dense = np.vstack(curve_chunks)
        arc = np.zeros(spline_dense.shape[0], dtype=np.float64)
        if spline_dense.shape[0] > 1:
            arc[1:] = np.cumsum(np.linalg.norm(np.diff(spline_dense, axis=0), axis=1))
        total_len = float(arc[-1])
        if total_len <= 1e-9:
            spline_points = recentered.copy()
        else:
            n_out = max(n_pts, int((n_pts - 1) * spline_density) + 1)
            s_targets = np.linspace(0.0, total_len, n_out, dtype=np.float64)
            row_interp = np.interp(s_targets, arc, spline_dense[:, 0])
            col_interp = np.interp(s_targets, arc, spline_dense[:, 1])
            spline_points = np.column_stack((row_interp, col_interp))

    # Preserve endpoints exactly after interpolation as well.
    spline_points[0] = sampled_arr[0]
    spline_points[-1] = sampled_arr[-1]

    if debug_output_dir is not None:
        _save_debug_artifacts(
            image=image,
            occupancy=occupancy,
            start=start,
            goal=goal,
            pixel_path=pixel_path,
            output_dir=Path(debug_output_dir),
            prefix=debug_prefix,
            spline_path=spline_points,
        )

    waypoints = [pixel_to_world(float(r), float(c), origin_world, effective_mpp) for r, c in spline_points]
    return np.vstack(waypoints)


def build_spline_from_waypoints(waypoints_world: FloatArray) -> ParametricSpline2D:
    """Build a reusable parametric spline model from ordered waypoints."""
    return build_parametric_spline(np.asarray(waypoints_world, dtype=np.float64))


def select_points_on_map(
    map_path: str | Path,
    origin_world: FloatArray,
    meters_per_pixel: FloatArray,
    traversable_threshold: float = 0.5,
    map_resize_factor: float = 1.0,
) -> tuple[FloatArray, FloatArray]:
    """Open a click UI and return two traversable world points (A then B)."""
    try:
        import tkinter as tk
        from PIL import ImageTk
    except Exception as exc:  # pragma: no cover - depends on local GUI availability
        raise RuntimeError("Interactive point selection requires tkinter and PIL.ImageTk.") from exc

    base_image = Image.open(Path(map_path)).convert("RGB")
    base_gray = np.asarray(base_image.convert("L"), dtype=np.float64) / 255.0
    gray = _resize_grayscale_image(base_gray, map_resize_factor)
    occupancy = gray >= float(traversable_threshold)

    img_u8 = (np.clip(gray, 0.0, 1.0) * 255.0).astype(np.uint8)
    image = Image.fromarray(np.repeat(img_u8[..., np.newaxis], 3, axis=2), mode="RGB")
    effective_mpp = np.asarray(meters_per_pixel, dtype=np.float64) / float(map_resize_factor)

    root = tk.Tk()
    root.title("Select start/end points: click A then B on traversable area")
    root.resizable(True, True)

    screen_w = max(1, int(root.winfo_screenwidth()))
    screen_h = max(1, int(root.winfo_screenheight()))
    max_w = max(1, screen_w - 140)
    max_h = max(1, screen_h - 220)
    scale = min(1.0, max_w / image.width, max_h / image.height)

    disp_w = max(1, int(round(image.width * scale)))
    disp_h = max(1, int(round(image.height * scale)))
    resample = Image.Resampling.NEAREST if scale < 1.0 else Image.Resampling.BILINEAR
    display_image = image.resize((disp_w, disp_h), resample=resample)
    tk_image = ImageTk.PhotoImage(display_image)

    info = tk.Label(
        root,
        text="Click A then B on bright/traversable area. Resize window as needed.",
        anchor="w",
    )
    info.pack(fill="x")

    frame = tk.Frame(root)
    frame.pack(fill="both", expand=True)
    h_scroll = tk.Scrollbar(frame, orient="horizontal")
    v_scroll = tk.Scrollbar(frame, orient="vertical")
    canvas = tk.Canvas(
        frame,
        width=min(disp_w, max_w),
        height=min(disp_h, max_h),
        xscrollcommand=h_scroll.set,
        yscrollcommand=v_scroll.set,
    )

    h_scroll.config(command=canvas.xview)
    v_scroll.config(command=canvas.yview)
    h_scroll.pack(side="bottom", fill="x")
    v_scroll.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    canvas.create_image(0, 0, anchor="nw", image=tk_image)
    canvas.config(scrollregion=(0, 0, disp_w, disp_h))

    root.geometry(f"{min(max_w, disp_w + 30)}x{min(max_h, disp_h + 90)}")

    selected_pixels: list[tuple[int, int]] = []
    marker_radius = 4

    def on_click(event: tk.Event) -> None:
        disp_col = int(canvas.canvasx(event.x))
        disp_row = int(canvas.canvasy(event.y))
        col = int(round(disp_col / scale))
        row = int(round(disp_row / scale))
        if row < 0 or row >= occupancy.shape[0] or col < 0 or col >= occupancy.shape[1]:
            return
        if not occupancy[row, col]:
            print(f"Blocked cell clicked at (row={row}, col={col}). Choose a brighter traversable area.")
            return

        selected_pixels.append((row, col))
        color = "green" if len(selected_pixels) == 1 else "red"
        mark_col = int(round(col * scale))
        mark_row = int(round(row * scale))
        canvas.create_oval(
            mark_col - marker_radius,
            mark_row - marker_radius,
            mark_col + marker_radius,
            mark_row + marker_radius,
            outline=color,
            width=2,
        )
        if len(selected_pixels) >= 2:
            root.after(50, root.destroy)

    canvas.bind("<Button-1>", on_click)
    root.mainloop()

    if len(selected_pixels) < 2:
        raise RuntimeError("Point selection cancelled before choosing both A and B.")

    a_row, a_col = selected_pixels[0]
    b_row, b_col = selected_pixels[1]
    point_a_world = pixel_to_world(a_row, a_col, origin_world, effective_mpp)
    point_b_world = pixel_to_world(b_row, b_col, origin_world, effective_mpp)
    return point_a_world, point_b_world
