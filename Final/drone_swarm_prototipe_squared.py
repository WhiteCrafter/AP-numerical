import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


SWARM_SPACING = 2.0
SWARM_FRAMES = 80
SWARM_DELAY = 2.0
SWARM_INTERVAL = 50
SWARM_V_MAX = 4.5
SWARM_K_V = 1.2
SWARM_K_D = 0.4
SWARM_K_REP = 0.6
SWARM_R_SAFE = 0.8
SWARM_MASS = 1.0
SWARM_ARRIVE_TOL = 0.15
SWARM_MAX_EXTRA = 600
SWARM_HOLD = 0.0
SWARM_HIDE_TARGETS = False


def build_font_8x8():
    # Simple 5x7 patterns padded to 8x8 AI generated
    font_5x7 = {
        "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
        "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
        "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
        "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
        "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
        "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
        "G": ["01110", "10001", "10000", "10111", "10001", "10001", "01110"],
        "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
        "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
        "J": ["00111", "00010", "00010", "00010", "00010", "10010", "01100"],
        "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
        "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
        "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
        "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
        "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
        "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
        "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
        "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
        "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
        "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
        "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
        "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
        "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
        "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
        "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
        "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
        "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
        "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
        "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
        "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
        "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
        "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
        "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
        "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
        "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
        "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
        "?": ["01110", "10001", "00001", "00010", "00100", "00000", "00100"],
        "!": ["00100", "00100", "00100", "00100", "00100", "00000", "00100"],
        ",": ["00000", "00000", "00000", "00000", "00000", "00100", "01000"],
        ".": ["00000", "00000", "00000", "00000", "00000", "00100", "00100"],
        "<": ["00010", "00100", "01000", "10000", "01000", "00100", "00010"],
        ">": ["01000", "00100", "00010", "00001", "00010", "00100", "01000"],
        "=": ["00000", "00000", "11111", "00000", "11111", "00000", "00000"],
        "+": ["00000", "00100", "00100", "11111", "00100", "00100", "00000"],
        " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
    }

    font_8x8 = {}
    for ch, rows in font_5x7.items():
        padded = []
        for row in rows:
            padded.append("0" + row + "00")
        padded.append("00000000")
        grid = np.array([[1 if c == "1" else 0 for c in r] for r in padded], dtype=int)
        font_8x8[ch] = grid
    return font_8x8


def text_to_grid(text, font, gap=1):
    text = text.upper()
    if not text:
        return np.zeros((8, 0), dtype=int)

    grids = []
    for ch in text:
        grids.append(font.get(ch, font["?"]))

    parts = []
    for i, grid in enumerate(grids):
        parts.append(grid)
        if i != len(grids) - 1:
            parts.append(np.zeros((8, gap), dtype=int))
    return np.concatenate(parts, axis=1)


def grid_to_targets(grid, spacing=1.0):
    rows, cols = grid.shape
    positions = []
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 1:
                x = c * spacing
                y = (rows - 1 - r) * spacing
                positions.append([x, y])

    if not positions:
        return np.zeros((0, 2), dtype=float)

    positions = np.array(positions, dtype=float)
    positions[:, 0] -= (cols - 1) * spacing / 2.0
    positions[:, 1] -= (rows - 1) * spacing / 2.0
    return positions


def grid_start_positions(num_drones, targets):
    x_min, x_max = targets[:, 0].min(), targets[:, 0].max()
    y_min, y_max = targets[:, 1].min(), targets[:, 1].max()

    unique_x = np.unique(targets[:, 0])
    unique_y = np.unique(targets[:, 1])
    dx = np.min(np.diff(unique_x)) if len(unique_x) > 1 else 1.0
    dy = np.min(np.diff(unique_y)) if len(unique_y) > 1 else 1.0
    spacing = max(dx, dy) * 2

    cols = int(np.ceil(np.sqrt(num_drones)))
    rows = int(np.ceil(num_drones / cols))

    grid_x = np.arange(cols) * spacing
    grid_y = np.arange(rows) * spacing
    gx, gy = np.meshgrid(grid_x, grid_y)
    points = np.column_stack([gx.ravel(), gy.ravel()])[:num_drones]

    center_x = (x_min + x_max) / 2.0
    margin = spacing * 2.0
    base_y = y_min - margin - (rows - 1) * spacing
    points[:, 0] -= (cols - 1) * spacing / 2.0
    points[:, 1] -= (rows - 1) * spacing / 2.0
    points[:, 0] += center_x
    points[:, 1] += base_y + (rows - 1) * spacing / 2.0
    return points


def standby_positions(count, targets, spacing):
    if count <= 0:
        return np.zeros((0, 2), dtype=float)

    if len(targets) == 0:
        xs = (np.arange(count) - (count - 1) / 2.0) * spacing
        ys = np.zeros(count)
        return np.column_stack([xs, ys])

    x_min, x_max = targets[:, 0].min(), targets[:, 0].max()
    y_min = targets[:, 1].min()
    center_x = (x_min + x_max) / 2.0

    xs = (np.arange(count) - (count - 1) / 2.0) * spacing + center_x
    ys = np.full(count, y_min - spacing * 2.0)
    return np.column_stack([xs, ys])


def pad_targets(targets, total, spacing):
    if len(targets) >= total:
        return targets[:total]
    standby = standby_positions(total - len(targets), targets, spacing)
    return np.vstack([targets, standby])


class AssignmentEngine:
    def assign(self, starts, targets):
        # Minimize sum of squared distances (requires SciPy).
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError as exc:
            raise ImportError("SciPy is required for squared assignment.") from exc

        count = min(len(starts), len(targets))
        starts = starts[:count]
        targets = targets[:count]

        diff = starts[:, None, :] - targets[None, :, :]
        cost = diff[:, :, 0] ** 2 + diff[:, :, 1] ** 2
        row_ind, col_ind = linear_sum_assignment(cost)

        assigned = -np.ones(count, dtype=int)
        assigned[row_ind] = col_ind
        return assigned


class MotionPlanner:
    def __init__(
        self,
        frames=80,
        delay=2.0,
        interval=50,
        v_max=2.5,
        k_v=1.2,
        k_d=0.4,
        k_rep=0.6,
        r_safe=0.8,
        mass=1.0,
        arrive_tol=0.3,
        max_extra=200,
        hold_after=2.0,
    ):
        self.frames = frames
        self.delay = delay
        self.interval = interval
        self.dt = interval / 1000.0
        self.v_max = v_max
        self.k_v = k_v
        self.k_d = k_d
        self.k_rep = k_rep
        self.r_safe = r_safe
        self.mass = mass
        self.arrive_tol = arrive_tol
        self.max_extra = max_extra
        self.hold_after = hold_after

    def _saturate(self, vec):
        norms = np.linalg.norm(vec, axis=1)
        scale = np.ones_like(norms)
        mask = norms > self.v_max
        scale[mask] = self.v_max / norms[mask]
        return vec * scale[:, None]

    def _repulsion_forces(self, positions):
        n = len(positions)
        forces = np.zeros_like(positions)
        for i in range(n):
            for j in range(i + 1, n):
                diff = positions[i] - positions[j]
                dist = np.linalg.norm(diff)
                if dist < self.r_safe and dist > 1e-6:
                    force = self.k_rep * diff / (dist ** 3)
                    forces[i] += force
                    forces[j] -= force
        return forces

    def _step(self, positions, velocities, targets, time_left):
        # Initial value problem: x(0) = positions, v(0) = velocities.
        desired = targets - positions
        distance = np.linalg.norm(desired, axis=1)
        direction = np.zeros_like(desired)
        mask = distance > 1e-8
        direction[mask] = desired[mask] / distance[mask][:, None]
        speed = np.minimum(self.v_max, self.k_v * distance)
        v_des = direction * speed[:, None]
        v_sat = self._saturate(v_des)
        repulsion = self._repulsion_forces(positions)
        accel = self.k_v * v_sat - self.k_d * velocities + repulsion / self.mass

        velocities = velocities + accel * self.dt
        speeds = np.linalg.norm(velocities, axis=1)
        mask = speeds > self.v_max
        if np.any(mask):
            velocities[mask] *= (self.v_max / speeds[mask])[:, None]
        positions = positions + velocities * self.dt
        return positions, velocities

    def _simulate_segment(self, starts, targets, move_steps, hold_steps, v0):
        positions = starts.copy()
        velocities = v0.copy()
        frames = []
        steps = 0
        max_dist = np.inf
        total_time = max(self.dt, move_steps * self.dt)
        while steps < move_steps or (steps < move_steps + self.max_extra and max_dist > self.arrive_tol):
            frames.append(positions.copy())
            time_left = max(self.dt, total_time - steps * self.dt)
            positions, velocities = self._step(positions, velocities, targets, time_left)
            max_dist = np.max(np.linalg.norm(targets - positions, axis=1))
            steps += 1

        if max_dist <= self.arrive_tol:
            positions = targets.copy()
            velocities = np.zeros_like(velocities)

        for _ in range(hold_steps):
            frames.append(positions.copy())
        return np.array(frames), positions, velocities

    def build_segments(self, starts, target_sets, texts, assigner):
        segments = []
        current = starts
        velocities = np.zeros_like(starts)

        hold_frames = int(round((self.delay * 1000.0) / self.interval)) if len(target_sets) > 1 else 0
        hold_frames = max(0, hold_frames)
        hold_after_frames = int(round((self.hold_after * 1000.0) / self.interval))
        hold_after_frames = max(0, hold_after_frames)
        segment_lengths = []

        for idx, targets in enumerate(target_sets):
            assignment = assigner.assign(current, targets)
            end_targets = targets[assignment]

            hold_steps = hold_after_frames
            if idx < len(target_sets) - 1:
                hold_steps += hold_frames
            frames, current, velocities = self._simulate_segment(
                current,
                end_targets,
                self.frames,
                hold_steps,
                velocities,
            )

            segments.append(
                {
                    "positions": frames,
                    "targets": targets,
                    "text": texts[idx],
                }
            )
            segment_lengths.append(len(frames))

        total_frames = sum(segment_lengths)
        return segments, segment_lengths, total_frames


class SwarmRenderer:
    def __init__(self, starts, segments, segment_lengths, total_frames, planner, show_targets=True):
        self.starts = starts
        self.segments = segments
        self.segment_lengths = segment_lengths
        self.total_frames = total_frames
        self.planner = planner
        self.show_targets = show_targets
        self.current_segment = {"idx": None}

        all_points = [starts] + [s["targets"] for s in segments]
        all_points = np.vstack(all_points)
        min_x, max_x = all_points[:, 0].min(), all_points[:, 0].max()
        min_y, max_y = all_points[:, 1].min(), all_points[:, 1].max()
        margin = 2.0

        self.fig, self.ax = plt.subplots(figsize=(9, 5))
        self.ax.set_aspect("equal")
        self.ax.set_facecolor("black")
        self.fig.patch.set_facecolor("black")
        self.ax.set_xlim(min_x - margin, max_x + margin)
        self.ax.set_ylim(min_y - margin, max_y + margin)
        self.ax.axis("off")
        self.ax.set_title(f"Dronw Swarm Prototype: {segments[0]['text']}", color="white")

        target_alpha = 0.2 if show_targets else 0.0
        self.target_scatter = self.ax.scatter(
            segments[0]["targets"][:, 0],
            segments[0]["targets"][:, 1],
            s=8,
            c="white",
            alpha=target_alpha,
        )
        self.scatter = self.ax.scatter(starts[:, 0], starts[:, 1], s=28, c="cyan")

    def init_draw(self):
        if self.segments:
            self.target_scatter.set_offsets(self.segments[0]["targets"])
        self.scatter.set_offsets(self.starts)
        return (self.scatter, self.target_scatter)

    def _locate_segment(self, frame):
        idx = 0
        local = frame
        for length in self.segment_lengths:
            if local < length:
                return idx, local
            local -= length
            idx += 1
        return len(self.segment_lengths) - 1, self.segment_lengths[-1] - 1

    def update(self, frame):
        seg_idx, local = self._locate_segment(frame)
        segment = self.segments[seg_idx]

        if self.current_segment["idx"] != seg_idx:
            self.target_scatter.set_offsets(segment["targets"])
            self.ax.set_title(f"Dronw Swarm Prototype: {segment['text']}", color="white")
            self.current_segment["idx"] = seg_idx

        positions = segment["positions"]
        if local >= len(positions):
            current = positions[-1]
        else:
            current = positions[local]
        self.scatter.set_offsets(current)
        return (self.scatter, self.target_scatter)

    def show(self):
        self.init_draw()
        self.anim = FuncAnimation(
            self.fig,
            self.update,
            frames=self.total_frames,
            interval=self.planner.interval,
            init_func=self.init_draw,
            blit=False,
        )
        plt.show()


def run_texts(
    texts,
    spacing=2.0,
    frames=80,
    delay=2.0,
    interval=50,
    v_max=4.5,
    k_v=1.2,
    k_d=0.4,
    k_rep=0.6,
    r_safe=0.8,
    mass=1.0,
    arrive_tol=0.15,
    max_extra=600,
    hold_after=0.0,
    hide_targets=False,
):
    font = build_font_8x8()
    clean_texts = [t for t in texts if t.strip()]
    if not clean_texts:
        print("No text provided.")
        return

    raw_targets = []
    for text in clean_texts:
        grid = text_to_grid(text, font, gap=1)
        raw_targets.append(grid_to_targets(grid, spacing=spacing))

    total_drones = max(len(t) for t in raw_targets)
    if total_drones == 0:
        print("No active dots for these texts.")
        return

    target_sets = [pad_targets(t, total_drones, spacing) for t in raw_targets]
    reference_targets = next((t for t in raw_targets if len(t) > 0), None)
    if reference_targets is None:
        print("No active dots for these texts.")
        return

    starts = grid_start_positions(total_drones, reference_targets)
    assigner = AssignmentEngine()
    planner = MotionPlanner(
        frames=frames,
        delay=max(0.0, delay),
        interval=interval,
        v_max=v_max,
        k_v=k_v,
        k_d=k_d,
        k_rep=k_rep,
        r_safe=r_safe,
        mass=mass,
        arrive_tol=arrive_tol,
        max_extra=max_extra,
        hold_after=max(0.0, hold_after),
    )
    segments, segment_lengths, total_frames = planner.build_segments(
        starts,
        target_sets,
        clean_texts,
        assigner,
    )
    renderer = SwarmRenderer(
        starts,
        segments,
        segment_lengths,
        total_frames,
        planner,
        show_targets=not hide_targets,
    )
    renderer.show()


def main():
    parser = argparse.ArgumentParser(description="Dronw swarm prototype text morphing demo.")
    parser.add_argument("texts", nargs="*", default=["HELLO"], help="List of texts to morph through.")
    args = parser.parse_args()

    run_texts(
        args.texts,
        spacing=SWARM_SPACING,
        frames=SWARM_FRAMES,
        delay=SWARM_DELAY,
        interval=SWARM_INTERVAL,
        v_max=SWARM_V_MAX,
        k_v=SWARM_K_V,
        k_d=SWARM_K_D,
        k_rep=SWARM_K_REP,
        r_safe=SWARM_R_SAFE,
        mass=SWARM_MASS,
        arrive_tol=SWARM_ARRIVE_TOL,
        max_extra=SWARM_MAX_EXTRA,
        hold_after=SWARM_HOLD,
        hide_targets=SWARM_HIDE_TARGETS,
    )


if __name__ == "__main__":
    main()
