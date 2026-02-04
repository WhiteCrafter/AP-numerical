# Workflow Overview

This document explains how the project flows from OCR image input to animated drone motion, with small code snippets and brief explanations. It also includes a few light math notes where useful.

---

## 1) Entry point: `command.py`

The main entry point is `Final/command.py`. It reads an image, performs OCR, then passes the text into the swarm renderer and appends `"HAPPY NEW YEAR!"`.

```python
def main():
    parser = argparse.ArgumentParser(description="OCR image and render drone swarm text.")
    parser.add_argument("image", help="Path to input image.")
    parser.add_argument("--lang", default="en", help="Language code (default: en).")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available.")
    parser.add_argument("--no-preprocess", action="store_true", help="Skip OCR preprocessing step.")
    args = parser.parse_args()

    text = handwriting_ocr.read_handwriting_from_path(
        args.image,
        lang=args.lang,
        use_gpu=args.gpu,
        preprocess=not args.no_preprocess,
    )
    text = text.strip() or "?"

    swarm.run_texts([text, "HAPPY NEW YEAR!"], ...)
```

What happens:
- `command.py` is the main entry for the full pipeline.
- OCR runs first, then the output text is passed to the swarm.

---

## 2) Swarm entry function used by `command.py`

The OCR pipeline calls `run_texts()` from `Final/drone_swarm_prototipe_squared.py`. This is the only swarm entry point used by `command.py`.

---

## 2) Text → grid (8x8 font)

Each character uses a 5x7 bitmap padded to an 8x8 grid. Unknown characters map to `"?"`.

```python
def build_font_8x8():
    # Simple 5x7 patterns padded to 8x8 AI generated
    font_5x7 = {
        "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
        # ...
    }
    # pad to 8x8
    font_8x8 = {}
    for ch, rows in font_5x7.items():
        padded = ["0" + row + "00" for row in rows]
        padded.append("00000000")
        grid = np.array([[1 if c == "1" else 0 for c in r] for r in padded], dtype=int)
        font_8x8[ch] = grid
    return font_8x8
```

```python
def text_to_grid(text, font, gap=1):
    text = text.upper()
    grids = [font.get(ch, font["?"]) for ch in text]
    parts = []
    for i, grid in enumerate(grids):
        parts.append(grid)
        if i != len(grids) - 1:
            parts.append(np.zeros((8, gap), dtype=int))
    return np.concatenate(parts, axis=1)
```

What happens:
- Each character becomes an 8x8 bitmap.
- Characters are concatenated with a 1-column gap to form a full text grid.

---

## 3) Grid → 2D target points

Active pixels (1s) become 2D target coordinates with fixed spacing.

```python
def grid_to_targets(grid, spacing=1.0):
    rows, cols = grid.shape
    positions = []
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 1:
                x = c * spacing
                y = (rows - 1 - r) * spacing
                positions.append([x, y])
    positions = np.array(positions, dtype=float)
    positions[:, 0] -= (cols - 1) * spacing / 2.0
    positions[:, 1] -= (rows - 1) * spacing / 2.0
    return positions
```

What happens:
- The bitmap is converted to point coordinates.
- The set is centered around (0,0).

---

## 4) Start positions and standby padding

Drones start below the text in a grid. If text lengths differ, extra drones are placed in a standby line below.

```python
def grid_start_positions(num_drones, targets):
    # place a rectangular grid below the target text
    # ...
    return points
```

```python
def pad_targets(targets, total, spacing):
    if len(targets) >= total:
        return targets[:total]
    standby = standby_positions(total - len(targets), targets, spacing)
    return np.vstack([targets, standby])
```

---

## 5) Assignment (min squared distance)

Each drone is matched to a target by minimizing the sum of squared distances.

```python
class AssignmentEngine:
    def assign(self, starts, targets):
        from scipy.optimize import linear_sum_assignment
        diff = starts[:, None, :] - targets[None, :, :]
        cost = diff[:, :, 0] ** 2 + diff[:, :, 1] ** 2
        row_ind, col_ind = linear_sum_assignment(cost)
        assigned = -np.ones(len(starts), dtype=int)
        assigned[row_ind] = col_ind
        return assigned
```

Math note:
$$\min_{\pi} \sum_i \|x_i - y_{\pi(i)}\|^2$$

---

## 6) Motion planner (physics-style)

Each drone updates its velocity using a saturated velocity command, damping, and repulsion. The key steps:

```python
def _step(self, positions, velocities, targets, time_left):
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
    positions = positions + velocities * self.dt
    return positions, velocities
```

Math note (simplified):
$$\dot{x} = v,\quad \dot{v} = k_v v_{sat} - k_d v + f_{rep}/m$$

Repulsion between drones (only if within safety radius):
$$f_{rep}(x_i, x_j) = k_{rep}\frac{x_i - x_j}{\|x_i - x_j\|^3},\ \ \|x_i - x_j\| < R_{safe}$$

---

## 7) Segment building and timing

Text transitions are split into segments. Each segment simulates `frames` steps, plus optional extra steps if drones haven’t arrived, and a hold time after arrival.

```python
frames, current, velocities = self._simulate_segment(
    current,
    end_targets,
    self.frames,
    hold_steps,
    velocities,
)
```

What happens:
- Each segment simulates motion from current → next targets.
- It can hold for `hold_after` seconds at the end.

---

## 8) Rendering

Matplotlib animates the drones. It updates positions frame by frame.

```python
class SwarmRenderer:
    def update(self, frame):
        seg_idx, local = self._locate_segment(frame)
        positions = self.segments[seg_idx]["positions"]
        current = positions[min(local, len(positions) - 1)]
        self.scatter.set_offsets(current)
        return (self.scatter, self.target_scatter)
```

What happens:
- A scatter plot is updated with precomputed positions.
- The title changes per segment to show the current text.

---

## 9) OCR → Swarm (if used)

`Final/command.py` uses `Final/handwriting_ocr.py` to read a handwritten image, then passes the text to the swarm to render, followed by `"HAPPY NEW YEAR!"`.

```python
text = handwriting_ocr.read_handwriting_from_path(...)
swarm.run_texts([text, "HAPPY NEW YEAR!"], ...)
```

---

## Summary of the call chain

```text
main()
  -> run_texts(texts)
      -> build_font_8x8()
      -> text_to_grid() -> grid_to_targets()
      -> pad_targets()
      -> AssignmentEngine.assign()
      -> MotionPlanner.build_segments()
      -> SwarmRenderer.show() -> animation
```

This is the full pipeline from input text to animated drone formation.
