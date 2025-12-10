import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt

# simple defaults (change here if you like)
CLUSTER_METHOD = "kmeans"
CLUSTER_K = 1
CLUSTER_EPS = 8.0
CLUSTER_MIN_SAMPLES = 3
SLOWDOWN = 3.0  # playback/video slowdown

# runtime globals (set in main)
GLOBAL_FRAMES: List[np.ndarray] | None = None
GLOBAL_FPS: float | None = None
USE_DERIVS = True
SCALE: float | None = None
KEEP_MASKS = False
CENTRAL_DIFF = False # true if we want our custom differential instead of built in

# ---------------- basic io ----------------
def load_video(path: Path, limit_frames: int | None = None, resize_width: int | None = None) -> Tuple[List[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames: List[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if resize_width:
            scale = resize_width / frame.shape[1]
            frame = cv2.resize(frame, (resize_width, int(frame.shape[0] * scale)))
        frames.append(frame)
        if limit_frames and len(frames) >= limit_frames:
            break
    cap.release()
    if not frames:
        raise ValueError("video empty?")
    return frames, float(fps)


def gray_blur(img: np.ndarray, k: int = 5) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(g, (k, k), 0) if k > 1 else g


# ---------------- detection ----------------
def detect_scratch(
    frames: Sequence[np.ndarray],
    thresh: int = 35,
    min_area: int = 120,
    keep_masks: bool = False,
):
    detections: List[List[Tuple[np.ndarray, int]]] = []
    masks: List[np.ndarray] = []
    bg = gray_blur(frames[0]).astype(np.float32)
    kernel = np.ones((5, 5), np.uint8)
    for frame in frames:
        g = gray_blur(frame)
        cv2.accumulateWeighted(g, bg, 0.02)  # faster adaptation to reduce ghosts
        diff = cv2.absdiff(g, cv2.convertScaleAbs(bg))
        _, mask = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
        color_mask = None
        coords = np.column_stack(np.nonzero(mask))  # (y,x)
        total_pts = len(coords)
        frame_dets: List[Tuple[np.ndarray, int]] = []
        if total_pts > 0:
            sample = coords
            if len(sample) > 5000:
                sample = sample[np.random.choice(len(sample), 5000, replace=False)]
            if CLUSTER_METHOD == "kmeans":
                k = max(1, CLUSTER_K)
                data = sample[:, ::-1].astype(np.float32)  # (x,y)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
                _, labels, centers = cv2.kmeans(data, k, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
                labels = labels.flatten()
                scale = total_pts / float(len(sample))
                for ki in range(k):
                    pts = data[labels == ki]
                    if len(pts) == 0:
                        continue
                    area_est = int(len(pts) * scale)
                    if area_est < min_area:
                        continue
                    centroid = np.mean(pts, axis=0)
                    frame_dets.append((centroid, area_est))
                if keep_masks:
                    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
                    colors = palette()
                    for pt, lab in zip(sample, labels):
                        c = colors[int(lab % len(colors))]
                        y, x = int(pt[0]), int(pt[1])
                        color_mask[y, x] = c
            elif CLUSTER_METHOD == "dbscan":
                data = sample.astype(np.float32)  # (y,x)
                eps2 = CLUSTER_EPS * CLUSTER_EPS
                labels = -np.ones(len(data), dtype=int)
                cluster_id = 0
                for i in range(len(data)):
                    if labels[i] != -1:
                        continue
                    neigh = np.where(np.sum((data - data[i]) ** 2, axis=1) <= eps2)[0]
                    if len(neigh) < CLUSTER_MIN_SAMPLES:
                        labels[i] = -2
                        continue
                    labels[neigh] = cluster_id
                    queue = list(neigh)
                    while queue:
                        j = queue.pop()
                        neigh_j = np.where(np.sum((data - data[j]) ** 2, axis=1) <= eps2)[0]
                        if len(neigh_j) >= CLUSTER_MIN_SAMPLES:
                            for n in neigh_j:
                                if labels[n] in (-1, -2):
                                    labels[n] = cluster_id
                                    queue.append(n)
                cluster_id += 1
                scale = total_pts / float(len(sample))
                for cid in range(cluster_id):
                    pts = data[labels == cid][:, ::-1]  # (x,y)
                    if len(pts) == 0:
                        continue
                    area_est = int(len(pts) * scale)
                    if area_est < min_area:
                        continue
                    centroid = np.mean(pts, axis=0)
                    frame_dets.append((centroid, area_est))
                if keep_masks:
                    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
                    colors = palette()
                    for pt, lab in zip(sample, labels):
                        if lab < 0:
                            continue
                        c = colors[int(lab % len(colors))]
                        y, x = int(pt[0]), int(pt[1])
                        color_mask[y, x] = c
        if keep_masks:
            if color_mask is not None:
                masks.append(color_mask)
            else:
                masks.append(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
        detections.append(frame_dets)
    return (detections, masks) if keep_masks else detections


def detect_mog(
    frames: Sequence[np.ndarray], min_area: int = 30, keep_masks: bool = False
):
    fgbg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=False)
    detections: List[List[Tuple[np.ndarray, int]]] = []
    kernel = np.ones((5, 5), np.uint8)
    masks: List[np.ndarray] = []
    for frame in frames:
        g = gray_blur(frame)
        mask = fgbg.apply(g)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
        if keep_masks:
            masks.append(mask.copy())
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_dets: List[Tuple[np.ndarray, int]] = []
        for c in cnts:
            area = int(cv2.contourArea(c))
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            frame_dets.append((np.array([x + w / 2, y + h / 2], dtype=float), area))
        detections.append(frame_dets)
    return (detections, masks) if keep_masks else detections


# ---------------- tracking ----------------
def track(
    dets_per_frame: Sequence[Sequence[Tuple[np.ndarray, int]]],
    max_dist: float = 50.0,
    max_miss: int = 30,
) -> List[Dict[str, List]]:
    active: List[Dict[str, List]] = []
    finished: List[Dict[str, List]] = []
    for frame_idx, dets in enumerate(dets_per_frame):
        used = set()
        for tr in active:
            tr.setdefault("miss", 0)
            tr["miss"] += 1
        for det_idx, (pt, area) in enumerate(dets):
            best_id = -1
            best_d = max_dist
            for i, tr in enumerate(active):
                if tr["miss"] > max_miss or not tr["pos"]:
                    continue
                d = float(np.linalg.norm(pt - tr["pos"][-1]))
                if d < best_d:
                    best_d = d
                    best_id = i
            if best_id >= 0:
                tr = active[best_id]
                tr["pos"].append(pt)
                tr["frame"].append(frame_idx)
                tr["area"].append(area)
                tr["miss"] = 0
                used.add(det_idx)
        for j, (pt, area) in enumerate(dets):
            if j in used:
                continue
            active.append({"pos": [pt], "frame": [frame_idx], "area": [area], "miss": 0})
        # move stale tracks to finished so we keep their history
        still_active = []
        for tr in active:
            if tr["miss"] > max_miss:
                finished.append(tr)
            else:
                still_active.append(tr)
        active = still_active
    finished.extend(active)
    return finished


# ---------------- derivatives ----------------
def derivatives(positions: np.ndarray, fps: float, scale: float | None, central: bool) -> Dict[str, np.ndarray]:
    dt = 1.0 / fps
    pos = positions * scale if scale else positions
    if central and len(pos) >= 3:
        # central differences on positions
        v_vec = (pos[2:] - pos[:-2]) / (2 * dt)  # length n-2
        a_vec = (pos[2:] - 2 * pos[1:-1] + pos[:-2]) / (dt * dt)  # length n-2
        j_vec = np.diff(a_vec, axis=0) / dt
        jo_vec = np.diff(j_vec, axis=0) / dt
    else:
        v_vec = np.diff(pos, axis=0) / dt
        a_vec = np.diff(v_vec, axis=0) / dt
        j_vec = np.diff(a_vec, axis=0) / dt
        jo_vec = np.diff(j_vec, axis=0) / dt
    return {
        "pos": pos,
        "speed": np.linalg.norm(v_vec, axis=1) if len(v_vec) else np.array([]),
        "acc": np.linalg.norm(a_vec, axis=1) if len(a_vec) else np.array([]),
        "jerk": np.linalg.norm(j_vec, axis=1) if len(j_vec) else np.array([]),
        "jounce": np.linalg.norm(jo_vec, axis=1) if len(jo_vec) else np.array([]),
    }


# ---------------- helper for visualization ----------------
def palette():
    return [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (128, 0, 255),
        (255, 128, 0),
        (0, 128, 255),
        (128, 255, 0),
        (255, 0, 128),
        (0, 255, 128),
    ]


def helper1(arr, idx, offset):
    pos = idx + offset
    return float(arr[pos]) if 0 <= pos < len(arr) else 0.0


def render_vis(
    frames: Sequence[np.ndarray],
    tracks: List[Dict[str, List]],
    derivs: List[Dict[str, np.ndarray]],
    masks: List[np.ndarray] | None,
    fps: float,
    mode: str,
    show: bool,
    save_path: Path | None,
    slowdown: float,
) -> None:
    if not frames:
        return
    labels = np.arange(len(tracks))
    cols = palette()
    writer = None
    factor = max(1.0, float(slowdown))
    target_fps = max(1.0, fps / factor)
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        h, w = frames[0].shape[:2]
        out_w = w * 2 if masks else w
        writer = cv2.VideoWriter(str(save_path), fourcc, target_fps, (out_w, h))
    for frame_idx, f in enumerate(frames):
        frame = f.copy()
        info_lines: List[str] = []
        for tid, tr in enumerate(tracks):
            if frame_idx not in tr["frame"]:
                continue
            j = tr["frame"].index(frame_idx)
            p = tr["pos"][j]
            col = cols[int(labels[tid] % len(cols))] if len(labels) > tid else (200, 200, 200)
            r = int(max(3, (tr["area"][j] / np.pi) ** 0.5))
            cv2.circle(frame, (int(p[0]), int(p[1])), r, col, 2)
            if j > 0:
                prev = tr["pos"][j - 1]
                cv2.line(frame, (int(prev[0]), int(prev[1])), (int(p[0]), int(p[1])), col, 2)
            d = derivs[tid]
            s = helper1(d["speed"], j, -1)
            a = helper1(d["acc"], j, -2)
            jk = helper1(d["jerk"], j, -3)
            jo = helper1(d["jounce"], j, -4)
            info_lines.append(
                f"id {tid} s={s:.1f} a={a:.1f} j={jk:.1f} jo={jo:.1f}"
            )
        cv2.putText(frame, f"{mode} frame {frame_idx}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        # info stacked in top-left, not attached to objects
        for k, line in enumerate(info_lines):
            cv2.putText(frame, line, (10, 40 + 16 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        display = frame
        if masks and frame_idx < len(masks):
            mask_img = masks[frame_idx]
            if mask_img.ndim == 2 or mask_img.shape[2] == 1:
                mask_img = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)
            cv2.putText(mask_img, "mask/clusters", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            display = np.hstack([frame, mask_img])
        if writer:
            writer.write(display)
        if show:
            cv2.imshow("tracking", display)
            delay = max(1, int(1000 / target_fps))
            if cv2.waitKey(delay) & 0xFF == ord("q"):
                break
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()


# ---------------- pipeline ----------------
def run_pipeline(
    mode: str,
) -> Dict:
    if GLOBAL_FRAMES is None or GLOBAL_FPS is None:
        raise RuntimeError("Globals not initialized. Run main() first.")
    raw = (
        detect_scratch(
            GLOBAL_FRAMES,
            keep_masks=KEEP_MASKS,
        )
        if mode == "scratch"
        else detect_mog(GLOBAL_FRAMES, min_area=120, keep_masks=KEEP_MASKS)
    )
    if KEEP_MASKS:
        dets, masks = raw  # type: ignore
    else:
        dets, masks = raw, None  # type: ignore
    # Allow small gaps, but still restart if it disappears too long.
    tracks = [tr for tr in track(dets, max_dist=100.0, max_miss=8) if len(tr["pos"]) > 2]
    feats = []
    derivs = []
    for tr in tracks:
        d = derivatives(
            np.stack(tr["pos"], axis=0),
            GLOBAL_FPS,
            SCALE,
            central=CENTRAL_DIFF,
        )
        derivs.append(d)
        base = [np.mean(d["speed"]), np.max(d["speed"])]
        if USE_DERIVS:
            base.extend([np.mean(d["acc"]), np.mean(d["jerk"]), np.mean(d["jounce"])])
        feats.append(base)
    feats_arr = np.array(feats) if feats else np.zeros((0, 2))
    return {"tracks": tracks, "derivs": derivs, "features": feats_arr, "masks": masks}


# ---------------- cli ----------------
def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simple motion extractor")
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--pipeline", choices=["scratch", "library", "both"], default="both")
    p.add_argument("--no-derivatives", action="store_true")
    p.add_argument("--scale-m-per-px", type=float, default=None, help="convert pixels to meters if known")
    p.add_argument("--visualize", action="store_true", help="Show tracking window (press q to quit).")
    p.add_argument("--visualize-stages", action="store_true", help="Side-by-side view with foreground mask.")
    p.add_argument("--central-diff", action="store_true", help="Use central differences for derivatives.")
    return p.parse_args()


def main() -> None:
    args = parse()
    global GLOBAL_FRAMES, GLOBAL_FPS, USE_DERIVS, SCALE, KEEP_MASKS, CENTRAL_DIFF
    frames, fps = load_video(args.video, None, None)
    GLOBAL_FRAMES = frames
    GLOBAL_FPS = fps
    modes = ["scratch", "library"] if args.pipeline == "both" else [args.pipeline]
    USE_DERIVS = not args.no_derivatives
    SCALE = args.scale_m_per_px
    KEEP_MASKS = args.visualize_stages or args.visualize
    CENTRAL_DIFF = args.central_diff
    for mode in modes:
        out = run_pipeline(mode)
        if args.visualize or args.visualize_stages:
            save_path = Path(f"annotated_{mode}.mp4") if len(modes) > 1 else Path("annotated.mp4")
            render_vis(
                frames=frames,
                tracks=out["tracks"],
                derivs=out["derivs"],
                masks=out["masks"],
                fps=fps,
                mode=mode,
                show=args.visualize or args.visualize_stages,
                save_path=save_path,
                slowdown=SLOWDOWN,
            )
        # simple plot of speed/acc for each track
        plt.figure(figsize=(8, 5))
        for i, (tr, d) in enumerate(zip(out["tracks"], out["derivs"])):
            frame_ids = tr["frame"]
            if len(d["speed"]):
                plt.plot(frame_ids[1:], d["speed"], label=f"track {i} speed")
            if len(d["acc"]):
                plt.plot(frame_ids[2:], d["acc"], linestyle="--", label=f"track {i} acc")
        plt.title(f"{mode} tracks={len(out['tracks'])}")
        plt.xlabel("frame")
        plt.ylabel("pixels/frame (speed) or pixels/frame^2 (acc)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
