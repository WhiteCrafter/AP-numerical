# Motion Tracking (CP_1)

Simple video motion extraction with two pipelines:
- **scratch**: basic frame differencing, mask clustering (k-means/DBSCAN), nearest-neighbor tracking.
- **library**: OpenCV MOG2 background subtractor with the same tracker.

Outputs:
- Tracks with speed/acceleration/jerk/jounce (pixel units; optional scale to real units).
- Annotated video(s) and optional stage view (raw/diff/mask).
- Matplotlib plots of per-track speed/acc.

## Quick start
```
python main.py --video path/to/video.mp4 --pipeline scratch --visualize
# or
python main.py --video path/to/video.mp4 --pipeline library --visualize
```

Useful flags:
- `--pipeline scratch|library|both` (default both).
- `--visualize` to show tracking; `--visualize-stages` to also show raw/diff/mask.
- `--scale-m-per-px <val>` to convert pixels to meters for derivatives.
- `--central-diff` to use central finite differences for derivatives (default is forward).
- `--no-derivatives` to skip derivative-based stats.

Outputs:
- Annotated video saved in the script directory (`annotated.mp4` or `annotated_<mode>.mp4` when both).
- Live windows (press `q` to exit).
- A simple plot window for speed/acc per track.

## Flow (scratch pipeline)
```
load video -> grayscale+blur -> running background -> diff + threshold ->
morph cleanup -> mask clustering (k-means or dbscan) -> centroids ->
nearest-neighbor tracking -> derivatives (speed/acc/jerk/jounce) ->
visualize/save + plots
```
Library pipeline swaps the detection step with MOG2 foreground masks.

## Options for single vs multiple objects
- Single object: leave k-means default `CLUSTER_K=1`.
- Multiple objects: set `CLUSTER_METHOD="kmeans"` and bump `CLUSTER_K` in the code (module constants at top), or use DBSCAN (`CLUSTER_METHOD="dbscan"`) and tune `CLUSTER_EPS`/`CLUSTER_MIN_SAMPLES`.

## Notes on artifacts and limitations
- Background ghosts: running average background updates can leave faint after-images when an object stops or moves suddenly; they clear after a few frames. Faster background updates reduce ghosts but may absorb slow objects.
- Occlusion/gaps: if detections vanish for many frames, the tracker may start a new track when the object reappears.
- Sudden acceleration or sharp turns can momentarily leave a small residual blob behind.
- Stationary objects may fade into the background (especially in the scratch pipeline).
- Threshold and clustering parameters are simple; noisy videos or touching blobs may need minor tuning of the module-level constants (`CLUSTER_*` at top of `main.py`).

## Project requirements
- Python 3, `opencv-python`, `numpy`, `matplotlib`.
