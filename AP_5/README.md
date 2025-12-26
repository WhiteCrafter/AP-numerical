# AP_5 - Spline interpolation of character shapes

This folder contains a small, reproducible experiment for spline interpolation. The code builds three character outlines (C, 6, Z) from simple 2D point sets, fits splines to those points, and visualizes how spline choice, node placement, and node count affect the fit.

## Files
- `AP_5/main.py`: Generates the shapes, selects nodes, fits splines, and plots results.
- `AP_5/output/*.png`: Figures saved when you run the script.

## How the code is organized
- Shape construction (`build_shapes`): Each character is assembled from sampled geometric primitives (arc for C, loop plus tail for 6, and polyline for Z).
- Parameterization (`arc_length_param`): Points are mapped to a normalized arc-length parameter so x(t) and y(t) can be interpolated consistently.
- Node selection (`select_nodes_*`): Two strategies are used: uniform subsampling and a corner-aware method that favors high-turning-angle points.
- Spline fitting (`fit_cubic_spline`, `fit_bspline`): A natural cubic spline is compared with a cubic B-spline. Both interpolate the selected nodes.
- Quality proxy (`mean_nearest_distance`): A simple distance-to-curve metric is shown in subplot titles to compare fits.
- Visualization (`plot_shape_experiments`): For each character, a 2x2 grid compares the baseline, corner-aware nodes, B-splines, and a higher node count.

## How the experiments relate to spline quality
- Node count: The higher-node panel (n=20) typically reduces error and better follows sharper features.
- Node placement: The corner-aware strategy keeps sharp turns for the Z, improving the fit without increasing the total nodes.
- Spline type: B-splines tend to smooth corners slightly, while cubic interpolants pass exactly through nodes and can overshoot if nodes are sparse.

## Run
```bash
python AP_5/main.py
```

The script saves figures to `AP_5/output` and also displays them with matplotlib.


