import numpy as np
import plotly.graph_objects as go


def norm1(v):
    return np.abs(v).sum()


def norm2(v):
    return np.sqrt((v**2).sum())


def norm_inf(v):
    return np.max(np.abs(v))


# change this to norm2 or norm_inf if you want another shape
ACTIVE_NORM = norm1

# how many samples / slider slices / tolerances
POINTS = 200_000
BOUNDARY_TOL = 0.02
SLICE_TOL = 0.05
W_STEPS = 31


def make_shell(show=True):
    pts4 = np.random.uniform(-1, 1, (POINTS, 4))
    vals = np.apply_along_axis(ACTIVE_NORM, 1, pts4)
    mask = (vals > 1 - BOUNDARY_TOL) & (vals < 1 + BOUNDARY_TOL)
    pts = pts4[mask]
    if pts.size == 0:
        print("no points on the shell, try increasing POINTS or tolerance")
        return None

    x, y, z, w = [pts[:, i] for i in range(4)]
    w_vals = np.linspace(-1, 1, W_STEPS)

    slices = []
    for target in w_vals:
        hit = np.abs(w - target) <= SLICE_TOL
        slices.append((x[hit], y[hit], z[hit]))

    start = 0
    for idx, (sx, _, _) in enumerate(slices):
        if sx.size:
            start = idx
            break

    sx, sy, sz = slices[start]
    if sx.size == 0:
        sx, sy, sz = x, y, z

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=sx,
                y=sy,
                z=sz,
                mode="markers",
                marker=dict(size=2),
            )
        ]
    )

    slider_steps = []
    for idx, (sx, sy, sz) in enumerate(slices):
        if sx.size == 0:
            sx = sy = sz = np.array([np.nan])
        slider_steps.append(
            dict(
                method="update",
                args=[{"x": [sx], "y": [sy], "z": [sz]}],
                label=f"{w_vals[idx]:.2f}",
            )
        )

    fig.update_layout(
        scene=dict(
            aspectmode="cube",
            xaxis=dict(range=[-1, 1], title="x"),
            yaxis=dict(range=[-1, 1], title="y"),
            zaxis=dict(range=[-1, 1], title="z"),
        ),
        sliders=[dict(active=start, steps=slider_steps)],
        margin=dict(l=0, r=0, t=20, b=0),
        title=f"4D shell slices ({ACTIVE_NORM.__name__})",
    )
    fig.update_traces(hoverinfo="skip")
    if show:
        fig.show()
    return fig


if __name__ == "__main__":
    make_shell()
