import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -------- NORM FUNCTIONS --------
def norm1(v):   return np.abs(v).sum()
def norm2(v):   return np.sqrt((v**2).sum())

# -------- SAMPLE BASE POINTS --------
N = 200000
pts3 = np.random.uniform(-1, 1, (N, 3))

values = np.linspace(-1, 1, 21)
eps = 0.05   # slice thickness

frames = []
steps = []

# -------- CREATE SUBPLOTS --------
fig = make_subplots(
    rows=1, cols=2,
    specs=[[{'type': 'scene'}, {'type': 'scene'}]],
    subplot_titles=("L1 Norm Slice", "L2 Norm Slice")
)

# -------- INITIAL FRAME COMPUTATION --------
w0 = values[0]

# L1 slice
pts4_L1 = np.column_stack([pts3, np.full(N, w0)])
vals_L1 = np.apply_along_axis(norm1, 1, pts4_L1)
mask_L1 = (vals_L1 > 1 - eps) & (vals_L1 < 1 + eps)
slice_L1 = pts3[mask_L1]

# L2 slice
vals_L2 = np.apply_along_axis(norm2, 1, pts4_L1)
mask_L2 = (vals_L2 > 1 - eps) & (vals_L2 < 1 + eps)
slice_L2 = pts3[mask_L2]

# -------- ADD INITIAL TRACES --------
fig.add_trace(
    go.Scatter3d(
        x=slice_L1[:,0], y=slice_L1[:,1], z=slice_L1[:,2],
        mode='markers', marker=dict(size=2)
    ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter3d(
        x=slice_L2[:,0], y=slice_L2[:,1], z=slice_L2[:,2],
        mode='markers', marker=dict(size=2)
    ),
    row=1, col=2
)

# -------- BUILD FRAMES FOR SLIDER --------
for idx, w0 in enumerate(values):

    pts4 = np.column_stack([pts3, np.full(N, w0)])

    # L1
    vals_L1 = np.apply_along_axis(norm1, 1, pts4)
    mask_L1 = (vals_L1 > 1 - eps) & (vals_L1 < 1 + eps)
    slice_L1 = pts3[mask_L1]

    # L2
    vals_L2 = np.apply_along_axis(norm2, 1, pts4)
    mask_L2 = (vals_L2 > 1 - eps) & (vals_L2 < 1 + eps)
    slice_L2 = pts3[mask_L2]

    frame = go.Frame(
        data=[
            go.Scatter3d(
                x=slice_L1[:,0], y=slice_L1[:,1], z=slice_L1[:,2],
                mode='markers', marker=dict(size=2)
            ),
            go.Scatter3d(
                x=slice_L2[:,0], y=slice_L2[:,1], z=slice_L2[:,2],
                mode='markers', marker=dict(size=2)
            )
        ],
        name=str(idx)
    )

    frames.append(frame)

    steps.append(
        dict(
            method="animate",
            args=[
                [str(idx)],
                dict(
                    frame=dict(duration=0, redraw=True),
                    mode="immediate",
                    transition=dict(duration=0)
                )
            ],
            label=f"{w0:.2f}"
        )
    )


# -------- APPLY LAYOUT SETTINGS --------
fig.update_layout(
    sliders=[
        dict(
            steps=steps,
            currentvalue=dict(prefix="w = "),
            pad=dict(t=50),
            transition=dict(duration=0)
        )
    ],
    scene=dict(
        xaxis=dict(range=[-1, 1]),
        yaxis=dict(range=[-1, 1]),
        zaxis=dict(range=[-1, 1]),
        aspectmode='cube'
    ),
    scene2=dict(
        xaxis=dict(range=[-1, 1]),
        yaxis=dict(range=[-1, 1]),
        zaxis=dict(range=[-1, 1]),
        aspectmode='cube'
    ),
    frames=frames
)

fig.show()
