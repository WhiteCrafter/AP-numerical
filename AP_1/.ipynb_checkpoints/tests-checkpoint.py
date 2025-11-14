# %%
import numpy as np
import plotly.graph_objects as go

# %%
# -------- CHOOSE NORM HERE --------
def norm1(v):    return np.abs(v).sum()

# %%
def norm2(v):    return np.sqrt((v**2).sum())

# %%
def norm_inf(v):    return np.max(np.abs(v))

# %%
def induced1(A):    return np.abs(A).sum(axis=0).max()

# %%
def induced2(A):
    return np.sqrt(np.linalg.eigvalsh(A.T @ A).max())


# %%

# %%
vector1 = np.array([2, 1, 3 , 4])
matrix1 = np.array([[2, 1],[3 , 4]])

# %%
# -------- SAMPLE POINTS --------
res = 61

# %%
xs, ys, zs = np.mgrid[
    -1:1:complex(res),
    -1:1:complex(res),
    -1:1:complex(res)
]

# %%
def distance(vec1, vec2, norm):
    return norm(vec1 - vec2)


# %%

# %%
def ballSliceVector(pts4, vec,  norm):
    vals = np.apply_along_axis(norm, 1, (pts4-vec))  #maps points4d(pts4) to its distance from vector
    mask = (vals >0.96) & (vals <= 1) #boolean array that tells which of the points are in our unit ball

    pts_slice = pts4[:,:3][mask]
    return pts_slice[:,0], pts_slice[:,1], pts_slice[:,2]

# %%
def ballSliceMatrix(pts4, vec,  norm):
    diff = (pts4 - vec).reshape(-1, 2, 2)
    vals = np.array([norm(M) for M in diff])      #maps points4d(pts4) to its distance from vector
    mask = (vals >0.96) & (vals <= 1) #boolean array that tells which of the points are in our unit ball

    pts_slice = pts4[:,:3][mask]
    return (pts_slice[:,0], pts_slice[:,1], pts_slice[:,2])

# %%
def unitBall(vec ,norm,  matrix = False):
    frames = []
    steps = []


    if (matrix): vec = vec.reshape(4)
    pts3 = np.vstack((xs.ravel(), ys.ravel(), zs.ravel())).T + vec[:3]
    N = pts3.shape[0] 


    values = np.linspace(-1+ vec[3], 1 + vec[3], 21)





    for idx, w0 in enumerate(values):

        pts4 = np.column_stack([pts3, np.full(N, w0)]) #add 4th dimmention

        if (matrix) : x,y,z = ballSliceMatrix(pts4, vec, norm)
        else:  x,y,z = ballSliceVector(pts4,vec, norm)

        frame = go.Frame(
            data=[
                go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='markers',
                    marker=dict(size=2)
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

    # -------- INITIAL FRAME --------
    w0 = values[0]
    pts4 = np.column_stack([pts3, np.full(N, values[0])])
    if (matrix) : x,y,z = ballSliceMatrix(pts4, vec, norm)
    else:  x,y,z = ballSliceVector(pts4,vec, norm)

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(size=2)
            )
        ],
        layout=go.Layout(
            sliders=[
                dict(
                    steps=steps,
                    currentvalue=dict(prefix="w = "),
                    pad=dict(t=50),
                    transition=dict(duration=0),
                )
            ],
            scene=dict(
                xaxis=dict(range=[-1 + vec[0], 1 + vec[0]]),
                yaxis=dict(range=[-1 + vec[1], 1 + vec[1]]),
                zaxis=dict(range=[-1 + vec[2], 1 + vec[2]]),
                aspectmode='cube'
            )
        ),
        frames=frames
    )

    fig.show()

# %%
unitBall(matrix1, induced1, matrix = True)
