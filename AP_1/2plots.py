from ipywidgets import FloatSlider
from IPython.display import display
import plotly.graph_objects as go
import numpy as np

fig = go.FigureWidget(data=[go.Scatter3d(mode='markers')])
display(fig)

slider = FloatSlider(min=-1, max=1, step=0.01)
display(slider)

def update(change):
    w0 = change["new"]
    # update the trace here
    fig.data[0].x = np.random.randn(100)
    fig.data[0].y = np.random.randn(100)
    fig.data[0].z = np.random.randn(100)

slider.observe(update, names="value")