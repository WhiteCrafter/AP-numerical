#!/usr/bin/env python3

# Option C — super simple, student-style script
# No classes, no configs, no CSVs, no saving images.
# Just compute a few things and pop up some plots.

import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# 1D function and its exact derivative (at scalars)
def f1d(x):
    return math.exp(x) * math.sin(x)


def df1d_exact(x):
    # d/dx [e^x sin x] = e^x (sin x + cos x)
    return math.exp(x) * (math.sin(x) + math.cos(x))


# 2D function (scalar) and numpy version for grids
def f2d(x, y):
    return x * x * y + math.sin(x * y)


def f2d_np(X, Y):
    return X**2 * Y + np.sin(X * Y)


def main():
    # Point and steps
    x0 = 0.5
    y0 = 0.4
    hs = np.logspace(-1, -6, 8)  # a few h values

    # 1D finite differences at x0
    df_exact = df1d_exact(x0)
    err_forward = []
    err_central = []
    for h in hs:
        df_f = (f1d(x0 + h) - f1d(x0)) / h
        df_c = (f1d(x0 + h) - f1d(x0 - h)) / (2 * h)
        err_forward.append(abs(df_f - df_exact))
        err_central.append(abs(df_c - df_exact))

    # Plot 1D errors
    plt.figure()
    plt.loglog(hs, err_forward, 'o-', label='forward')
    plt.loglog(hs, err_central, 'o-', label='central')
    plt.gca().invert_xaxis()
    plt.xlabel('h')
    plt.ylabel('abs error')
    plt.title("1D derivative error vs h")
    plt.grid(True, which='both', ls=':')
    plt.legend()

    # 1D curve + tangent line + a secant
    xs = np.linspace(x0 - 2.0, x0 + 2.0, 400)
    ys = np.exp(xs) * np.sin(xs)  # numpy version for vector inputs
    y0 = f1d(x0)
    m = df_exact
    y_tan = y0 + m * (xs - x0)
    h_show = 0.2
    sec_slope = (f1d(x0 + h_show) - y0) / h_show
    y_sec = y0 + sec_slope * (xs - x0)

    plt.figure()
    plt.plot(xs, ys, label='f(x)')
    plt.plot(xs, y_tan, '--', label='tangent')
    plt.plot(xs, y_sec, ':', label=f'secant h={h_show}')
    plt.scatter([x0, x0 + h_show], [y0, f1d(x0 + h_show)], color='k', s=25)
    plt.scatter([x0], [y0], color='r', s=30, zorder=5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('1D curve, tangent, one secant')
    plt.grid(True, ls=':')
    plt.legend()

    # 2D: simple central-diff gradient at (x0,y0)
    h2 = 1e-4
    fx = (f2d(x0 + h2, y0) - f2d(x0 - h2, y0)) / (2 * h2)
    fy = (f2d(x0, y0 + h2) - f2d(x0, y0 - h2)) / (2 * h2)

    # 2D contours + small arrow for gradient
    X = np.linspace(x0 - 1.0, x0 + 1.0, 120)
    Y = np.linspace(y0 - 1.0, y0 + 1.0, 120)
    XX, YY = np.meshgrid(X, Y)
    ZZ = f2d_np(XX, YY)
    plt.figure()
    cs = plt.contour(XX, YY, ZZ, levels=20)
    plt.clabel(cs, inline=True, fontsize=7)
    plt.scatter([x0], [y0], color='k', s=25)
    n = np.hypot(fx, fy)
    if n > 0:
        u, v = fx / n, fy / n
        s = 0.5
        plt.arrow(x0, y0, u * s, v * s, head_width=0.05, color='r', length_includes_head=True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('2D contours and gradient (central diff)')
    plt.grid(True, ls=':')

    # Simple 3D surface view around (x0, y0)
    fig = plt.figure()
    ax3d = fig.add_subplot(111, projection='3d')
    ax3d.plot_surface(XX, YY, ZZ, cmap='viridis', alpha=0.9, linewidth=0)
    z0 = f2d(x0, y0)
    ax3d.scatter([x0], [y0], [z0], color='r', s=20)
    ax3d.set_title('3D surface z=f(x,y)')
    ax3d.set_xlabel('x')
    ax3d.set_ylabel('y')
    ax3d.set_zlabel('z')

    plt.show()


if __name__ == '__main__':
    main()
