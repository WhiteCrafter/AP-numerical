import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# 1D FUNCTION + EXACT DERIVATIVE
# ------------------------------

def f1(x):
    return np.exp(x) * np.sin(x)

def df1_exact(x):
    return np.exp(x) * (np.sin(x) + np.cos(x))

# Simple finite differences
def fd_forward(f, x, h):
    return (f(x+h) - f(x)) / h

def fd_backward(f, x, h):
    return (f(x) - f(x-h)) / h

def fd_central(f, x, h):
    return (f(x+h) - f(x-h)) / (2*h)

def fd_five_point(f, x, h):
    return (-f(x+2*h) + 8*f(x+h) - 8*f(x-h) + f(x-2*h)) / (12*h)


# ------------------------------
# 2D FUNCTION + EXACT GRADIENT
# ------------------------------

def f2(x, y):
    return x*x*y + np.sin(x*y)

def grad2_exact(x, y):
    xy = x*y
    fx = 2*x*y + y*np.cos(xy)
    fy = x*x + x*np.cos(xy)
    return fx, fy

def fd_central_x(f, x, y, h):
    return (f(x+h, y) - f(x-h, y)) / (2*h)

def fd_central_y(f, x, y, h):
    return (f(x, y+h) - f(x, y-h)) / (2*h)

def fd_five_x(f, x, y, h):
    return (-f(x+2*h, y)+8*f(x+h,y)-8*f(x-h,y)+f(x-2*h,y))/(12*h)

def fd_five_y(f, x, y, h):
    return (-f(x, y+2*h)+8*f(x,y+h)-8*f(x,y-h)+f(x,y-2*h))/(12*h)


# ------------------------------
# ERROR ANALYSIS
# ------------------------------

def derivative_error_plots():
    x0 = 0.5
    hs = np.logspace(-1, -8, 15)
    exact = df1_exact(x0)

    errors = {
        "forward": [],
        "backward": [],
        "central": [],
        "five-point": []
    }

    for h in hs:
        errors["forward"].append(abs(fd_forward(f1, x0, h) - exact))
        errors["backward"].append(abs(fd_backward(f1, x0, h) - exact))
        errors["central"].append(abs(fd_central(f1, x0, h) - exact))
        errors["five-point"].append(abs(fd_five_point(f1, x0, h) - exact))

    plt.figure()
    for name, vals in errors.items():
        plt.loglog(hs, vals, marker="o", label=name)
    plt.gca().invert_xaxis()
    plt.xlabel("h")
    plt.ylabel("absolute error")
    plt.title("1D Finite Difference Error")
    plt.grid(True, which="both", ls=":")
    plt.legend()
    plt.show()


# ------------------------------
# 1D TANGENT + SECANTS PLOT
# ------------------------------

def tangent_and_secants():
    x0 = 0.5
    y0 = f1(x0)
    m = df1_exact(x0)

    xs = np.linspace(x0 - 2, x0 + 2, 400)
    plt.figure()
    plt.plot(xs, f1(xs), label="f(x)")

    # exact tangent
    plt.plot(xs, y0 + m*(xs - x0), '--', label="tangent (exact)")

    # a few secants
    for h in [0.5, 0.1, 0.01]:
        slope = fd_forward(f1, x0, h)
        plt.plot(xs, y0 + slope*(xs-x0), label=f"secant h={h}")

    plt.scatter([x0], [y0], color="black")
    plt.legend()
    plt.title("Tangent and Secants")
    plt.grid(True, ls=":")
    plt.show()


# ------------------------------
# 2D CONTOUR + GRADIENT ARROW
# ------------------------------

def contour_and_gradient():
    x0, y0 = 0.5, 0.4
    fx, fy = grad2_exact(x0, y0)

    xs = np.linspace(x0-1, x0+1, 100)
    ys = np.linspace(y0-1, y0+1, 100)
    X, Y = np.meshgrid(xs, ys)
    Z = f2(X, Y)

    plt.figure()
    cs = plt.contour(X, Y, Z, levels=20)
    plt.clabel(cs)
    plt.scatter([x0], [y0], c='red')

    g = np.array([fx, fy])
    g = g / np.linalg.norm(g)
    plt.arrow(x0, y0, g[0]*0.4, g[1]*0.4,
              head_width=0.05, color="red")

    plt.title("2D Contours + Gradient Direction")
    plt.grid(True, ls=":")
    plt.show()


# ------------------------------
# 3D SURFACE + TANGENT PLANE
# ------------------------------

def surface_and_plane():
    from mpl_toolkits.mplot3d import Axes3D  # required import

    x0, y0 = 0.5, 0.4
    fx, fy = grad2_exact(x0, y0)
    z0 = f2(x0, y0)

    xs = np.linspace(x0-1, x0+1, 40)
    ys = np.linspace(y0-1, y0+1, 40)
    X, Y = np.meshgrid(xs, ys)
    Z = f2(X, Y)
    Zp = z0 + fx*(X-x0) + fy*(Y-y0)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.7)
    ax.plot_surface(X, Y, Zp, cmap="autumn", alpha=0.4)
    ax.scatter([x0], [y0], [z0], c='black')
    ax.set_title("Surface and Tangent Plane")
    plt.show()


# ------------------------------
# MAIN
# ------------------------------

if __name__ == "__main__":
    derivative_error_plots()
    tangent_and_secants()
    contour_and_gradient()
    surface_and_plane()
