#!/usr/bin/env python3

"""
AP3 Option A: Single window with multiple subplots (1D/2D errors and geometry)

Features
- 1D: tangent and normal lines; FD: forward/backward/central/five-point
- 2D: tangent plane and normal vector; FD: central/five-point for fx, fy
- Error plots shown interactively; CSV tables optional to AP_3/output

Usage
- Run defaults: python AP_3/main_option_a.py
- To customize, edit Config in this file to set callables and parameters.

Notes
- Pure-numpy callables for f1d/f2d; optional exact derivatives can be provided as callables.
- Plots are displayed in one window using subplots; CSV tables optional. Works without pandas.
"""

from __future__ import annotations

import math
import os
import importlib
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple, Optional


def _require(package: str, import_name: str | None = None):
    try:
        return importlib.import_module(import_name or package)
    except ImportError as exc:
        raise SystemExit(
            f"Missing dependency '{import_name or package}'. Install with: pip install {package}\n"
            f"Original error: {exc}"
        )


np = _require("numpy")
plt = _require("matplotlib", "matplotlib.pyplot")


# --------------------- Default functions and optional exact derivatives ---------------------

def default_f1d(x: np.ndarray) -> np.ndarray:
    return np.exp(x) * np.sin(x)


def default_df1d(x: float) -> float:
    # d/dx [e^x sin x] = e^x (sin x + cos x)
    return float(np.exp(x) * (np.sin(x) + np.cos(x)))


def default_f2d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x**2 * y + np.sin(x * y)


def default_grad2d(x: float, y: float) -> Tuple[float, float]:
    # f(x,y) = x^2 y + sin(xy)
    # fx = 2xy + y cos(xy)
    # fy = x^2 + x cos(xy)
    xy = x * y
    c = float(np.cos(xy))
    fx = 2.0 * x * y + y * c
    fy = x * x + x * c
    return float(fx), float(fy)


# --------------------- Finite differences ---------------------


def fd_forward_1d(f: Callable[[np.ndarray], np.ndarray], x0: float, h: float) -> float:
    return float((f(x0 + h) - f(x0)) / h)


def fd_backward_1d(f: Callable[[np.ndarray], np.ndarray], x0: float, h: float) -> float:
    return float((f(x0) - f(x0 - h)) / h)


def fd_central_1d(f: Callable[[np.ndarray], np.ndarray], x0: float, h: float) -> float:
    return float((f(x0 + h) - f(x0 - h)) / (2.0 * h))


def fd_five_point_1d(f: Callable[[np.ndarray], np.ndarray], x0: float, h: float) -> float:
    # 4th-order accurate central five-point stencil
    return float((-f(x0 + 2 * h) + 8 * f(x0 + h) - 8 * f(x0 - h) + f(x0 - 2 * h)) / (12.0 * h))


def fd_central_2d_x(f: Callable[[np.ndarray, np.ndarray], np.ndarray], x0: float, y0: float, h: float) -> float:
    return float((f(x0 + h, y0) - f(x0 - h, y0)) / (2.0 * h))


def fd_central_2d_y(f: Callable[[np.ndarray, np.ndarray], np.ndarray], x0: float, y0: float, h: float) -> float:
    return float((f(x0, y0 + h) - f(x0, y0 - h)) / (2.0 * h))


def fd_five_point_2d_x(f: Callable[[np.ndarray, np.ndarray], np.ndarray], x0: float, y0: float, h: float) -> float:
    return float((-f(x0 + 2 * h, y0) + 8 * f(x0 + h, y0) - 8 * f(x0 - h, y0) + f(x0 - 2 * h, y0)) / (12.0 * h))


def fd_five_point_2d_y(f: Callable[[np.ndarray, np.ndarray], np.ndarray], x0: float, y0: float, h: float) -> float:
    return float((-f(x0, y0 + 2 * h) + 8 * f(x0, y0 + h) - 8 * f(x0, y0 - h) + f(x0, y0 - 2 * h)) / (12.0 * h))


# --------------------- Geometry: tangent/normal ---------------------


def tangent_normal_1d(f: Callable[[np.ndarray], np.ndarray], x0: float, df_exact: float) -> Dict[str, Tuple[float, float]]:
    y0 = float(f(x0))
    m = df_exact
    tvec = np.array([1.0, m], dtype=float)
    tvec = tvec / (np.linalg.norm(tvec) or 1.0)
    nvec = np.array([-m, 1.0], dtype=float)
    nvec = nvec / (np.linalg.norm(nvec) or 1.0)
    return {
        "point": (x0, y0),
        "tangent_dir": (float(tvec[0]), float(tvec[1])),
        "normal_dir": (float(nvec[0]), float(nvec[1])),
        "slope": (m, -1.0 / m if m != 0 else math.inf),
    }


def tangent_plane_normal_2d(
    f: Callable[[np.ndarray, np.ndarray], np.ndarray], x0: float, y0: float, fx: float, fy: float
) -> Dict[str, Tuple[float, float, float]]:
    z0 = float(f(x0, y0))
    n = np.array([-fx, -fy, 1.0], dtype=float)
    n = n / (np.linalg.norm(n) or 1.0)
    return {
        "point": (x0, y0, z0),
        "normal_dir": (float(n[0]), float(n[1]), float(n[2])),
        "plane_coeffs": (z0, fx, fy),
    }


# --------------------- Reporting helpers ---------------------


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def print_table(headers: List[str], rows: List[List[float]]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(f"{cell:g}"))

    def fmt_row(items: Iterable):
        return " | ".join(f"{str(x):>{w}}" for x, w in zip(items, widths))

    print(fmt_row(headers))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(fmt_row([f"{v:g}" for v in row]))


def save_csv(path: str, headers: List[str], rows: List[List[float]]) -> None:
    try:
        import csv

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
    except Exception as e:
        print(f"[warn] Could not save CSV to {path}: {e}")


# --------------------- Drawing (axes-based, no show/close) ---------------------


def draw_errors(ax, hs: np.ndarray, series: Dict[str, np.ndarray], title: str) -> None:
    for label, errs in series.items():
        ax.loglog(hs, errs, marker="o", label=label)
    ax.invert_xaxis()
    ax.set_xlabel("h")
    ax.set_ylabel("absolute error")
    ax.set_title(title)
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend(fontsize=8)


def draw_1d_tangent_and_secants(ax, f: Callable[[np.ndarray], np.ndarray], x0: float, df_exact: float, hs_show: List[float]) -> None:
    y0 = float(f(x0))
    width = max(1.0, 20.0 * max(hs_show))
    xs = np.linspace(x0 - width, x0 + width, 400)
    ys = f(xs)
    ax.plot(xs, ys, label="f(x)")
    y_tan = y0 + df_exact * (xs - x0)
    ax.plot(xs, y_tan, '--', label="tangent (exact)")
    if df_exact != 0 and np.isfinite(df_exact):
        m_n = -1.0 / df_exact
        y_norm = y0 + m_n * (xs - x0)
        ax.plot(xs, y_norm, ':', label="normal line")
    else:
        ax.axvline(x0, color='gray', linestyle=':', label='normal (vertical)')
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(hs_show)))
    for c, h in zip(colors, hs_show):
        slope = (float(f(x0 + h)) - y0) / h
        y_sec = y0 + slope * (xs - x0)
        ax.plot(xs, y_sec, color=c, alpha=0.9, label=f"secant h={h:g}")
        ax.scatter([x0, x0 + h], [y0, float(f(x0 + h))], color=c, s=20)
    ax.scatter([x0], [y0], color='k', s=25, zorder=5, label='(x0,f(x0))')
    ax.set_title("1D: function with tangent and secants")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, ls=':', alpha=0.6)
    ax.legend(loc='best', fontsize=7)


def draw_2d_contour_with_gradient(ax, f: Callable[[np.ndarray, np.ndarray], np.ndarray], x0: float, y0: float, fx: float, fy: float) -> None:
    width = 1.0
    xs = np.linspace(x0 - width, x0 + width, 120)
    ys = np.linspace(y0 - width, y0 + width, 120)
    X, Y = np.meshgrid(xs, ys)
    Z = f(X, Y)
    cs = ax.contour(X, Y, Z, levels=20, cmap='viridis')
    ax.clabel(cs, inline=True, fontsize=7, fmt='%.2f')
    ax.scatter([x0], [y0], color='k', s=25)
    g = np.array([fx, fy])
    if np.linalg.norm(g) > 0:
        g_unit = g / np.linalg.norm(g)
        scale = width * 0.5
        ax.arrow(x0, y0, g_unit[0] * scale, g_unit[1] * scale, head_width=0.05, color='red', length_includes_head=True)
    ax.set_title("2D: contours with gradient at (x0,y0)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, ls=':', alpha=0.5)


def draw_3d_surface_and_plane(ax, f: Callable[[np.ndarray, np.ndarray], np.ndarray], x0: float, y0: float, fx: float, fy: float) -> None:
    width = 1.0
    xs = np.linspace(x0 - width, x0 + width, 60)
    ys = np.linspace(y0 - width, y0 + width, 60)
    X, Y = np.meshgrid(xs, ys)
    Z = f(X, Y)
    z0 = float(f(x0, y0))
    Zp = z0 + fx * (X - x0) + fy * (Y - y0)
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, linewidth=0)
    ax.plot_surface(X, Y, Zp, cmap='autumn', alpha=0.35, linewidth=0)
    ax.scatter([x0], [y0], [z0], color='k', s=20)
    ax.set_title('Surface and tangent plane')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


# --------------------- Main analysis ---------------------


@dataclass
class Config:
    # User-supplied callables
    f1d: Callable[[np.ndarray], np.ndarray] = default_f1d
    # Optional exact derivative at a point; if None, use high-accuracy FD as reference
    df1d: Optional[Callable[[float], float]] = default_df1d
    x0: float = 0.5
    f2d: Callable[[np.ndarray, np.ndarray], np.ndarray] = default_f2d
    # Optional exact gradient at a point; if None, use high-accuracy FD as reference
    grad2d: Optional[Callable[[float, float], Tuple[float, float]]] = default_grad2d
    y0: float = 0.4
    outdir: str = os.path.join("AP_3", "output")
    save_csvs: bool = True
    h_min: float = 1e-8
    h_max: float = 1e-1
    num_h: int = 12
    show_h: tuple[float, ...] = ()


def run(config: Config) -> None:
    ensure_dir(config.outdir)

    # 1D setup
    f1d = config.f1d
    if config.df1d is not None:
        df_exact = float(config.df1d(config.x0))
    else:
        h_ref = 1e-6
        df_exact = fd_five_point_1d(f1d, config.x0, h_ref)

    # 2D setup
    f2d = config.f2d
    if config.grad2d is not None:
        fx_exact, fy_exact = config.grad2d(config.x0, config.y0)
    else:
        h_ref = 1e-6
        fx_exact = fd_five_point_2d_x(f2d, config.x0, config.y0, h_ref)
        fy_exact = fd_five_point_2d_y(f2d, config.x0, config.y0, h_ref)

    # Geometry info (printed)
    g1d = tangent_normal_1d(f1d, config.x0, df_exact)
    g2d = tangent_plane_normal_2d(f2d, config.x0, config.y0, fx_exact, fy_exact)

    print("-- 1D function --")
    name_1d = getattr(f1d, "__name__", "callable")
    print(f"f1d = {name_1d}")
    print(f"x0 = {config.x0:g}")
    print(f"f(x0) = {f1d(config.x0):.10g}")
    print(f"f'(x0) exact = {df_exact:.10g}")
    print(f"tangent dir = {g1d['tangent_dir']}, normal dir = {g1d['normal_dir']}")
    print(f"tangent line: y = f(x0) + f'(x0)*(x - x0)")
    if math.isfinite(g1d["slope"][1]):
        print(f"normal line slope = {g1d['slope'][1]:.10g}")
    else:
        print("normal line is vertical (slope = inf)")

    print("\n-- 2D function --")
    name_2d = getattr(f2d, "__name__", "callable")
    print(f"f2d = {name_2d}")
    print(f"(x0, y0) = ({config.x0:g}, {config.y0:g})")
    print(f"f(x0,y0) = {f2d(config.x0, config.y0):.10g}")
    print(f"fx exact = {fx_exact:.10g}, fy exact = {fy_exact:.10g}")
    print(f"normal dir = {g2d['normal_dir']}")
    print("tangent plane: z = z0 + fx*(x-x0) + fy*(y-y0)")

    # FD sweep
    hs = np.logspace(math.log10(config.h_max), math.log10(config.h_min), num=config.num_h)

    # 1D errors
    schemes_1d = {
        "forward O(h)": lambda h: fd_forward_1d(f1d, config.x0, h),
        "backward O(h)": lambda h: fd_backward_1d(f1d, config.x0, h),
        "central O(h^2)": lambda h: fd_central_1d(f1d, config.x0, h),
        "five-point O(h^4)": lambda h: fd_five_point_1d(f1d, config.x0, h),
    }
    err_table_1d: List[List[float]] = []
    series_1d: Dict[str, np.ndarray] = {name: np.zeros_like(hs) for name in schemes_1d}
    for i, h in enumerate(hs):
        row = [float(h)]
        for name, approx in schemes_1d.items():
            val = approx(float(h))
            err = abs(val - df_exact)
            series_1d[name][i] = err
            row.append(err)
        err_table_1d.append(row)

    print("\n1D absolute error vs h")
    print_table(["h"] + list(schemes_1d.keys()), err_table_1d)
    if config.save_csvs:
        save_csv(os.path.join(config.outdir, "errors_1d.csv"), ["h"] + list(schemes_1d.keys()), err_table_1d)

    # 2D errors for fx and fy
    def sweep_2d(partial_fun, exact_val, label_prefix: str) -> Tuple[List[List[float]], Dict[str, np.ndarray]]:
        schemes = {
            "central O(h^2)": partial_fun["central"],
            "five-point O(h^4)": partial_fun["five"],
        }
        table: List[List[float]] = []
        series: Dict[str, np.ndarray] = {name: np.zeros_like(hs) for name in schemes}
        for i, h in enumerate(hs):
            row = [float(h)]
            for name, approx in schemes.items():
                val = approx(float(h))
                err = abs(val - exact_val)
                series[name][i] = err
                row.append(err)
            table.append(row)
        print(f"\n2D absolute error vs h for {label_prefix}")
        print_table(["h"] + list(schemes.keys()), table)
        if config.save_csvs:
            save_csv(os.path.join(config.outdir, f"errors_2d_{label_prefix}.csv"), ["h"] + list(schemes.keys()), table)
        return table, series

    _, series_fx = sweep_2d(
        {
            "central": lambda h: fd_central_2d_x(f2d, config.x0, config.y0, h),
            "five": lambda h: fd_five_point_2d_x(f2d, config.x0, config.y0, h),
        },
        fx_exact,
        "fx",
    )
    _, series_fy = sweep_2d(
        {
            "central": lambda h: fd_central_2d_y(f2d, config.x0, config.y0, h),
            "five": lambda h: fd_five_point_2d_y(f2d, config.x0, config.y0, h),
        },
        fy_exact,
        "fy",
    )

    # Illustrative subplots
    hs_show: List[float] = list(config.show_h) if config.show_h else [float(hs[0]), float(hs[len(hs)//2]), float(hs[-1])]

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 3)
    ax_err1d = fig.add_subplot(gs[0, 0])
    ax_err_fx = fig.add_subplot(gs[0, 1])
    ax_err_fy = fig.add_subplot(gs[0, 2])
    ax_tan = fig.add_subplot(gs[1, 0])
    ax_contour = fig.add_subplot(gs[1, 1])
    ax3d = fig.add_subplot(gs[1, 2], projection='3d')

    draw_errors(ax_err1d, hs, series_1d, "1D derivative error vs h")
    draw_errors(ax_err_fx, hs, series_fx, "2D fx error vs h")
    draw_errors(ax_err_fy, hs, series_fy, "2D fy error vs h")
    draw_1d_tangent_and_secants(ax_tan, f1d, config.x0, df_exact, hs_show)
    draw_2d_contour_with_gradient(ax_contour, f2d, config.x0, config.y0, fx_exact, fy_exact)
    draw_3d_surface_and_plane(ax3d, f2d, config.x0, config.y0, fx_exact, fy_exact)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    cfg = Config(
        # Customize here if desired
        # f1d=default_f1d, df1d=default_df1d,
        # f2d=default_f2d, grad2d=default_grad2d,
        # x0=0.5, y0=0.4,
        # show_h=(0.1, 0.01, 0.001),
    )
    run(cfg)

