#!/usr/bin/env python3
"""
AP3: Tangents, normals, and finite-difference accuracy (1D and 2D)

Features
- 1D: exact derivative, tangent and normal lines; FD: forward/backward/central/five-point
- 2D: exact gradient, tangent plane and normal vector; FD: central/five-point for fx, fy
- Error vs step size plots and tables saved to AP_3/output

Usage (examples)
- Default demo (recommended):
    python AP_3/main.py

- Custom functions and point:
    python AP_3/main.py \
        --f1d "exp(x)*sin(x)" --x0 0.5 \
        --f2d "x**2*y + sin(x*y)" --y0 0.4

Notes
- Symbolic differentiation via sympy; numeric via numpy.
- Plots saved as PNG; CSV tables optional. Works without pandas.
"""

from __future__ import annotations

import argparse
import math
import os
import importlib
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple


def _require(package: str, import_name: str | None = None):
    try:
        return importlib.import_module(import_name or package)
    except ImportError as exc:
        raise SystemExit(
            f"Missing dependency '{import_name or package}'. Install with: pip install {package}\n"
            f"Original error: {exc}"
        )


np = _require("numpy")
sp = _require("sympy")
plt = _require("matplotlib", "matplotlib.pyplot")


# --------------------- Parsing and exact derivatives ---------------------


def parse_function_1d(expr: str) -> Tuple[sp.Symbol, sp.Expr, Callable[[np.ndarray], np.ndarray]]:
    x = sp.symbols("x", real=True)
    # Ensure the same symbol instance is used inside the expression
    f_expr = sp.sympify(expr, locals={"x": x})
    f = sp.lambdify(x, f_expr, modules=["numpy", {"sin": np.sin, "cos": np.cos, "exp": np.exp}])
    return x, f_expr, f  # type: ignore


def parse_function_2d(expr: str) -> Tuple[Tuple[sp.Symbol, sp.Symbol], sp.Expr, Callable[[np.ndarray, np.ndarray], np.ndarray]]:
    x, y = sp.symbols("x y", real=True)
    # Ensure the same symbol instances are used inside the expression
    f_expr = sp.sympify(expr, locals={"x": x, "y": y})
    f = sp.lambdify((x, y), f_expr, modules=["numpy", {"sin": np.sin, "cos": np.cos, "exp": np.exp}])
    return (x, y), f_expr, f  # type: ignore


def exact_derivative_1d(f_expr: sp.Expr, x: sp.Symbol, x0: float) -> float:
    dfx = sp.diff(f_expr, x)
    return float(dfx.subs({x: x0}))


def exact_gradient_2d(f_expr: sp.Expr, x: sp.Symbol, y: sp.Symbol, x0: float, y0: float) -> Tuple[float, float]:
    dfx = sp.diff(f_expr, x)
    dfy = sp.diff(f_expr, y)
    fx = float(dfx.subs({x: x0, y: y0}))
    fy = float(dfy.subs({x: x0, y: y0}))
    return fx, fy


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
    # Tangent line: y = y0 + m (x - x0)
    # Normal line slope is -1/m if m != 0; represent as vector to avoid inf slope when m=0
    # Return direction vectors for tangent and normal, normalized
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
    # Plane: z = z0 + fx*(x - x0) + fy*(y - y0)
    # Normal vector to z = f(x,y) is [-fx, -fy, 1]
    n = np.array([-fx, -fy, 1.0], dtype=float)
    n = n / (np.linalg.norm(n) or 1.0)
    return {
        "point": (x0, y0, z0),
        "normal_dir": (float(n[0]), float(n[1]), float(n[2])),
        "plane_coeffs": (z0, fx, fy),  # z = z0 + fx*(x-x0) + fy*(y-y0)
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


def plot_errors(hs: np.ndarray, series: Dict[str, np.ndarray], title: str, out_path: str) -> None:
    plt.figure(figsize=(6, 4))
    for label, errs in series.items():
        plt.loglog(hs, errs, marker="o", label=label)
    plt.gca().invert_xaxis()  # decreasing h to the right
    plt.xlabel("h")
    plt.ylabel("absolute error")
    plt.title(title)
    plt.grid(True, which="both", ls=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_1d_tangent_and_secants(
    f: Callable[[np.ndarray], np.ndarray],
    x0: float,
    df_exact: float,
    hs_show: List[float],
    out_path: str,
) -> None:
    y0 = float(f(x0))
    width = max(1.0, 20.0 * max(hs_show))
    xs = np.linspace(x0 - width, x0 + width, 400)
    ys = f(xs)

    plt.figure(figsize=(7, 4.5))
    plt.plot(xs, ys, label="f(x)")
    # Exact tangent line
    y_tan = y0 + df_exact * (xs - x0)
    plt.plot(xs, y_tan, '--', label="tangent (exact)")
    # Normal line
    if df_exact != 0 and np.isfinite(df_exact):
        m_n = -1.0 / df_exact
        y_norm = y0 + m_n * (xs - x0)
        plt.plot(xs, y_norm, ':', label="normal line")
    else:
        plt.axvline(x0, color='gray', linestyle=':', label='normal (vertical)')

    # Secant lines using forward differences for selected h
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(hs_show)))
    for c, h in zip(colors, hs_show):
        slope = (float(f(x0 + h)) - y0) / h
        y_sec = y0 + slope * (xs - x0)
        plt.plot(xs, y_sec, color=c, alpha=0.9, label=f"secant h={h:g}")
        plt.scatter([x0, x0 + h], [y0, float(f(x0 + h))], color=c, s=25)

    plt.scatter([x0], [y0], color='k', s=35, zorder=5, label='point (x0,f(x0))')
    plt.title("1D: function with tangent and secants")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, ls=':', alpha=0.6)
    plt.legend(loc='best', fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_2d_contour_with_gradient(
    f: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x0: float,
    y0: float,
    fx: float,
    fy: float,
    out_path: str,
) -> None:
    width = 1.0
    xs = np.linspace(x0 - width, x0 + width, 120)
    ys = np.linspace(y0 - width, y0 + width, 120)
    X, Y = np.meshgrid(xs, ys)
    Z = f(X, Y)

    plt.figure(figsize=(6, 5))
    cs = plt.contour(X, Y, Z, levels=20, cmap='viridis')
    plt.clabel(cs, inline=True, fontsize=7, fmt='%.2f')
    plt.scatter([x0], [y0], color='k', s=30)
    g = np.array([fx, fy])
    if np.linalg.norm(g) > 0:
        g_unit = g / np.linalg.norm(g)
        scale = width * 0.5
        plt.arrow(x0, y0, g_unit[0] * scale, g_unit[1] * scale, head_width=0.05, color='red', length_includes_head=True)
    plt.title("2D: contours with gradient vector at (x0,y0)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, ls=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_3d_surface_and_plane(
    f: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x0: float,
    y0: float,
    fx: float,
    fy: float,
    out_path: str,
) -> None:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    width = 1.0
    xs = np.linspace(x0 - width, x0 + width, 60)
    ys = np.linspace(y0 - width, y0 + width, 60)
    X, Y = np.meshgrid(xs, ys)
    Z = f(X, Y)
    z0 = float(f(x0, y0))
    Zp = z0 + fx * (X - x0) + fy * (Y - y0)

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, linewidth=0)
    ax.plot_surface(X, Y, Zp, cmap='autumn', alpha=0.35, linewidth=0)
    ax.scatter([x0], [y0], [z0], color='k', s=20)
    ax.set_title('Surface and tangent plane')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# --------------------- Main analysis ---------------------


@dataclass
class Config:
    f1d_expr: str = "exp(x)*sin(x)"
    x0: float = 0.5
    f2d_expr: str = "x**2*y + sin(x*y)"
    y0: float = 0.4
    outdir: str = os.path.join("AP_3", "output")
    save_csvs: bool = True
    # Step sizes to probe; exclude extremely tiny h that underflows double precision around 1e-16
    h_min: float = 1e-8
    h_max: float = 1e-1
    num_h: int = 12
    show_h: tuple[float, ...] = ()


def run(config: Config) -> None:
    ensure_dir(config.outdir)

    # 1D setup
    x, f1d_expr, f1d = parse_function_1d(config.f1d_expr)
    df_exact = exact_derivative_1d(f1d_expr, x, config.x0)

    # 2D setup
    (sx, sy), f2d_expr, f2d = parse_function_2d(config.f2d_expr)
    fx_exact, fy_exact = exact_gradient_2d(f2d_expr, sx, sy, config.x0, config.y0)

    # Geometry
    g1d = tangent_normal_1d(f1d, config.x0, df_exact)
    g2d = tangent_plane_normal_2d(f2d, config.x0, config.y0, fx_exact, fy_exact)

    print("-- 1D function --")
    print(f"f(x) = {f1d_expr}")
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
    print(f"f(x,y) = {f2d_expr}")
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
    plot_errors(hs, series_1d, "1D derivative error vs h", os.path.join(config.outdir, "errors_1d.png"))

    # 2D errors for fx and fy
    def sweep_2d(partial_fun, exact_val, label_prefix: str) -> None:
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
        plot_errors(hs, series, f"2D {label_prefix} error vs h", os.path.join(config.outdir, f"errors_2d_{label_prefix}.png"))

    sweep_2d(
        {
            "central": lambda h: fd_central_2d_x(f2d, config.x0, config.y0, h),
            "five": lambda h: fd_five_point_2d_x(f2d, config.x0, config.y0, h),
        },
        fx_exact,
        "fx",
    )
    sweep_2d(
        {
            "central": lambda h: fd_central_2d_y(f2d, config.x0, config.y0, h),
            "five": lambda h: fd_five_point_2d_y(f2d, config.x0, config.y0, h),
        },
        fy_exact,
        "fy",
    )

    # Illustrative plots for a few h values and geometry
    hs_show: List[float] = list(config.show_h) if config.show_h else [float(hs[0]), float(hs[len(hs)//2]), float(hs[-1])]
    try:
        plot_1d_tangent_and_secants(
            f1d,
            config.x0,
            df_exact,
            hs_show,
            os.path.join(config.outdir, "curve_tangent_secants.png"),
        )
    except Exception as e:
        print(f"[warn] Could not create 1D plot: {e}")
    try:
        plot_2d_contour_with_gradient(
            f2d,
            config.x0,
            config.y0,
            fx_exact,
            fy_exact,
            os.path.join(config.outdir, "surface_contour_gradient.png"),
        )
    except Exception as e:
        print(f"[warn] Could not create 2D contour plot: {e}")
    try:
        plot_3d_surface_and_plane(
            f2d,
            config.x0,
            config.y0,
            fx_exact,
            fy_exact,
            os.path.join(config.outdir, "surface_plane_3d.png"),
        )
    except Exception as e:
        print(f"[warn] Could not create 3D surface/plane plot: {e}")

    # Save a small README of formulas used
    readme_path = os.path.join(config.outdir, "README.txt")
    try:
        with open(readme_path, "w") as f:
            f.write(
                "Tangent/Normal and FD summary\n\n"
                "1D: tangent y = f(x0) + f'(x0)(x - x0); normal vector n ~ (-f'(x0), 1).\n"
                "2D: tangent plane z = f0 + fx(x - x0) + fy(y - y0); normal n ~ (-fx, -fy, 1).\n"
                "FD 1D: forward/backward O(h), central O(h^2), five-point O(h^4).\n"
                "FD 2D: central O(h^2) and five-point O(h^4) for partials.\n"
            )
    except Exception as e:
        print(f"[warn] Could not write {readme_path}: {e}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AP3: tangents/normals and FD accuracy")
    p.add_argument("--f1d", dest="f1d_expr", default=Config.f1d_expr, help="1D function f(x) expression")
    p.add_argument("--x0", type=float, default=Config.x0, help="point for 1D and 2D")
    p.add_argument("--f2d", dest="f2d_expr", default=Config.f2d_expr, help="2D function f(x,y) expression")
    p.add_argument("--y0", type=float, default=Config.y0, help="point y0 for 2D")
    p.add_argument("--outdir", default=Config.outdir, help="output directory for plots/CSVs")
    p.add_argument("--no-csv", dest="save_csvs", action="store_false", help="do not save CSV tables")
    p.add_argument("--h-min", type=float, default=Config.h_min, help="min step size")
    p.add_argument("--h-max", type=float, default=Config.h_max, help="max step size")
    p.add_argument("--num-h", type=int, default=Config.num_h, help="number of h values")
    p.add_argument(
        "--show-h",
        type=str,
        default="",
        help="comma-separated h values to visualize secant lines (e.g., 0.1,0.01,0.001)",
    )
    return p


def main(argv: List[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    cfg = Config(
        f1d_expr=args.f1d_expr,
        x0=args.x0,
        f2d_expr=args.f2d_expr,
        y0=args.y0,
        outdir=args.outdir,
        save_csvs=args.save_csvs,
        h_min=args.h_min,
        h_max=args.h_max,
        num_h=args.num_h,
        show_h=tuple(float(s) for s in args.show_h.split(',') if s.strip()) if args.show_h else (),
    )
    run(cfg)


if __name__ == "__main__":
    main()
