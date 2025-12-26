import numpy as np
from scipy.spatial import Voronoi, Delaunay, cKDTree, voronoi_plot_2d
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def generate_points(count, seed, low=0.0, high=100.0):
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=(count, 2))


def assign_to_nearest(population, facilities):
    tree = cKDTree(facilities)
    _, indices = tree.query(population)
    return indices


def plot_diagrams(facilities, population, assignment, bounds):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Population points colored by nearest facility
    if population is not None and len(population) > 0:
        ax.scatter(
            population[:, 0],
            population[:, 1],
            c=assignment,
            cmap="tab20",
            s=12,
            alpha=0.4,
            zorder=1,
        )

    # Voronoi diagram for service areas
    vor = Voronoi(facilities)
    voronoi_plot_2d(
        vor,
        ax=ax,
        show_points=False,
        show_vertices=False,
        line_colors="tab:orange",
        line_width=1.2,
        line_alpha=0.8,
    )

    # Delaunay triangulation edges (draw each edge once)
    tri = Delaunay(facilities)
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            a = simplex[i]
            b = simplex[(i + 1) % 3]
            edges.add(tuple(sorted((a, b))))
    for a, b in edges:
        p0, p1 = facilities[a], facilities[b]
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="tab:blue", lw=0.9, alpha=0.7)

    # Facility locations
    ax.scatter(
        facilities[:, 0],
        facilities[:, 1],
        marker="^",
        s=90,
        c="black",
        edgecolors="white",
        linewidths=0.6,
        zorder=3,
    )

    # Legend proxies
    legend_items = [
        Line2D([0], [0], color="tab:orange", lw=1.4, label="Voronoi edges"),
        Line2D([0], [0], color="tab:blue", lw=1.0, label="Delaunay edges"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Population (assigned)",
            markerfacecolor="gray",
            markersize=6,
            alpha=0.6,
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            label="Facilities",
            markerfacecolor="black",
            markeredgecolor="white",
            markersize=8,
        ),
    ]
    ax.legend(handles=legend_items, loc="upper right")

    ax.set_title("Emergency Facilities: Voronoi and Delaunay")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[0], bounds[1])
    ax.set_aspect("equal", "box")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    # Simple setup for a city-sized square area
    bounds = (0.0, 100.0)
    num_facilities = 10
    num_population = 300

    facilities = generate_points(num_facilities, seed=7, low=bounds[0], high=bounds[1])
    population = generate_points(
        num_population, seed=21, low=bounds[0], high=bounds[1]
    )
    assignment = assign_to_nearest(population, facilities)

    plot_diagrams(facilities, population, assignment, bounds)


if __name__ == "__main__":
    main()
