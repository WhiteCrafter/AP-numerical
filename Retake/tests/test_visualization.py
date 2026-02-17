from __future__ import annotations

from pathlib import Path

import numpy as np

from src.io_or_visualization import render_simulation_video
from src.path_curve import build_parametric_spline
from src.types import ScenarioSetup, SimulationResult


def test_render_simulation_video_saves_gif(tmp_path: Path) -> None:
    times = np.linspace(0.0, 0.2, 6)
    positions = np.zeros((times.size, 1, 2), dtype=np.float64)
    positions[:, 0, 0] = np.linspace(0.0, 1.0, times.size)
    positions[:, 0, 1] = np.linspace(0.0, 0.6, times.size)
    velocities = np.zeros_like(positions)

    result = SimulationResult(times=times, positions=positions, velocities=velocities)
    setup = ScenarioSetup(
        positions=positions[0],
        velocities=velocities[0],
        targets=np.array([[1.0, 0.6]], dtype=np.float64),
        path_curve=build_parametric_spline(np.array([[0.0, 0.0], [1.0, 0.6]], dtype=np.float64)),
        path_radius=0.2,
    )

    out = tmp_path / "sim.gif"
    path = render_simulation_video(result, setup, output_path=out, show_popup=False, fps=10)

    assert path is not None
    assert path.exists()
    assert path.stat().st_size > 0
