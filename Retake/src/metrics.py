from __future__ import annotations

import numpy as np

from .types import FloatArray, SimulationResult


def min_pairwise_distance(result: SimulationResult) -> float:
    if result.positions.size == 0 or result.positions.shape[1] < 2:
        return float("inf")

    min_dist = float("inf")
    for step in result.positions:
        diff = step[:, np.newaxis, :] - step[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        min_dist = min(min_dist, float(np.min(dist)))
    return min_dist


def final_goal_error(result: SimulationResult, targets: FloatArray) -> float:
    if result.positions.size == 0:
        return 0.0
    final_pos = result.positions[-1]
    return float(np.mean(np.linalg.norm(final_pos - targets, axis=1)))
