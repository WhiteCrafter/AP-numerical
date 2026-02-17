from __future__ import annotations

import copy
import json

import numpy as np

from src.scenarios import setup_swarm_scenario


def _load_config() -> dict:
    with open("configs/default.json", "r", encoding="utf-8") as fh:
        return json.load(fh)


def _pairwise_min_distance(points: np.ndarray) -> float:
    diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    np.fill_diagonal(dists, np.inf)
    return float(np.min(dists))


def test_swarm_setup_uses_opposite_lanes_and_safe_spacing() -> None:
    config = _load_config()
    setup = setup_swarm_scenario(config)

    n_a = config["swarm"]["n_a"]
    start_a = np.array(config["swarm"]["start_a"], dtype=np.float64)
    start_b = np.array(config["swarm"]["start_b"], dtype=np.float64)
    centerline = start_b - start_a
    tangent = centerline / np.linalg.norm(centerline)
    normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)

    pos_a = setup.positions[:n_a]
    pos_b = setup.positions[n_a:]
    lat_a = (pos_a - start_a) @ normal
    lat_b = (pos_b - start_a) @ normal

    assert np.mean(lat_a) < 0.0
    assert np.mean(lat_b) > 0.0

    min_dist = _pairwise_min_distance(setup.positions)
    assert min_dist >= config["simulation"]["r_safe"]


def test_swarm_setup_seed_is_deterministic() -> None:
    config = _load_config()
    config["swarm"]["spawn_seed"] = 123
    first = setup_swarm_scenario(copy.deepcopy(config))
    second = setup_swarm_scenario(copy.deepcopy(config))

    np.testing.assert_allclose(first.positions, second.positions)
    np.testing.assert_allclose(first.targets, second.targets)
