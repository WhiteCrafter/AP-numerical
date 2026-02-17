from __future__ import annotations

import copy
import json

import numpy as np

import src.scenarios as scenarios
from src.scenarios import setup_swarm_scenario


def _load_config() -> dict:
    with open("configs/default.json", "r", encoding="utf-8") as fh:
        return json.load(fh)


def _pairwise_min_distance(points: np.ndarray) -> float:
    diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    np.fill_diagonal(dists, np.inf)
    return float(np.min(dists))


def test_swarm_setup_uses_cluster_spawns_and_safe_spacing() -> None:
    config = _load_config()
    setup = setup_swarm_scenario(config)

    n_a = config["swarm"]["n_a"]
    start_a = np.array(config["swarm"]["start_a"], dtype=np.float64)
    start_b = np.array(config["swarm"]["start_b"], dtype=np.float64)

    pos_a = setup.positions[:n_a]
    pos_b = setup.positions[n_a:]

    cluster_radius = float(config["swarm"]["path_radius"])
    dist_from_a = np.linalg.norm(pos_a - start_a, axis=1)
    dist_from_b = np.linalg.norm(pos_b - start_b, axis=1)
    assert np.all(dist_from_a <= cluster_radius + 1e-6)
    assert np.all(dist_from_b <= cluster_radius + 1e-6)

    # Ensure each group is a cluster, not a single exact spawn point.
    assert np.max(np.linalg.norm(pos_a - np.mean(pos_a, axis=0), axis=1)) > 1e-3
    assert np.max(np.linalg.norm(pos_b - np.mean(pos_b, axis=0), axis=1)) > 1e-3

    np.testing.assert_allclose(setup.targets[:n_a], np.repeat(start_b[np.newaxis, :], n_a, axis=0))
    np.testing.assert_allclose(setup.targets[n_a:], np.repeat(start_a[np.newaxis, :], config["swarm"]["n_b"], axis=0))

    min_dist = _pairwise_min_distance(setup.positions)
    assert min_dist >= config["simulation"]["r_safe"]


def test_swarm_setup_seed_is_deterministic() -> None:
    config = _load_config()
    config["swarm"]["spawn_seed"] = 123
    first = setup_swarm_scenario(copy.deepcopy(config))
    second = setup_swarm_scenario(copy.deepcopy(config))

    np.testing.assert_allclose(first.positions, second.positions)
    np.testing.assert_allclose(first.targets, second.targets)


def test_swarm_setup_uses_map_extracted_path(monkeypatch, tmp_path) -> None:
    config = _load_config()
    fake_map = tmp_path / "map.png"
    fake_map.write_bytes(b"placeholder")
    config["path_extraction"]["map_path"] = str(fake_map)

    expected = np.array([[1.0, 1.0], [2.0, 2.8], [5.0, 3.2], [8.0, 1.0]], dtype=np.float64)

    def fake_extract_waypoints(**_: object) -> np.ndarray:
        return expected.copy()

    monkeypatch.setattr(scenarios, "extract_waypoints", fake_extract_waypoints)

    setup = setup_swarm_scenario(config)
    assert setup.path_curve is not None
    np.testing.assert_allclose(setup.path_curve.control_points, expected)
