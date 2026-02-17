from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.path_extraction import extract_waypoints


def _build_test_map(path: Path) -> None:
    image = np.zeros((120, 120), dtype=np.uint8)
    image[17:24, 20:93] = 255
    image[20:37, 89:96] = 255
    Image.fromarray(image, mode="L").save(path)


def test_extract_waypoints_returns_ordered_world_points_and_debug_artifacts(tmp_path: Path) -> None:
    map_path = tmp_path / "map.png"
    _build_test_map(map_path)

    debug_dir = tmp_path / "debug"
    waypoints = extract_waypoints(
        map_path=map_path,
        point_a_world=np.array([1.0, 1.0], dtype=np.float64),
        point_b_world=np.array([4.6, 1.8], dtype=np.float64),
        origin_world=np.array([0.0, 0.0], dtype=np.float64),
        meters_per_pixel=np.array([0.05, 0.05], dtype=np.float64),
        waypoint_stride=8,
        debug_output_dir=debug_dir,
        debug_prefix="unit",
    )

    assert waypoints.shape[1] == 2
    assert np.allclose(waypoints[0], np.array([1.0, 1.0]), atol=0.051)
    assert np.allclose(waypoints[-1], np.array([4.6, 1.8]), atol=0.051)

    deltas = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    assert np.all(deltas > 0.0)

    assert (debug_dir / "unit_grayscale.png").exists()
    assert (debug_dir / "unit_traversable_with_points.png").exists()
    assert (debug_dir / "unit_path_only.png").exists()


def test_extract_waypoints_with_map_resize_factor_preserves_world_endpoints(tmp_path: Path) -> None:
    map_path = tmp_path / "map.png"
    _build_test_map(map_path)

    waypoints = extract_waypoints(
        map_path=map_path,
        point_a_world=np.array([1.0, 1.0], dtype=np.float64),
        point_b_world=np.array([4.6, 1.8], dtype=np.float64),
        origin_world=np.array([0.0, 0.0], dtype=np.float64),
        meters_per_pixel=np.array([0.05, 0.05], dtype=np.float64),
        waypoint_stride=8,
        map_resize_factor=0.5,
    )

    assert waypoints.shape[1] == 2
    assert np.allclose(waypoints[0], np.array([1.0, 1.0]), atol=0.12)
    assert np.allclose(waypoints[-1], np.array([4.6, 1.8]), atol=0.12)
