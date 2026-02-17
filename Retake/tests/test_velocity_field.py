from __future__ import annotations

import copy

import numpy as np

from src.scenarios import (
    _build_interpolated_velocity_field,
    _video_frames_to_velocity_grids,
    setup_pedestrian_scenario,
)


class _FakeCV2:
    @staticmethod
    def calcOpticalFlowFarneback(prev, nxt, *_args, **_kwargs):
        h, w = prev.shape
        flow = np.zeros((h, w, 2), dtype=np.float32)
        flow[..., 0] = 2.0
        flow[..., 1] = -1.0
        return flow


def test_interpolated_field_shape_and_range() -> None:
    grid_t0 = np.zeros((3, 3, 2), dtype=np.float64)
    grid_t1 = np.full((3, 3, 2), [2.0, -2.0], dtype=np.float64)
    grids = np.stack([grid_t0, grid_t1], axis=0)

    field = _build_interpolated_velocity_field(
        velocity_grids_world=grids,
        fps=2.0,
        world_origin=np.array([0.0, 0.0], dtype=np.float64),
        meters_per_pixel=np.array([1.0, 1.0], dtype=np.float64),
        invert_y_axis=False,
        interpolation="bilinear",
    )

    samples = np.array([[0.5, 0.5], [1.2, 0.2]], dtype=np.float64)
    out = field(samples, 0.25)

    assert out.shape == (2, 2)
    assert np.all(out >= np.array([0.0, -2.0]))
    assert np.all(out <= np.array([2.0, 0.0]))


def test_velocity_grid_conversion_is_deterministic(monkeypatch) -> None:
    import src.scenarios as scenarios

    monkeypatch.setattr(scenarios, "cv2", _FakeCV2())
    frames = [np.zeros((4, 4), dtype=np.uint8) for _ in range(3)]

    grids = _video_frames_to_velocity_grids(
        frames_gray=frames,
        fps=10.0,
        meters_per_pixel=np.array([0.1, 0.2], dtype=np.float64),
        invert_y_axis=True,
    )

    expected = np.array([2.0, 2.0], dtype=np.float64)
    np.testing.assert_allclose(grids[0, 0, 0], expected)
    np.testing.assert_allclose(grids[1, 3, 2], expected)


def test_pedestrian_flow_fallback_to_placeholder_on_missing_video() -> None:
    config = {
        "pedestrian": {
            "start": [0.0, 0.0],
            "goal": [1.0, 0.0],
            "flow_estimation": {
                "mode": "video",
                "video_path": "does/not/exist.mp4",
                "on_error": "placeholder",
                "placeholder_velocity": [0.25, -0.1],
            },
        }
    }

    setup = setup_pedestrian_scenario(copy.deepcopy(config))
    assert setup.velocity_field is not None
    v = setup.velocity_field(np.array([0.0, 0.0], dtype=np.float64), 0.0)
    np.testing.assert_allclose(v, np.array([0.25, -0.1], dtype=np.float64))
