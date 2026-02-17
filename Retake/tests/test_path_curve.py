from __future__ import annotations

import numpy as np

from src.path_curve import build_parametric_spline


def test_parametric_spline_endpoints_and_projection() -> None:
    waypoints = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    curve = build_parametric_spline(waypoints)

    np.testing.assert_allclose(curve.position(0.0), waypoints[0], atol=1e-6)
    np.testing.assert_allclose(curve.position(curve.length), waypoints[-1], atol=1e-6)

    s_proj, closest = curve.project(np.array([[0.1, 0.8]], dtype=np.float64))
    assert s_proj.shape == (1,)
    assert closest.shape == (1, 2)
    assert 0.0 <= s_proj[0] <= curve.length
