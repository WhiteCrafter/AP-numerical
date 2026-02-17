from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass
class ParametricSpline2D:
    """Arc-length parameterized Catmull-Rom spline over 2D control points."""

    control_points: FloatArray
    samples: FloatArray
    sample_s: FloatArray

    @property
    def length(self) -> float:
        return float(self.sample_s[-1]) if self.sample_s.size else 0.0

    def _clip_s(self, s: float | FloatArray) -> FloatArray:
        arr = np.asarray(s, dtype=np.float64)
        if self.sample_s.size == 0:
            return np.zeros_like(arr)
        return np.clip(arr, 0.0, self.sample_s[-1])

    def position(self, s: float | FloatArray) -> FloatArray:
        s_arr = self._clip_s(s)
        x = np.interp(s_arr, self.sample_s, self.samples[:, 0])
        y = np.interp(s_arr, self.sample_s, self.samples[:, 1])
        if np.ndim(s) == 0:
            return np.array([float(x), float(y)], dtype=np.float64)
        return np.column_stack((x, y))

    def tangent(self, s: float) -> FloatArray:
        ds = max(1e-3, 1e-3 * max(self.length, 1.0))
        p_prev = self.position(max(0.0, s - ds))
        p_next = self.position(min(self.length, s + ds))
        vec = np.asarray(p_next - p_prev, dtype=np.float64)
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-12:
            return np.array([1.0, 0.0], dtype=np.float64)
        return vec / norm

    def project(self, points: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Project points onto sampled curve, returning arclength and closest points."""
        pts = np.asarray(points, dtype=np.float64)
        diffs = pts[:, np.newaxis, :] - self.samples[np.newaxis, :, :]
        d2 = np.sum(diffs * diffs, axis=2)
        idx = np.argmin(d2, axis=1)
        s = self.sample_s[idx]
        return s.astype(np.float64), self.samples[idx].astype(np.float64)



def catmull_rom_chain(control_points: FloatArray, points_per_segment: int = 12) -> FloatArray:
    cps = np.asarray(control_points, dtype=np.float64)
    n = cps.shape[0]
    if n < 2:
        return cps.copy()
    if n == 2:
        t = np.linspace(0.0, 1.0, points_per_segment + 1)[:, np.newaxis]
        return cps[0] * (1.0 - t) + cps[1] * t

    out: list[np.ndarray] = []
    seg_pts = max(3, int(points_per_segment))
    for i in range(n - 1):
        p0 = cps[max(0, i - 1)]
        p1 = cps[i]
        p2 = cps[i + 1]
        p3 = cps[min(n - 1, i + 2)]
        tvals = np.linspace(0.0, 1.0, seg_pts, endpoint=(i == n - 2), dtype=np.float64)
        t = tvals[:, np.newaxis]
        t2 = t * t
        t3 = t2 * t
        seg = 0.5 * ((2.0 * p1) + (-p0 + p2) * t + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
        out.append(seg)
    return np.vstack(out)


def build_parametric_spline(control_points: FloatArray, points_per_segment: int = 12) -> ParametricSpline2D:
    cps = np.asarray(control_points, dtype=np.float64)
    dense = catmull_rom_chain(cps, points_per_segment=points_per_segment)
    if dense.shape[0] == 0:
        dense = cps.copy()
    s = np.zeros(dense.shape[0], dtype=np.float64)
    if dense.shape[0] > 1:
        s[1:] = np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))
    return ParametricSpline2D(control_points=cps, samples=dense, sample_s=s)
