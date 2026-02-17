from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .types import SimulationParams


DEFAULT_CONFIG = Path("configs/default.json")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def simulation_params(config: dict[str, Any]) -> SimulationParams:
    sim = config["simulation"]
    return SimulationParams(
        dt=float(sim["dt"]),
        t_end=float(sim["t_end"]),
        v_max=float(sim["v_max"]),
        mass=float(sim["mass"]),
        k_p=float(sim["k_p"]),
        k_d=float(sim["k_d"]),
        k_rep=float(sim["k_rep"]),
        r_safe=float(sim["r_safe"]),
        k_wall=float(sim.get("k_wall", 0.0)),
        wall_margin=float(sim.get("wall_margin", 0.0)),
    )
