from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_config, simulation_params
from src.metrics import final_goal_error, min_pairwise_distance
from src.path_extraction import select_points_on_map
from src.scenarios import build_scenario
from src.simulation import run_simulation
from src.io_or_visualization import export_trajectories_csv, maybe_visualize



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autonomous navigation skeleton runner")
    parser.add_argument("--scenario", choices=["single", "swarm", "pedestrian"], default="single")
    parser.add_argument("--config", default="configs/default.json")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.scenario in {"single", "swarm"}:
        path_cfg = config.get("path_extraction", {})
        map_path = path_cfg.get("map_path")
        if not map_path:
            raise ValueError(
                "path_extraction.map_path must be set for single/swarm scenario manual point selection."
            )

        point_a, point_b = select_points_on_map(
            map_path=map_path,
            origin_world=path_cfg["world_origin"],
            meters_per_pixel=path_cfg["meters_per_pixel"],
            traversable_threshold=float(path_cfg.get("traversable_threshold", 0.5)),
            map_resize_factor=float(path_cfg.get("map_resize_factor", 1.0)),
        )
        print(f"Selected A={point_a.tolist()} B={point_b.tolist()}")

        if args.scenario == "single":
            config.setdefault("single", {})
            config["single"]["start"] = point_a.tolist()
            config["single"]["goal"] = point_b.tolist()
        else:
            config.setdefault("swarm", {})
            config["swarm"]["start_a"] = point_a.tolist()
            config["swarm"]["start_b"] = point_b.tolist()

        config.setdefault("path_extraction", {})
        config["path_extraction"]["point_a"] = point_a.tolist()
        config["path_extraction"]["point_b"] = point_b.tolist()


    params = simulation_params(config)

    setup = build_scenario(args.scenario, config)
    result = run_simulation(setup, params)

    output_dir = Path(config["output"]["output_dir"])
    output_file = output_dir / f"{args.scenario}_trajectories.csv"
    artifact_path = export_trajectories_csv(result, output_file)

    config.setdefault("output", {})
    config["output"].setdefault("video_name", f"{args.scenario}_simulation.gif")

    metrics = {
        "min_pairwise_distance": min_pairwise_distance(result),
        "mean_final_goal_error": final_goal_error(result, setup.targets),
    }

    viz_path = maybe_visualize(result, setup, config)

    print(f"Saved: {artifact_path}")
    if viz_path is not None:
        print(f"Saved video: {viz_path}")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
