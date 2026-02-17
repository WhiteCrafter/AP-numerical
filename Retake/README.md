# Autonomous Navigation Project Skeleton

This repository contains a modular skeleton for a swarm-navigation project with three scenarios:

1. Single robot in a constrained path.
2. Bidirectional swarm navigation in a constrained path.
3. Robot navigation in a pedestrian flow field.

## Quick start

```bash
python main.py --scenario single --config configs/default.json
```

For `single` and `swarm`, the app opens map point selection so you can choose Point A and Point B interactively; swarm then spawns one group near A moving toward B and one group near B moving toward A. Because of this prompt flow, `swarm.start_a`/`swarm.start_b` are optional in config.

This first scaffold intentionally includes placeholder implementations and TODO markers so each module can be developed independently.

## Repository layout

- `main.py`: CLI entrypoint and scenario dispatcher.
- `src/config.py`: configuration loading and defaults.
- `src/types.py`: shared dataclasses and type contracts.
- `src/dynamics.py`: force and acceleration model.
- `src/simulation.py`: time-stepping orchestration.
- `src/scenarios.py`: single/swarm/pedestrian scenario setup and placeholders.
- `src/io_or_visualization.py`: trajectory export utility.
- `src/metrics.py`: scalar metrics and checks.
- `tests/test_smoke.py`: basic smoke tests for CLI orchestration.

## Notes

- `single.stop_radius`: distance to goal B where robot velocity is clamped to zero.
- `swarm.stop_radius`: distance to each robot's respective finish target where velocity is clamped to zero.


- The skeleton currently uses only Python standard library modules to stay runnable in restricted environments.
- Trajectories are exported to CSV files in `outputs/plots/`.
- You can later swap in NumPy/Matplotlib/OpenCV without changing top-level orchestration.

## Next implementation steps

- Replace placeholder path extraction with map-based centerline extraction.
- Implement corridor boundary forces.
- Integrate video optical flow for scenario 3.
- Expand tests with collision and goal-reaching assertions.
