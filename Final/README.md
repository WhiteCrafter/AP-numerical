# Drone Swarm Text Formation System

## Overview

This project implements a **drone swarm simulation system** that converts text (either predefined or obtained via OCR) into a spatial formation and animates a group of drones transitioning between text shapes.

The swarm dynamics are modeled as a **continuous-time dynamical system solved as an Initial Value Problem (IVP)** using numerical integration. Each drone is controlled independently while interacting with other drones through repulsion forces to avoid collisions.

The system consists of three major components:

1. **Text Processing and Target Generation**
2. **Assignment Optimization**
3. **Dynamic Swarm Motion Simulation and Rendering**

## System Architecture

```
Input Image / Text
        ↓
OCR / Text Parser
        ↓
Pixel Grid Generator
        ↓
Target Point Cloud
        ↓
Optimal Assignment (Hungarian Algorithm)
        ↓
Swarm Motion Planner (IVP)
        ↓
Numerical Integration
        ↓
Matplotlib Animation Renderer
```


## Mathematical Model

### State Variables

Each drone is represented by:

Position vector:

$$
x_i(t) \in \mathbb{R}^2
$$

Velocity vector:

$$
v_i(t) \in \mathbb{R}^2
$$


## Initial Value Problem (IVP)

The swarm motion is modeled as a second-order dynamical system:

$$
\dot{x}_i = v_i
$$

$$
\dot{v}_i = a_i(x, v)
$$

Where acceleration is computed as:

$$
a_i = k_v (v_{des,i} - v_i) - k_d v_i + \frac{F_{rep,i}}{m}
$$

Initial conditions:

$$
x_i(0) = x_{start,i}
$$

$$
v_i(0) = 0
$$


## Desired Velocity Controller

Each drone computes a desired velocity toward its assigned target:

$$
v_{des,i} = \frac{x_{target,i} - x_i}{T_{remaining}}
$$

Velocity saturation constraint:

$$
\|v_i\| \le v_{max}
$$

## Damping Term (Stabilization)

The damping term:

$$
- k_d v_i
$$

models viscous drag and stabilizes the system to prevent oscillations.


## Collision Avoidance

To prevent collisions between drones, a repulsive force is applied:

$$
F_{rep} = k_{rep} \frac{x_i - x_j}{\|x_i - x_j\|^3}
$$

This enforces a minimum safety distance between agents.

## Numerical Integration

The system is discretized using explicit Euler integration:

$$
v_{t+1} = v_t + a_t \Delta t
$$

$$
x_{t+1} = x_t + v_{t+1} \Delta t
$$

Where:

$$
\Delta t = \frac{interval}{1000}
$$


## Assignment Optimization

Drone-to-target matching is solved using the Hungarian algorithm:

Objective function:

$$
\min \sum_i \|x_{start,i} - x_{target,i}\|^2
$$

This minimizes total travel distance and prevents path crossings.



## Text Processing Pipeline

Characters are rasterized into **8×8 binary grids**.  
Each active pixel is converted into a 2D spatial coordinate used as a drone target.


## OCR Pipeline

When OCR is enabled:

```
Input Image → OCR → Extracted Text → Swarm Renderer
```

GPU acceleration is supported optionally.

---

## Motion Segmentation

The system supports multi-stage transitions:

```
HELLO → MODDING → NEXT
```

Each segment is simulated independently and concatenated into a single animation timeline.

---

## Rendering System

Visualization is implemented using **Matplotlib Animation**:

---

## Parameter Summary

| Parameter | Description |
|----------|------------|
| SWARM_V_MAX | Maximum drone speed |
| SWARM_K_V | Attraction / tracking gain |
| SWARM_K_D | Damping coefficient |
| SWARM_K_REP | Repulsion force strength |
| SWARM_R_SAFE | Minimum separation radius |
| SWARM_MASS | Drone inertia scaling |
| SWARM_FRAMES | Simulation resolution |
| SWARM_INTERVAL | Integration timestep |

## Conclusion

This project demonstrates practical application of numerical methods, control systems, optimization algorithms, and real-time simulation to produce a visually coherent drone swarm formation system.
