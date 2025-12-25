# Smartphone Cooling Model (Implicit Euler)

## 1. Project title
Smartphone Cooling Model with a Phone Case

## 2. Short explanation of the physical problem
This project models how a phone cools down while it is generating heat. A phone case can slow down or speed up cooling depending on how well it transfers heat to the air. I compare a thin case and a thick case.

## 3. Description of the ODE system
The model has two temperatures:
- `T_p(t)`: phone interior temperature
- `T_c(t)`: case temperature
- `T_amb`: ambient air temperature (constant)

The equations are:

```
T_p' = (Q_gen - k_pc (T_p - T_c)) / C_p
T_c' = (k_pc (T_p - T_c) - k_ca (T_c - T_amb)) / C_c
```

`Q_gen` is constant heat generation, `k_pc` is phone-to-case heat transfer, `k_ca` is case-to-air heat transfer, and `C_p`, `C_c` are thermal capacities.

## 4. Explanation of why implicit Euler is used
Implicit Euler is stable for stiff problems, and heat transfer systems can be stiff when the time step is not very small. This method lets me use a reasonable time step without the solution blowing up.

## 5. Brief explanation of the solvers
- **Fixed-Point Iteration:** I guess the next temperature and keep updating it using the implicit equation until it stops changing much.
- **Newton's Method:** I solve the implicit equation with a linearization (Jacobian). It usually converges faster.

## 6. Description of the simulation scenarios
- **Thin case:** larger `k_ca` (better heat transfer to air).
- **Thick case:** smaller `k_ca` (worse heat transfer to air).

This shows how the case affects cooling.

## 7. How to run the code
From the project folder:

```
python CP_2/main.py
```

## 8. What the plots show
The plots show `T_p(t)` for each case. I plot both the fixed-point and Newton results to compare solution behavior. The thin case should cool faster, and the thick case should stay warmer longer.
