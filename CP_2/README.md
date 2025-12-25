# Modeling Smartphone Thermal Dynamics Using Implicit ODE Methods

## 1. Introduction

In this project, I model the thermal behavior of a smartphone over time using a system of ordinary differential equations (ODEs).
The goal is to understand how heat is generated inside the phone, how it is transferred to the phone case, and how it is eventually lost to the surrounding air.

In addition to modeling the physical system, the project focuses on **numerical methods**.
The ODE system is solved using the **Implicit Euler method**, and two different nonlinear solvers are compared:

- Fixed-Point Iteration  
- Newton’s Method  

The comparison is based on convergence behavior and computational efficiency.



## 2. Physical Problem Description

A smartphone generates heat internally due to components such as the CPU and battery.
This heat flows through several stages:

1. Heat is generated inside the phone
2. Heat transfers from the phone interior to the phone case
3. Heat is released from the case into the surrounding air

To capture this process, the phone is modeled as a **two-node thermal system**:

- Phone interior temperature
- Phone case temperature

This is a simplified but physically reasonable model that allows us to study thermal dynamics without unnecessary complexity.


## 3. State Variables and Parameters

### State variables

The system state is represented as:

$$
u(t) = \begin{bmatrix} T_p(t) \\ T_c(t) \end{bmatrix}
$$

where:
- $T_p(t)$ is the internal phone temperature (°C)
- $T_c(t)$ is the phone case temperature (°C)

### Parameters

The model uses the following parameters:

- $Q_{gen}$ – constant internal heat generation (W)
- $k_{pc}$ – heat transfer coefficient between phone and case (W/K)
- $k_{ca}$ – heat transfer coefficient between case and air (W/K)
- $C_p$ – thermal capacity of the phone interior (J/K)
- $C_c$ – thermal capacity of the phone case (J/K)
- $T_{amb}$ – ambient air temperature (°C)

Changing $k_{ca}$ allows simulation of **thin** versus **thick** phone cases.


## 4. Mathematical Model (ODE System)

The heat transfer process is modeled using the following system of ODEs:

$$
\frac{dT_p}{dt} = \frac{Q_{gen} - k_{pc}(T_p - T_c)}{C_p}
$$

$$
\frac{dT_c}{dt} = \frac{k_{pc}(T_p - T_c) - k_{ca}(T_c - T_{amb})}{C_c}
$$

### Interpretation

- The phone temperature increases due to internal heat generation.
- Heat flows from the phone to the case depending on the temperature difference.
- The case loses heat to the environment.

These equations are implemented in the code inside the `rhs()` function.

```python
def rhs(state, params):
    T_p, T_c = state
    ...
    dT_p = (Q_gen - k_pc * (T_p - T_c)) / C_p
    dT_c = (k_pc * (T_p - T_c) - k_ca * (T_c - T_amb)) / C_c
    return np.array([dT_p, dT_c])
```

## 5. Why Implicit Euler?

The ODE system represents a **thermal process**, which often exhibits **stiff behavior**.  
This means that some parts of the system change much faster than others.

Explicit methods would require very small time steps to remain stable.  
To avoid this, the **Implicit Euler method** is used.

The implicit Euler update rule is:

$$u_{n+1} = u_n + \Delta t \ f(u_{n+1})$$

Since the right-hand side depends on the unknown future state $u_(n+1)$, a nonlinear system must be solved at each time step.

---

## 6. Nonlinear Solvers Used
I compare two Nonlinear Solving method Fixed-Point Iteration & Newton's Method

### 6.1 Fixed-Point Iteration

Fixed-point iteration solves the implicit equation by repeatedly substituting the current guess:
$$
u^{(k+1)} = u_n + \Delta t \, f(u^{(k)})$$

This method is:
- Simple to implement    
- Computationally cheap per iteration    
- Potentially slow to converge for stiff problems   

In the code, fixed-point iteration is implemented in:
```python
def implicit_step_fixed_point(u_n, dt, params, tol=1e-6, max_iter=50):
    ...
```

### 6.2 Newton’s Method

Newton’s method solves the nonlinear system more efficiently by using derivative information.

We define the function:

$$G(u) = u - u_n - \Delta t f(u)$$

Newton’s update step is:

$$u^((k+1)) = u^((k)) - J^((-1)) G(u^((k)))$$

where $J$ is the Jacobian matrix.

In this project, the Jacobian is constant and easy to compute:
```python
def jacobian_f(params):
    return np.array([
        [-k_pc / C_p,  k_pc / C_p],
        [ k_pc / C_c, -(k_pc + k_ca) / C_c]
    ])
```
Newton’s method converges faster and is more robust for stiff systems.

---

## 7. Comparison of Methods

Both solvers are used with the **same Implicit Euler scheme**.  
This isolates the effect of the nonlinear solver.

The comparison criteria are:

- Number of iterations per time step
- Stability and convergence    
- Agreement of the final temperature results    

The iteration counts are printed and averaged:

`print(f"Thin fixed: {np.mean(it_fp_thin):.2f}") print(f"Thin newton: {np.mean(it_newton_thin):.2f}")`

---

## 8. Simulation Scenarios

Two scenarios are simulated:

### Thin phone case

- High heat transfer to air ($k_{ca} = 1.2$)
    
- Better cooling
    

### Thick phone case

- Low heat transfer to air ($k_{ca} = 0.4$)
    
- Worse cooling
    

All other parameters are kept the same to isolate the effect of insulation.


## 9. Results and Visualization

The results are visualized by plotting:

- Phone temperature $T_p(t)$
    
- Case temperature $T_c(t)$
    

for both solvers and both cases.

```python
axes[0].plot(t, fp_thin[:, 0], label="T_p fixed-point")
axes[0].plot(t, newton_thin[:, 0], "--", label="T_p newton")
```

### Observations

- Thin cases result in lower steady-state temperatures.
- Thick cases trap heat and lead to higher temperatures.    
- Fixed-point and Newton solutions overlap closely.    
- Newton’s method requires significantly fewer iterations.


## 10. Conclusion
<img width="984" height="385" alt="image" src="https://github.com/user-attachments/assets/456c6509-6894-4609-87fc-a6f86d5b1f1a" />

This project demonstrates how a real-world thermal system can be modeled using a system of ODEs and solved numerically.  
By using the Implicit Euler method, stability is maintained even with relatively large time steps.

Comparing Fixed-Point Iteration and Newton’s Method shows that:

- Both methods produce accurate solutions
- Newton’s method converges faster and more efficiently
- Solver choice matters even when the time integration method is the same

Overall, the project highlights both physical modeling and numerical analysis aspects of solving stiff ODE systems.
