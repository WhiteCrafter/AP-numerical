import numpy as np
import matplotlib.pyplot as plt


def rhs(state, params):
    # Right-hand side of the ODE system
    T_p, T_c = state
    Q_gen = params["Q_gen"]
    k_pc = params["k_pc"]
    k_ca = params["k_ca"]
    C_p = params["C_p"]
    C_c = params["C_c"]
    T_amb = params["T_amb"]

    dT_p = (Q_gen - k_pc * (T_p - T_c)) / C_p
    dT_c = (k_pc * (T_p - T_c) - k_ca * (T_c - T_amb)) / C_c
    return np.array([dT_p, dT_c], dtype=float)


def jacobian_f(params):
    # Jacobian of the RHS with respect to [T_p, T_c]
    k_pc = params["k_pc"]
    k_ca = params["k_ca"]
    C_p = params["C_p"]
    C_c = params["C_c"]
    return np.array(
        [
            [-k_pc / C_p, k_pc / C_p],
            [k_pc / C_c, -(k_pc + k_ca) / C_c],
        ],
        dtype=float,
    )


def implicit_step_fixed_point(u_n, dt, params, tol=1e-6, max_iter=50):
    # Fixed-point iteration for implicit Euler
    u = u_n.copy()
    for it in range(1, max_iter + 1):
        u_next = u_n + dt * rhs(u, params)
        if np.linalg.norm(u_next - u, ord=np.inf) < tol:
            return u_next, it
        u = u_next
    return u_next, max_iter


def implicit_step_newton(u_n, dt, params, tol=1e-6, max_iter=20):
    # Newton's method for implicit Euler
    u = u_n.copy()
    J = np.eye(2) - dt * jacobian_f(params)
    for it in range(1, max_iter + 1):
        G = u - u_n - dt * rhs(u, params)
        delta = np.linalg.solve(J, G)
        u = u - delta
        G_after = u - u_n - dt * rhs(u, params)
        if np.linalg.norm(G_after, ord=np.inf) < tol:
            return u, it
    return u, max_iter


def run_case(case_name, k_ca, base_params, t, dt, init_state):
    params = dict(base_params)
    params["k_ca"] = k_ca
    n_steps = len(t) - 1

    states_fp = np.zeros((n_steps + 1, 2), dtype=float)
    states_newton = np.zeros((n_steps + 1, 2), dtype=float)
    states_fp[0] = init_state
    states_newton[0] = init_state

    it_fp = np.zeros(n_steps, dtype=int)
    it_newton = np.zeros(n_steps, dtype=int)

    print(f"\n{case_name} case (k_ca={k_ca})")
    for n in range(n_steps):
        states_fp[n + 1], it_fp[n] = implicit_step_fixed_point(
            states_fp[n], dt, params
        )
        states_newton[n + 1], it_newton[n] = implicit_step_newton(
            states_newton[n], dt, params
        )
        print(f"step {n + 1:02d}: fixed={it_fp[n]:2d}, newton={it_newton[n]:2d}")

    max_diff = np.max(np.abs(states_fp[:, 0] - states_newton[:, 0]))
    print(f"max |T_p fixed - newton| = {max_diff:.4f} C")
    return params, states_fp, states_newton, it_fp, it_newton


def main():
    # Basic parameters (units: W, W/K, J/K, C)
    base_params = {
        "Q_gen": 5.0,
        "k_pc": 0.8,
        "k_ca": 0.6,  # overwritten per case
        "C_p": 500.0,
        "C_c": 300.0,
        "T_amb": 25.0,
    }

    # Initial temperatures
    init_state = np.array([40.0, 30.0], dtype=float)

    # Time grid
    t0 = 0.0
    t_end = 2300.0
    dt = 0.1
    t = np.arange(t0, t_end + dt, dt)

    # Two case scenarios
    thin_k_ca = 1.2
    thick_k_ca = 0.4

    _, fp_thin, newton_thin, it_fp_thin, it_newton_thin = run_case(
        "Thin",
        thin_k_ca,
        base_params,
        t,
        dt,
        init_state,
    )
    _, fp_thick, newton_thick, it_fp_thick, it_newton_thick = run_case(
        "Thick",
        thick_k_ca,
        base_params,
        t,
        dt,
        init_state,
    )

    print("\nAverage iterations per step:")
    print(f"Thin fixed: {np.mean(it_fp_thin):.2f}")
    print(f"Thin newton: {np.mean(it_newton_thin):.2f}")
    print(f"Thick fixed: {np.mean(it_fp_thick):.2f}")
    print(f"Thick newton: {np.mean(it_newton_thick):.2f}")

    # Plot T_p(t) and T_c(t) for both cases
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    axes[0].plot(t, fp_thin[:, 0], label="T_p fixed-point")
    axes[0].plot(t, newton_thin[:, 0], "--", label="T_p newton")
    axes[0].plot(t, fp_thin[:, 1], ":", label="T_c fixed-point")
    axes[0].plot(t, newton_thin[:, 1], "-.", label="T_c newton")
    axes[0].set_title("Thin case (high k_ca)")
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("Temperature (C)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(t, fp_thick[:, 0], label="T_p fixed-point")
    axes[1].plot(t, newton_thick[:, 0], "--", label="T_p newton")
    axes[1].plot(t, fp_thick[:, 1], ":", label="T_c fixed-point")
    axes[1].plot(t, newton_thick[:, 1], "-.", label="T_c newton")
    axes[1].set_title("Thick case (low k_ca)")
    axes[1].set_xlabel("time (s)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
