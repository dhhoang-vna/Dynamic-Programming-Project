import numpy as np
import matplotlib.pyplot as plt
import time

try:
    from DynamicProgramming.functions_pfi import solve_model
except ModuleNotFoundError:
    from functions_pfi import solve_model


# Defaults for interactive runs in VS Code.
FAST_RUN = True
SHOW_PLOTS = False


def main():
    #-------------------
    # Model parameters
    #-------------------
    beta = 0.95
    delta = 0.1
    psi = 0.25
    A = 2.0
    c_d = 4.0
    gamma = 2.0
    p_L = 0.6
    p_H = 1.6
    pi_01 = 0.1
    pi_10 = 0.4

    # Grids (larger than VFI defaults)
    nK = 81 if FAST_RUN else 301
    ns = 81 if FAST_RUN else 301
    K_grid = np.linspace(0, 1, nK)
    s_max = 2.0  # Choose a sufficiently large s_max for the optimal s to lie in-grid.
    s_grid = np.linspace(0, s_max, ns)

    #-----------------
    # Solve the model
    #-----------------
    tol = 1e-7 if FAST_RUN else 1e-9
    max_iter = 600 if FAST_RUN else 4000
    policy_eval_iter = 20 if FAST_RUN else 40
    print_every = 10 if FAST_RUN else 25
    print(
        f"Starting PFI solve_model with nK={nK}, ns={ns}, "
        f"tol={tol}, max_iter={max_iter}, policy_eval_iter={policy_eval_iter}"
    )

    t0 = time.perf_counter()
    results = solve_model(
        K_grid=K_grid,
        s_grid=s_grid,
        beta=beta,
        delta=delta,
        psi=psi,
        A=A,
        c_d=c_d,
        gamma=gamma,
        p_L=p_L,
        p_H=p_H,
        pi_01=pi_01,
        pi_10=pi_10,
        tol=tol,
        max_iter=max_iter,
        policy_eval_iter=policy_eval_iter,
        verbose=True,
        print_every=print_every,
    )
    print(f"PFI solve finished in {time.perf_counter() - t0:.2f} seconds")

    V = results["V"]
    s_policy = results["s_policy"]
    x_policy = results["x_policy"]
    Kp_policy = results["Kp_policy"]

    #-------
    # Plot
    #-------
    # Value function
    plt.figure(figsize=(8, 5))
    plt.plot(K_grid, V[:, 0], label="No disruption (d=0)")
    plt.plot(K_grid, V[:, 1], label="Disruption (d=1)")
    plt.xlabel("Domestic capacity K")
    plt.ylabel("Value function V(K,d)")
    plt.title("Value function (PFI)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("value_function_pfi.png", dpi=300)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

    # Subsidy policy function
    plt.figure(figsize=(8, 5))
    plt.plot(K_grid, s_policy[:, 0], label="No disruption (d=0)")
    plt.plot(K_grid, s_policy[:, 1], label="Disruption (d=1)")
    plt.xlabel("Domestic capacity K")
    plt.ylabel("Optimal subsidy s*(K,d)")
    plt.title("Subsidy policy function (PFI)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("subsidy_policy_pfi.png", dpi=300)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

    # Domestic sourcing policy function
    plt.figure(figsize=(8, 5))
    plt.plot(K_grid, x_policy[:, 0], label="No disruption (d=0)")
    plt.plot(K_grid, x_policy[:, 1], label="Disruption (d=1)")
    plt.xlabel("Domestic capacity K")
    plt.ylabel("Optimal domestic sourcing x*(K,d)")
    plt.title("Domestic sourcing policy function (PFI)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("domestic_sourcing_policy_pfi.png", dpi=300)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

    # Next-period capacity policy function
    plt.figure(figsize=(8, 5))
    plt.plot(K_grid, Kp_policy[:, 0], label="No disruption (d=0)")
    plt.plot(K_grid, Kp_policy[:, 1], label="Disruption (d=1)")
    plt.xlabel("Domestic capacity K")
    plt.ylabel("Optimal next-period capacity K'(K,d)")
    plt.title("Next-period capacity policy function (PFI)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("next_period_capacity_policy_pfi.png", dpi=300)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

    #------------------
    # Console summary
    #------------------
    print("\nSample policy values (PFI):")
    sample_indices = np.linspace(0, nK - 1, 5, dtype=int).tolist()
    for idx in sample_indices:
        K = K_grid[idx]
        print(
            f"K={K:.2f} | "
            f"s*(d=0)={s_policy[idx,0]:.3f}, x*(d=0)={x_policy[idx,0]:.3f} | "
            f"s*(d=1)={s_policy[idx,1]:.3f}, x*(d=1)={x_policy[idx,1]:.3f}"
        )


if __name__ == "__main__":
    main()
