import numpy as np


# Foreign input price as a function of disruption state
def foreign_price(d: int, p_L: float, p_H: float) -> float:
    return p_H if d == 1 else p_L


# Closed-form static policy for domestic sourcing
def x_policy(K: float, p_L: float, p_H: float, d: int, c_d: float) -> float:
    pf = foreign_price(d, p_L, p_H)
    return min(K, pf / c_d)


# One-period expected profit after substituting in the optimal x*(K,d)
def flow_utility(
    K: float,
    d: int,
    s: float,
    A: float,
    c_d: float,
    gamma: float,
    p_L: float,
    p_H: float,
) -> float:
    x = x_policy(K, p_L, p_H, d, c_d)
    pf = foreign_price(d, p_L, p_H)
    return A - 0.5 * c_d * x**2 - pf * (1.0 - x) - 0.5 * gamma * s**2


# Markov transition probabilities for d in {0,1}
def transition_matrix(pi_01: float, pi_10: float) -> np.ndarray:
    return np.array([
        [1 - pi_01, pi_01],
        [pi_10, 1 - pi_10],
    ])


# Compute E[V(K',d')] using linear interpolation over K_grid.
# V has shape (nK, 2), with second dimension corresponding to d=0,1
def expected_continuation_value(
    K_next: float, d: int, V: np.ndarray, K_grid: np.ndarray, P: np.ndarray
) -> float:
    EV = 0.0
    for d_next in (0, 1):
        V_interp = np.interp(K_next, K_grid, V[:, d_next])
        EV += P[d, d_next] * V_interp
    return EV


# Policy improvement step for PFI.
def improve_policy(
    V: np.ndarray,
    K_grid: np.ndarray,
    s_grid: np.ndarray,
    beta: float,
    delta: float,
    psi: float,
    A: float,
    c_d: float,
    gamma: float,
    p_L: float,
    p_H: float,
    pi_01: float,
    pi_10: float,
):
    V_imp = np.empty_like(V)
    s_policy = np.empty_like(V)
    x_pol = np.empty_like(V)
    Kp_pol = np.empty_like(V)

    P = transition_matrix(pi_01, pi_10)
    K_min = K_grid[0]
    K_max = K_grid[-1]

    for i, K in enumerate(K_grid):
        for d in (0, 1):
            best_val = -np.inf
            best_s = 0.0
            best_x = 0.0
            best_Kp = K_min
            x_star = x_policy(K, p_L, p_H, d, c_d)

            for s in s_grid:
                K_next = (1 - delta) * K + psi * s
                K_next = min(max(K_next, K_min), K_max)
                u = flow_utility(K, d, s, A, c_d, gamma, p_L, p_H)
                cont = expected_continuation_value(K_next, d, V, K_grid, P)
                val = u + beta * cont

                if val > best_val:
                    best_val = val
                    best_s = s
                    best_x = x_star
                    best_Kp = K_next

            V_imp[i, d] = best_val
            s_policy[i, d] = best_s
            x_pol[i, d] = best_x
            Kp_pol[i, d] = best_Kp

    return V_imp, s_policy, x_pol, Kp_pol


# Policy evaluation step: keep policy fixed and update V.
def evaluate_policy(
    V_start: np.ndarray,
    K_grid: np.ndarray,
    s_policy: np.ndarray,
    beta: float,
    A: float,
    c_d: float,
    gamma: float,
    p_L: float,
    p_H: float,
    pi_01: float,
    pi_10: float,
    x_pol: np.ndarray,
    Kp_pol: np.ndarray,
    eval_iter: int,
) -> np.ndarray:
    V_eval = V_start.copy()
    P = transition_matrix(pi_01, pi_10)

    for _ in range(eval_iter):
        V_new = np.empty_like(V_eval)
        for i, K in enumerate(K_grid):
            for d in (0, 1):
                s = s_policy[i, d]
                x = x_pol[i, d]
                K_next = Kp_pol[i, d]
                pf = foreign_price(d, p_L, p_H)
                u = A - 0.5 * c_d * x**2 - pf * (1.0 - x) - 0.5 * gamma * s**2
                cont = expected_continuation_value(K_next, d, V_eval, K_grid, P)
                V_new[i, d] = u + beta * cont
        V_eval = V_new

    return V_eval


# Solve the model using policy function iteration (Howard improvement).
def solve_model(
    K_grid: np.ndarray,
    s_grid: np.ndarray,
    beta: float,
    delta: float,
    psi: float,
    A: float,
    c_d: float,
    gamma: float,
    p_L: float,
    p_H: float,
    pi_01: float,
    pi_10: float,
    tol: float = 1e-9,
    max_iter: int = 4000,
    policy_eval_iter: int = 25,
    verbose: bool = True,
    print_every: int = 25,
):
    V = np.zeros((len(K_grid), 2))
    diff = np.inf
    it = 0

    s_policy = np.zeros_like(V)
    x_pol = np.zeros_like(V)
    Kp_pol = np.zeros_like(V)

    while diff > tol and it < max_iter:
        V_imp, s_policy, x_pol, Kp_pol = improve_policy(
            V=V,
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
        )

        V_new = evaluate_policy(
            V_start=V_imp,
            K_grid=K_grid,
            s_policy=s_policy,
            beta=beta,
            A=A,
            c_d=c_d,
            gamma=gamma,
            p_L=p_L,
            p_H=p_H,
            pi_01=pi_01,
            pi_10=pi_10,
            x_pol=x_pol,
            Kp_pol=Kp_pol,
            eval_iter=policy_eval_iter,
        )

        diff = np.max(np.abs(V_new - V))
        V = V_new
        it += 1

        if verbose and (it % print_every == 0 or it == 1):
            print(f"PFI outer iter {it:4d} | sup diff = {diff:.10f}")

    if verbose:
        print(f"PFI converged in {it} outer iterations with sup diff = {diff:.10f}")

    return {
        "V": V,
        "s_policy": s_policy,
        "x_policy": x_pol,
        "Kp_policy": Kp_pol,
        "iteration": it,
        "diff": diff,
    }
