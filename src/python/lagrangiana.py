"""
Lagrangian relaxation heuristic for the HDM/HVM waste collection
location problem.
"""

from __future__ import annotations
import warnings

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from dataclasses import dataclass

from instancia import Instance, ModelParameters,compute_demand


@dataclass(frozen=False)
class Multipliers:
    """Lagrange multipliers for the problem instance."""

    mu: np.ndarray       # shape (n_j, n_k)
    lbd: np.ndarray      # shape (n_j)
    nu: np.ndarray       # shape (n_j, n_k)

@dataclass(frozen=False)
class LRSolution:
    """Solution of the Lagrangian Relaxation subproblem."""

    z:         np.ndarray  # shape (n_j,)       bool  — puntos abiertos
    x:         np.ndarray  # shape (n_j, n_k)   int   — nº contenedores
    y_assign:  np.ndarray  # shape (n_i, n_k)   int   — j asignado por edificio y tipo
    w:         np.ndarray  # shape (n_j, n_k)   bool  — hay contenedor tipo k en j
    obj_plrz:  float                          
    obj_plrx:  float                          
    obj_plryw: float       


@dataclass(frozen=False)
class FeasibleSolution:
    """Feasible solution for the original problem."""

    z:         np.ndarray  # shape (n_j,)
    x:         np.ndarray  # shape (n_j, n_k)
    y_assign:  np.ndarray  # shape (n_i, n_k)
    w:         np.ndarray  # shape (n_j, n_k)
    cost:      float

@dataclass(frozen=False)
class LagrangianResult:
    """Result of the Lagrangian Relaxation process."""

    best_feasible:  FeasibleSolution
    best_lb:        float
    best_ub:        float
    gap:            float
    n_iterations:   int
    lb_history:     list[float]
    ub_history:     list[float]


def precompute_valid_candidates(
    instance: Instance,
) -> list[list[list[int]]]:
    """
    For each building i and waste type k, precompute the sorted list
    of candidate indices j that satisfy the NIMBY and coverage
    constraints: r0 <= d_ij <= r_k.

    Returns valid_candidates such that:
        valid_candidates[i_idx][k] = [d_ij, j_idx] sorted by d_ij ASC
    """

    n_buildings = len(instance.I)
    n_waste_types = len(instance.K)

    temp: list[list[list[tuple[float, int]]]] = [[[] for _ in range(n_waste_types)] for _ in range(n_buildings)]

    for j_idx in instance.dij:
        for i_idx in instance.dij[j_idx]:
            d_ij = instance.dij[j_idx][i_idx]
            for k in instance.K:
                if instance.params.nimby_distance <= d_ij <= instance.params.coverage_radius[k]:
                    temp[i_idx][k].append((d_ij, j_idx))
    
    # Sort candidates for each (i, k) by distance
    valid_candidates: list[list[list[int]]] = [[[] for _ in range(n_waste_types)] for _ in range(n_buildings)]
    
    for i_idx in range(n_buildings):
        for k in range(n_waste_types):
            temp[i_idx][k].sort(key=lambda x: x[0])  # Sort by distance
            valid_candidates[i_idx][k] = [j_idx for _, j_idx in temp[i_idx][k]]


    return valid_candidates


def solve_plr_yw(
    instance: Instance,
    multipliers: Multipliers,
    valid_candidates: list[list[list[int]]],
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Solve the assignment subproblem P_LR_yw.

    Sets w_jk = 1 for all j, k (always optimal: negative coefficients).
    Assigns each building i to the nearest valid candidate j for each
    waste type k, respecting nearest-allocation (constraint 9).
    Returns -1 in y_assign where no valid candidate exists.

    Returns:
        y_assign  : np.ndarray shape (n_i, n_k) int   — assigned j_idx
        w         : np.ndarray shape (n_j, n_k) bool  — all True
        obj_plryw : float — objective value of P_LR_yw
    """

    n_buildings = len(instance.I)
    n_candidates = len(instance.J)
    n_waste_types = len(instance.K)

    w = np.ones((n_candidates, n_waste_types), dtype=bool)
    y_assign = np.full((n_buildings, n_waste_types), -1, dtype=int)

    positive_term = 0.0
    
    for i_idx in range(n_buildings):
        for k in range(n_waste_types):
            if valid_candidates[i_idx][k]:
                y_assign[i_idx, k] = valid_candidates[i_idx][k][0]
                q_ik = compute_demand(instance.I[i_idx].h_i, instance.params, k)
                positive_term += multipliers.mu[y_assign[i_idx, k], k] * q_ik

    negative_term = instance.params.max_bins * multipliers.nu.sum()
    obj_plryw = positive_term - negative_term

    return y_assign, w, obj_plryw

            

def solve_plr_x_greedy(
    instance: Instance,
    multipliers: Multipliers,
) -> tuple[np.ndarray, float]:
    """
    Solve the bin placement subproblem P_LR_x.

    For each (j, k), computes the modified cost coefficient
    (c_k + lambda_j + nu_jk - mu_jk * Q_k) and determines the
    optimal integer number of bins x_jk using a greedy algorithm
    sorted by coefficient ascending, satisfying the effective
    inequality (26): total installed capacity >= total demand per type.

    Returns:
        x        : np.ndarray shape (n_j, n_k) int — number of bins
        obj_plrx : float — objective value of P_LR_x
    """

    n_candidates = len(instance.J)
    n_waste_types = len(instance.K)
    n_buildings = len(instance.I)
    total_demand = np.zeros(n_waste_types)
    for i_idx in range(n_buildings):
        for k in range(n_waste_types):
            total_demand[k] += compute_demand(instance.I[i_idx].h_i, instance.params, k)
    
    # bin_cost_array[k] = c_k
    bin_cost_arr = np.array([instance.params.bin_cost[k] for k in instance.K])

    # bin_cap_arr[k] = capacity of bin type k (Q_k)
    bin_cap_arr = np.array([instance.params.bin_capacity[k] for k in instance.K])

    # coef[j, k] = c_k + lambda_j + nu_jk - mu_jk * Q_k - dimension (n_j, n_k)
    coef = (bin_cost_arr 
            + multipliers.lbd[:, np.newaxis] 
            + multipliers.nu 
            - multipliers.mu * bin_cap_arr)
    
    x = np.zeros((n_candidates, n_waste_types), dtype=int)
    for k in range(n_waste_types):
        # Ordenamos los candidatos por coeficiente ascendente para el tipo de residuo k
        sorted_index = np.argsort(coef[:, k])       # indices de j de menor a mayor coeficiente
        installed_capacity  = 0.0

        for j_idx in sorted_index:
            if coef[j_idx, k] < 0:
                # Coeficiente negativo: cada contenedor adicional reduce el objetivo.
                # Instalamos max_bins (cota de caja derivada de restricción 5)
                # para explotar el coste negativo al máximo permitido.
                x[j_idx, k] = instance.params.max_bins
                installed_capacity += instance.params.max_bins * bin_cap_arr[k]
            elif installed_capacity < total_demand[k]:
                # Coeficiente positivo: cada contenedor adicional aumenta el objetivo.
                # Instalamos solo lo necesario para cubrir la demanda restante.
                needed = total_demand[k] - installed_capacity
                n_bins = int(np.ceil(needed / bin_cap_arr[k]))
                n_bins = min(n_bins, instance.params.max_bins)  # cota de caja
                x[j_idx, k] = n_bins
                installed_capacity += n_bins * bin_cap_arr[k]
            else:
                break  # demanda cubierta y coeficientes positivos: paramos

    # Calcular el valor objetivo de P_LR_x
    obj_plrx: float = float(np.sum(coef * x))

    return x, obj_plrx


def solve_plr_x(
    instance: Instance,
    multipliers: Multipliers,
) -> tuple[np.ndarray, float]:
    """
    Solve P_LR_x with Gurobi (exact).
    Returns x (n_j x n_k) and objective value.
    """

    # ── BLOQUE 1: Extraer dimensiones ────────────────────
    n_candidates = len(instance.J)
    n_waste_types = len(instance.K)
    n_buildings = len(instance.I)

    # ── BLOQUE 2: Calcular demanda total por tipo ────────
    total_demand = np.zeros(n_waste_types)
    for i_idx in range(n_buildings):
        for k in range(n_waste_types):
            total_demand[k] += compute_demand(instance.I[i_idx].h_i, instance.params, k)

    # ── BLOQUE 3: Calcular coeficientes residuos  ───────
    # bin_cost_array[k] = c_k -> dimensión (n_k,)
    bin_cost_arr = np.array([instance.params.bin_cost[k] for k in instance.K])

    # bin_cap_arr[k] = capacity of bin type k (Q_k)
    bin_cap_arr = np.array([instance.params.bin_capacity[k] for k in instance.K])

    # coef[j, k] = c_k + lambda_j + nu_jk - mu_jk * Q_k - dimension (n_j, n_k)
    coef = (bin_cost_arr 
            + multipliers.lbd[:, np.newaxis] 
            + multipliers.nu 
            - multipliers.mu * bin_cap_arr)
    
    # ── BLOQUE 4: Construir modelo Gurobi ───────────────
    # crear modelo silencioso
    model = gp.Model("P_LR_x")
    model.Params.OutputFlag = 0
    model.Params.Threads    = 1

    # variables x[j, k] enteras no negativas
    x = model.addVars(n_candidates, n_waste_types,
                       vtype=GRB.INTEGER, lb=0, ub=instance.params.max_bins, name="x")

    # objetivo: min Σⱼ Σₖ coef[j,k] · x[j,k]
    model.setObjective(
        gp.quicksum(coef[j, k] * x[j, k] for j in range(n_candidates) for k in range(n_waste_types)),
        sense=GRB.MINIMIZE
    )

    # restricción (26): Σⱼ Q_k · x[j,k] ≥ D_k,  ∀k
    for k in range(n_waste_types):
        model.addConstr(
            gp.quicksum(x[j, k] * bin_cap_arr[k] for j in range(n_candidates)) >= total_demand[k]
        )

    # desigualdad válida: Σₖ x[j,k] ≤ N_j,       ∀j
    for j in range(n_candidates):
        model.addConstr(
            gp.quicksum(x[j, k] for k in range(n_waste_types)) <= instance.params.max_bins
        )
    

    # ── BLOQUE 5: Resolver y extraer ─────────────────────
    model.optimize()

    # verificar status == OPTIMAL
    if model.Status != GRB.OPTIMAL:
        raise ValueError(f"Gurobi no encontró solución óptima para P_LR_x (status {model.Status})")
    
    x_sol = np.zeros((n_candidates, n_waste_types), dtype=int)
    for j in range(n_candidates):
        for k in range(n_waste_types):
            x_sol[j, k] = int(round(x[j, k].X))
    obj_plrx = model.ObjVal

    return x_sol, obj_plrx
    


def solve_plr_z(
    instance: Instance,
    multipliers: Multipliers,
    valid_candidates: list[list[list[int]]],
) -> tuple[np.ndarray, float]:
    """
    Solve the facility location subproblem P_LR_z.

    Determines which candidate points to open minimising
    (C_j - N_j * lambda_j) * z_j subject to the effective
    inequality (23): for every building i and waste type k,
    at least one valid candidate must be open.

    Uses Gurobi if available, otherwise falls back to a greedy
    set-covering heuristic.

    Returns:
        z        : np.ndarray shape (n_j,) bool — open candidates
        obj_plrz : float — objective value of P_LR_z
    """

    # Fase 1: Calcular Coeficientes
    n_candidates = len(instance.J)
    n_buildings = len(instance.I)
    n_waste_types = len(instance.K)

    coef = instance.params.opening_cost - multipliers.lbd * instance.params.max_bins


    # Fase 2: Construir el modelo Gurobi
    model = gp.Model("P_LR_z")
    model.Params.OutputFlag = 0  # Silenciar salida de Gurobi

    z_vars = model.addVars(n_candidates, vtype=gp.GRB.BINARY, name="z")

    model.setObjective(
        gp.quicksum(coef[j] * z_vars[j] for j in range(n_candidates)),
        sense=gp.GRB.MINIMIZE
    )

    # Fase 3: Restricciones de cobertura (23)
    for i_idx in range(n_buildings):
        for k in range(n_waste_types):
            if valid_candidates[i_idx][k]:  # Solo agregar restricción si hay candidatos válidos
                model.addConstr(
                    gp.quicksum(z_vars[j] for j in valid_candidates[i_idx][k]) >= 1
                )
    
    # Fase 4: Resolver y extraer solución
    model.optimize()

    z_sol = np.array([z_vars[j].X > 0.5 for j in range(n_candidates)], dtype=bool)
    obj_plrz = model.ObjVal

    return z_sol, obj_plrz


def repair_solution(
    instance: Instance,
    z: np.ndarray,
    valid_candidates: list[list[list[int]]],
) -> FeasibleSolution:
    """
    Build a feasible HDM solution from the relaxed location
    decisions z. Reassigns buildings to the nearest open
    candidate, computes bin requirements from actual assigned
    demand, and evaluates the real HDM objective cost.

    Returns a FeasibleSolution with z, x, y_assign, w and cost.
    """

    n_buildings = len(instance.I)
    n_candidates = len(instance.J)
    n_waste_types = len(instance.K)

    # ── Paso 1: y_assign — reasignar al más cercano ABIERTO ────
    y_assign = np.full((n_buildings, n_waste_types), -1, dtype=int)

    for i_idx in range(n_buildings):
        for k in range(n_waste_types):
            for j_idx in valid_candidates[i_idx][k]:
                if z[j_idx]:
                    y_assign[i_idx, k] = j_idx
                    break
    

    # ── Paso 2: x — calcular contenedores por demanda real ─────
    demand_at = np.zeros((n_candidates, n_waste_types))
    for i_idx in range(n_buildings):
        for k in range(n_waste_types):
            j = y_assign[i_idx, k]
            if j != -1:
                demand_at[j, k] += compute_demand(instance.I[i_idx].h_i, instance.params, k)

    x = np.zeros((n_candidates, n_waste_types), dtype=int)
    for j_idx in range(n_candidates):
        for k_idx in range(n_waste_types):
            if demand_at[j_idx, k_idx] > 0:
                x[j_idx, k_idx] = int(np.ceil(demand_at[j_idx, k_idx] / instance.params.bin_capacity[k_idx]))
    
    # ── Paso 3: verificar restricción (5) — límite físico ──────
    for j_idx in range(n_candidates):
        if z[j_idx]:
            if np.sum(x[j_idx, :]) > instance.params.max_bins:
                warnings.warn(
                    f"Candidate {j_idx} exceeds bin limit with "
                    f"{int(np.sum(x[j_idx, :]))} bins assigned.",
                    stacklevel=2,
                )

    
    # ── Paso 4: w — derivar de x ─────────────────────────────────
    w = (x > 0).astype(bool)

    # ── Paso 5: calcular coste real de la solución HDM ───────────
    fixed_cost = np.sum(z * instance.params.opening_cost)
    variable_cost = np.sum(x * np.array([instance.params.bin_cost[k] for k in instance.K]))
    cost = fixed_cost + variable_cost

    return FeasibleSolution(z=z, x=x, y_assign=y_assign, w=w, cost=cost)



def compute_subgradient(
    instance: Instance,
    z: np.ndarray,
    x: np.ndarray,
    y_assign: np.ndarray,
    w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the subgradient vector φ = (φ_μ, φ_λ, φ_ν) measuring
    the violation of the relaxed constraints (4), (5) and (8).

    Returns:
        phi_mu  : np.ndarray shape (n_j, n_k) — capacity violation
        phi_lbd : np.ndarray shape (n_j,)     — spatial violation
        phi_nu  : np.ndarray shape (n_j, n_k) — presence violation
    """

    n_buildings = len(instance.I)
    n_candidates = len(instance.J)
    n_waste_types = len(instance.K)

    phi_mu = np.zeros((n_candidates, n_waste_types))
    phi_lbd = np.zeros(n_candidates)
    phi_nu = np.zeros((n_candidates, n_waste_types))

    # Bloque 1 — φ_μ[j,k]: violación de capacidad
    for i_idx in range(n_buildings):
        for k in range(n_waste_types):
            j = y_assign[i_idx, k]
            if j != -1:
                q_ik = compute_demand(instance.I[i_idx].h_i, instance.params, k)
                phi_mu[j, k] += q_ik
                
    phi_mu -= x * np.array([instance.params.bin_capacity[k] for k in instance.K])

    # Bloque 2 — φ_λ[j]: violación espacial
    for j_idx in range(n_candidates):
        for k in range(n_waste_types):
            phi_lbd[j_idx] += x[j_idx, k]
    phi_lbd -= z * instance.params.max_bins

    # Bloque 3 — φ_ν[j,k]: violación de presencia
    phi_nu = x.astype(float) - instance.params.max_bins * w.astype(float)

    return phi_mu, phi_lbd, phi_nu



def compute_step_length(
    theta: float,
    upper_bound: float,
    lower_bound: float,
    phi_mu: np.ndarray,
    phi_lbd: np.ndarray,
    phi_nu: np.ndarray,
) -> float:
    """
    Compute the step length σ for subgradient update (formula 30).
    Returns 0.0 if the subgradient norm is zero.
    """

    norm_squared = np.sum(phi_mu**2) + np.sum(phi_lbd**2) + np.sum(phi_nu**2)
    if norm_squared == 0:
        return 0.0

    return theta * abs(upper_bound - lower_bound) / norm_squared


def update_multipliers(
    multipliers: Multipliers,
    phi_mu: np.ndarray,
    phi_lbd: np.ndarray,
    phi_nu: np.ndarray,
    step_length: float,
) -> Multipliers:
    """
    Update Lagrange multipliers using subgradient direction (formula 29).
    Projects onto the non-negative orthant with max{0, ...}.
    Returns a NEW Multipliers object — does not modify the input.
    """

    new_mu = np.maximum(0, multipliers.mu + step_length * phi_mu)
    new_lbd = np.maximum(0, multipliers.lbd + step_length * phi_lbd)
    new_nu = np.maximum(0, multipliers.nu + step_length * phi_nu)

    return Multipliers(mu=new_mu, lbd=new_lbd, nu=new_nu)



def solve_lagrangian(
    instance: Instance,
    max_iterations: int = 2000,
    gap_tolerance: float = 0.01,
    no_improve_limit: int = 50,
    verbose: bool = True,
    print_every: int = 100,
) -> LagrangianResult:
    """
    Solve the HDM using Lagrangian relaxation with subgradient
    optimization, as described in Li et al. (2026), Section 4.

    Implements the 5-step algorithm:
      Step 1: Initialize multipliers, theta, bounds
      Step 2: Solve P_LR (three subproblems) → lower bound
      Step 3: Compute subgradient, repair → upper bound
      Step 4: Update multipliers, reduce theta if stalled
      Step 5: Check stopping criterion (gap or max iterations)

    Returns a LagrangianResult with the best feasible solution,
    bounds, gap, iteration count and convergence history.
    """

    # ── Step 1: Initialize ──────────────────────────────────────
    n_j = instance.n_candidates
    n_k = instance.n_waste_types

    multipliers = Multipliers(
        mu=np.zeros((n_j, n_k)),
        lbd=np.zeros(n_j),
        nu=np.zeros((n_j, n_k)),
    )

    theta = 2.0
    best_lb = 0.0
    best_ub = float('inf')
    best_feasible: FeasibleSolution | None = None

    lb_history = []
    ub_history = []

    # Precómputo de candidatos válidos para cada edificio y tipo de residuo
    valid_candidates = precompute_valid_candidates(instance)

    # contador de iteraciones sin mejora de la cota inferior
    iters_without_improvement = 0


    # ── Step 2-5: Iterar hasta convergencia ─────────────────────
    for iteration in range(max_iterations):

        # ── Step 2: Solve P_LR(γᵗ) → lower bound ──────────────
        z, obj_plrz = solve_plr_z(instance, multipliers, valid_candidates)
        x, obj_plrx = solve_plr_x(instance, multipliers)
        y_assign, w, obj_plryw = solve_plr_yw(instance, multipliers, valid_candidates)

        current_lb = obj_plrz + obj_plrx + obj_plryw

        if current_lb > best_lb:
            best_lb = current_lb
            iters_without_improvement = 0
        else:
            iters_without_improvement += 1

        lb_history.append(best_lb)


        # ── Step 3: Subgradient + feasibility → upper bound ────
        phi_mu, phi_lbd, phi_nu = compute_subgradient(
            instance, z, x, y_assign, w
        )

        feasible = repair_solution(instance, z, valid_candidates)

        if feasible.cost < best_ub:
            best_ub = feasible.cost
            best_feasible = feasible
        
        ub_history.append(best_ub)


        # ── Step 5: Stopping criterion ─────────────────────────
        # (checked before Step 4 to avoid unnecessary update)
        gap = (best_ub - best_lb) / best_ub if best_ub > 0 else 0.0
        if gap <= gap_tolerance:
            break

        # ── Progress report ────────────────────────────────────
        if verbose and (iteration % print_every == 0 or gap <= gap_tolerance):
            n_neg_coefs = int(np.sum(
                (np.array([instance.params.bin_cost[k] for k in instance.K])
                 + multipliers.lbd[:, np.newaxis]
                 + multipliers.nu
                 - multipliers.mu * np.array([instance.params.bin_capacity[k] for k in instance.K]))
                < 0
            ))
            print(
                f"  iter {iteration:4d} | "
                f"curLB {current_lb:12.1f} | "
                f"bestLB {best_lb:12.1f} | "
                f"UB {best_ub:12.1f} | "
                f"gap {gap:7.2%} | "
                f"θ {theta:.4f} | "
                f"neg_coefs {n_neg_coefs:4d}"
            )


        # ── Step 4: Update multipliers ─────────────────────────
        step_length = compute_step_length(
            theta, best_ub, current_lb, phi_mu, phi_lbd, phi_nu
        )

        multipliers = update_multipliers(
            multipliers, phi_mu, phi_lbd, phi_nu, step_length
        )

        if iters_without_improvement >= no_improve_limit:
            theta /= 2
            iters_without_improvement = 0

    
    # ── Build result ───────────────────────────────────────
    return LagrangianResult(
        best_feasible=best_feasible,
        best_lb=best_lb,
        best_ub=best_ub,
        gap=gap,
        n_iterations=iteration + 1,
        lb_history=lb_history,
        ub_history=ub_history,
    )


if __name__ == "__main__":
    from instancia import load_instance
    import time

    print("=" * 65)
    print("  Lagrangian Relaxation — HDM")
    print("=" * 65)

    inst = load_instance("data/processed/instancia_laguna.json")
    print(f"  Instance: {inst.study_case}")
    print(f"  Buildings: {inst.n_buildings}  Candidates: {inst.n_candidates}")
    print(f"  Waste types: {inst.n_waste_types}")
    print("-" * 65)

    start = time.time()
    result = solve_lagrangian(inst, max_iterations=100, print_every=1)
    elapsed = time.time() - start

    print("-" * 65)
    print(f"  Completed in {result.n_iterations} iterations ({elapsed:.1f}s)")
    print(f"  Best LB:  {result.best_lb:,.1f}")
    print(f"  Best UB:  {result.best_ub:,.1f}")
    print(f"  Gap:      {result.gap:.2%}")
    print(f"  Points open: {int(np.sum(result.best_feasible.z))}")
    print(f"  Total bins:  {int(np.sum(result.best_feasible.x))}")
    print("=" * 65)