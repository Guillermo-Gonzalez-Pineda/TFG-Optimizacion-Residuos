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
    demand_matrix: np.ndarray,
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
                q_ik = demand_matrix[i_idx, k]
                positive_term += multipliers.mu[y_assign[i_idx, k], k] * q_ik / instance.params.bin_capacity[k]

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

    # coef[j, k] = c_k + lambda_j + nu_jk - mu_jk  (mu in €/bin, normalized)
    coef = (bin_cost_arr
            + multipliers.lbd[:, np.newaxis]
            + multipliers.nu
            - multipliers.mu)

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


def build_plr_x_model(
    instance: Instance,
    demand_matrix: np.ndarray,
) -> tuple[gp.Model, dict]:
    """
    Build the Gurobi model for P_LR_x ONCE.

    The constraints (demand coverage (26) and N_j bound) do not depend
    on the multipliers, so the model can be reused across iterations:
    only the objective coefficients change. Returns the model and the
    x variables dict so the caller can update Obj and re-optimize.

    Returns:
        model  : gp.Model — built model (objective not yet set)
        x_vars : dict (j, k) -> gurobi var
    """

    # ── BLOQUE 1: Extraer dimensiones ────────────────────
    n_candidates = len(instance.J)
    n_waste_types = len(instance.K)

    # ── BLOQUE 2: Calcular demanda total por tipo ────────
    total_demand = demand_matrix.sum(axis=0)

    # bin_cap_arr[k] = capacity of bin type k (Q_k)
    bin_cap_arr = np.array([instance.params.bin_capacity[k] for k in instance.K])

    # ── BLOQUE 3: Construir modelo Gurobi ───────────────
    # crear modelo silencioso
    model = gp.Model("P_LR_x")
    model.Params.OutputFlag = 0
    model.Params.Threads    = 1

    # variables x[j, k] enteras no negativas
    x_vars = model.addVars(n_candidates, n_waste_types,
                           vtype=GRB.INTEGER, lb=0, ub=instance.params.max_bins, name="x")

    # restricción (26): Σⱼ Q_k · x[j,k] ≥ D_k,  ∀k
    for k in range(n_waste_types):
        model.addConstr(
            gp.quicksum(x_vars[j, k] * bin_cap_arr[k] for j in range(n_candidates)) >= total_demand[k]
        )

    # desigualdad válida: Σₖ x[j,k] ≤ N_j,       ∀j
    for j in range(n_candidates):
        model.addConstr(
            gp.quicksum(x_vars[j, k] for k in range(n_waste_types)) <= instance.params.max_bins
        )

    return model, x_vars


def solve_plr_x(
    model: gp.Model,
    x_vars: dict,
    instance: Instance,
    multipliers: Multipliers,
) -> tuple[np.ndarray, float]:
    """
    Solve P_LR_x with Gurobi (exact), reusing a pre-built model.

    Only updates the objective coefficients (which depend on the
    multipliers) and re-optimizes. Returns x (n_j x n_k) and objective.
    """

    n_candidates = len(instance.J)
    n_waste_types = len(instance.K)

    # bin_cost_array[k] = c_k -> dimensión (n_k,)
    bin_cost_arr = np.array([instance.params.bin_cost[k] for k in instance.K])

    # coef[j, k] = c_k + lambda_j + nu_jk - mu_jk  (mu in €/bin, normalized)
    coef = (bin_cost_arr
            + multipliers.lbd[:, np.newaxis]
            + multipliers.nu
            - multipliers.mu)

    # Actualizar coeficientes del objetivo: min Σⱼ Σₖ coef[j,k] · x[j,k]
    for j in range(n_candidates):
        for k in range(n_waste_types):
            x_vars[j, k].Obj = coef[j, k]

    # ── Resolver y extraer ─────────────────────
    model.optimize()

    # verificar status == OPTIMAL
    if model.Status != GRB.OPTIMAL:
        raise ValueError(f"Gurobi no encontró solución óptima para P_LR_x (status {model.Status})")

    x_sol = np.zeros((n_candidates, n_waste_types), dtype=int)
    for j in range(n_candidates):
        for k in range(n_waste_types):
            x_sol[j, k] = int(round(x_vars[j, k].X))
    obj_plrx = model.ObjVal

    model.reset()

    return x_sol, obj_plrx



def build_plr_z_model(
    instance: Instance,
    valid_candidates: list[list[list[int]]],
) -> tuple[gp.Model, dict]:
    """
    Build the Gurobi model for P_LR_z ONCE.

    The coverage constraints (23) do not depend on the multipliers,
    so the model can be reused across iterations: only the objective
    coefficients change. Returns the model and the z variables dict.

    Returns:
        model  : gp.Model — built model (objective not yet set)
        z_vars : dict j -> gurobi var
    """

    n_candidates = len(instance.J)
    n_buildings = len(instance.I)
    n_waste_types = len(instance.K)

    # Construir el modelo Gurobi
    model = gp.Model("P_LR_z")
    model.Params.OutputFlag = 0  # Silenciar salida de Gurobi
    model.Params.Threads    = 1

    z_vars = model.addVars(n_candidates, vtype=gp.GRB.BINARY, name="z")

    # Restricciones de cobertura (23)
    for i_idx in range(n_buildings):
        for k in range(n_waste_types):
            if valid_candidates[i_idx][k]:  # Solo agregar restricción si hay candidatos válidos
                model.addConstr(
                    gp.quicksum(z_vars[j] for j in valid_candidates[i_idx][k]) >= 1
                )

    return model, z_vars


def solve_plr_z(
    model: gp.Model,
    z_vars: dict,
    instance: Instance,
    multipliers: Multipliers,
) -> tuple[np.ndarray, float]:
    """
    Solve the facility location subproblem P_LR_z, reusing a pre-built
    model.

    Determines which candidate points to open minimising
    (C_j - N_j * lambda_j) * z_j subject to the effective
    inequality (23): for every building i and waste type k,
    at least one valid candidate must be open.

    Only updates the objective coefficients (which depend on the
    multipliers) and re-optimizes.

    Returns:
        z        : np.ndarray shape (n_j,) bool — open candidates
        obj_plrz : float — objective value of P_LR_z
    """

    n_candidates = len(instance.J)

    # Actualizar coeficientes del objetivo: coef[j] = C_j - N_j * lambda_j
    for j in range(n_candidates):
        z_vars[j].Obj = instance.J[j].opening_cost - multipliers.lbd[j] * instance.params.max_bins

    # Resolver y extraer solución
    model.optimize()

    z_sol = np.array([z_vars[j].X > 0.5 for j in range(n_candidates)], dtype=bool)
    obj_plrz = model.ObjVal

    model.reset()

    return z_sol, obj_plrz


def repair_solution(
    instance: Instance,
    z: np.ndarray,
    valid_candidates: list[list[list[int]]],
    demand_matrix: np.ndarray,
) -> FeasibleSolution:
    """
    Build a feasible HDM solution from the relaxed location decisions z.

    Strategy:
      Phase A — Nearest-allocation: assign each building to its closest
                open candidate for each waste type.
      Phase B — Compute bins: calculate x[j,k] = ceil(demand / Q_k)
                from the actual assigned demand.
      Phase C — Repair N_j violations: if any point has more than N_j
                bins, open a nearby closed candidate and redo A+B.
                Opening a new point redistributes demand automatically
                via nearest-allocation (respects constraint 9).

    Returns a FeasibleSolution with z_rep, x, y_assign, w and cost.
    """
    from collections import Counter

    n_buildings = len(instance.I)
    n_candidates = len(instance.J)
    n_waste_types = len(instance.K)
    N_j = instance.params.max_bins  # 8
    z_rep = z.copy()
    C_j = np.array([instance.J[j].opening_cost for j in range(n_candidates)])

    # ────────────────────────────────────────────────────────────
    #  Funciones internas para evitar repetir código
    # ────────────────────────────────────────────────────────────

    def do_nearest_allocation():
        """Phase A: assign each (building, type) to nearest open point."""
        y = np.full((n_buildings, n_waste_types), -1, dtype=int)
        for i in range(n_buildings):
            for k in range(n_waste_types):
                for j in valid_candidates[i][k]:
                    if z_rep[j]:
                        y[i, k] = j
                        break
        return y

    def do_compute_bins(y):
        """Phase B: compute demand and bins from actual assignments."""
        demand = np.zeros((n_candidates, n_waste_types))
        for i in range(n_buildings):
            for k in range(n_waste_types):
                j = y[i, k]
                if j != -1:
                    demand[j, k] += demand_matrix[i, k]

        bins = np.zeros((n_candidates, n_waste_types), dtype=int)
        for j in range(n_candidates):
            for k in range(n_waste_types):
                if demand[j, k] > 0:
                    bins[j, k] = int(np.ceil(
                        demand[j, k] / instance.params.bin_capacity[k]
                    ))
        return bins

    # ────────────────────────────────────────────────────────────
    #  Phase A + B: asignación inicial y cálculo de bins
    # ────────────────────────────────────────────────────────────
    y_assign = do_nearest_allocation()
    x = do_compute_bins(y_assign)

    # ────────────────────────────────────────────────────────────
    #  Phase C: reparar violaciones abriendo puntos nuevos
    # ────────────────────────────────────────────────────────────
    max_repair_iters = 20  # cada iteración abre un punto; 20 es más que suficiente

    for repair_iter in range(max_repair_iters):

        # C1. Encontrar el punto más saturado
        worst_j, worst_excess = None, 0
        for j in range(n_candidates):
            if z_rep[j]:
                excess = int(np.sum(x[j, :])) - N_j
                if excess > worst_excess:
                    worst_j, worst_excess = j, excess

        if worst_j is None:
            break  # ✅ sin violaciones

        # C2. Elegir qué punto abrir: el candidato cerrado que más
        #     edificios absorbería de worst_j.
        #     Recorremos los edificios asignados a worst_j y contamos
        #     cuál es la alternativa cerrada más frecuente.
        alt_counter = Counter()
        for i in range(n_buildings):
            for k in range(n_waste_types):
                if y_assign[i, k] == worst_j:
                    # Buscar su primera alternativa cerrada
                    for alt_j in valid_candidates[i][k]:
                        if alt_j != worst_j and z_rep[alt_j] == 0:
                            alt_counter[alt_j] += 1
                            break

        if not alt_counter:
            # No hay candidatos cerrados → intentar con abiertos
            # (la redistribución puede ayudar igualmente)
            warnings.warn(
                f"repair: no hay candidatos cerrados para aliviar "
                f"j={worst_j} (exceso +{worst_excess})"
            )
            break

        # Abrir el candidato que más edificios absorbería
        best_new_j = alt_counter.most_common(1)[0][0]
        z_rep[best_new_j] = 1

        # C3. Rehacer A + B desde cero con el nuevo z_rep
        #     La nearest-allocation redistribuye automáticamente
        #     → respeta constraint (9)
        y_assign = do_nearest_allocation()
        x = do_compute_bins(y_assign)

    # ── Fase D: búsqueda local — intentar cerrar puntos ────
    improved = True
    while improved:
        improved = False
        best_cost = (
            np.sum(z_rep * C_j)
            + np.sum(x * np.array([instance.params.bin_cost[k] for k in instance.K]))
        )

        for j_candidate in range(n_candidates):
            if not z_rep[j_candidate]:
                continue  # ya está cerrado

            # Intentar cerrar este punto
            z_rep[j_candidate] = 0

            # Reubicar
            y_trial = do_nearest_allocation()

            # ¿Todos los edificios cubiertos?
            if np.any(y_trial == -1):
                z_rep[j_candidate] = 1  # revertir
                continue

            x_trial = do_compute_bins(y_trial)

            # ¿Alguna violación de N_j?
            feasible = True
            for j in range(n_candidates):
                if z_rep[j] and np.sum(x_trial[j, :]) > instance.params.max_bins:
                    feasible = False
                    break

            if not feasible:
                z_rep[j_candidate] = 1  # revertir
                continue

            # ¿El coste mejoró?
            trial_cost = (
                np.sum(z_rep * C_j)
                + np.sum(x_trial * np.array([instance.params.bin_cost[k] for k in instance.K]))
            )

            if trial_cost < best_cost:
                # Aceptar el cierre
                y_assign = y_trial
                x = x_trial
                best_cost = trial_cost
                improved = True
            else:
                z_rep[j_candidate] = 1  # revertir, no mejoró

    # ────────────────────────────────────────────────────────────
    #  Diagnóstico final
    # ────────────────────────────────────────────────────────────
    final_violations = sum(
        1 for j in range(n_candidates)
        if z_rep[j] and np.sum(x[j, :]) > N_j
    )
    if final_violations > 0:
        cost = float('inf')
        warnings.warn(
            f"repair: {final_violations} violaciones NO resueltas "
            f"tras {repair_iter + 1} iteraciones"
        )


    # Fallback: si Gurobi falla, devolver solución heurística
    w = (x > 0).astype(bool)
    fixed_cost = np.sum(z_rep * C_j)
    variable_cost = np.sum(x * np.array([instance.params.bin_cost[k] for k in instance.K]))
    cost = fixed_cost + variable_cost
    return FeasibleSolution(z=z_rep, x=x, y_assign=y_assign, w=w, cost=cost)



def compute_subgradient(
    instance: Instance,
    z: np.ndarray,
    x: np.ndarray,
    y_assign: np.ndarray,
    w: np.ndarray,
    demand_matrix: np.ndarray,
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
                q_ik = demand_matrix[i_idx, k]
                phi_mu[j, k] += q_ik
                
    bin_cap_arr = np.array([instance.params.bin_capacity[k] for k in instance.K])
    phi_mu -= x * bin_cap_arr
    phi_mu /= bin_cap_arr   # normalize to bin-fraction units (mu in €/bin)

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


def solve_fixed_locations(
    instance: Instance,
    z_fixed: np.ndarray,
    valid_candidates: list[list[list[int]]],
    time_limit: float = 60.0,
    allow_closures: bool = False,
) -> FeasibleSolution | None:
    """
    Optimise bin placement and building assignment using Gurobi.

    Two modes, controlled by `allow_closures`:

    - allow_closures=False (default): 'restricted MIP'. The location
      decisions z are FIXED to z_fixed; only x, y, w are optimised.
      Much smaller than the full HDM and solves in seconds.

    - allow_closures=True: z becomes a DECISION VARIABLE warm-started
      with z_fixed (z[j].Start = z_fixed[j]). This is essentially the
      full HDM, but Gurobi departs from the known feasible repair
      solution instead of from scratch, so it can prune unnecessary
      points (close them) to lower the fixed cost.

    Parameters
    ----------
    instance         : Instance with buildings, candidates, params
    z_fixed          : bool array (n_j,) — open points (also the warm-start)
    valid_candidates : [i][k] → sorted list of valid j indices
    time_limit       : seconds before Gurobi stops (default 60)
    allow_closures   : if True, z is a variable warm-started with z_fixed

    Returns
    -------
    FeasibleSolution or None if infeasible.
    """

    n_i = len(instance.I)
    n_j = len(instance.J)
    n_k = len(instance.K)
    params = instance.params

    # ── Conjunto de candidatos según el modo ──────────────
    if allow_closures:
        # z es VARIABLE (warm-start): trabajamos con TODOS los candidatos
        # y todos los válidos por radio, para que Gurobi pueda decidir
        # libremente cuáles cerrar partiendo de z_fixed.
        open_j = list(range(n_j))
        valid_open = valid_candidates
    else:
        # Comportamiento original: z fija, solo los puntos abiertos.
        open_j = [j for j in range(n_j) if z_fixed[j]]

        # Candidatos válidos ABIERTOS para cada (edificio, tipo)
        # Intersección de valid_candidates[i][k] con open_j
        open_set = set(open_j)
        valid_open = [
            [
                [j for j in valid_candidates[i][k] if j in open_set]
                for k in range(n_k)
            ]
            for i in range(n_i)
        ]

    # ── Modelo Gurobi ─────────────────────────────────────
    model = gp.Model("fixed_locations")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = time_limit

    # ── Variables ──────────────────────────────────────────
    # x[j,k]: número de bins tipo k en punto j
    x = model.addVars(
        open_j, range(n_k),
        vtype=GRB.INTEGER, lb=0, name="x"
    )

    # w[j,k]: ¿hay bin tipo k en punto j?
    w = model.addVars(
        open_j, range(n_k),
        vtype=GRB.BINARY, name="w"
    )

    # y[i,j,k]: edificio i asignado a punto j para tipo k
    # Solo para combinaciones válidas (dentro del radio)
    y = {}
    for i in range(n_i):
        for k in range(n_k):
            for j in valid_open[i][k]:
                y[i, j, k] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}_{k}")

    # z[j]: ¿punto abierto? — solo VARIABLE en modo allow_closures
    if allow_closures:
        # z binaria con punto de partida = solución del repair (z_fixed).
        # El .Start hace que Gurobi arranque desde una solución factible
        # conocida en vez de desde cero, acelerando mucho la búsqueda y
        # garantizando que el UB nunca empeore respecto al warm-start.
        z = model.addVars(n_j, vtype=GRB.BINARY, name="z")
        for j in range(n_j):
            z[j].Start = int(z_fixed[j])

    # ── Objetivo ──────────────────────────────────────────
    if allow_closures:
        # z es variable → el coste fijo Σⱼ C_j·z[j] forma parte del
        # objetivo: cerrar puntos innecesarios reduce el coste total.
        model.setObjective(
            gp.quicksum(instance.J[j].opening_cost * z[j] for j in range(n_j))
            + gp.quicksum(
                params.bin_cost[k] * x[j, k]
                for j in open_j for k in range(n_k)
            ),
            GRB.MINIMIZE
        )
    else:
        # Coste fijo es constante (z ya decidido), solo optimizamos bins
        # Lo sumamos al final al extraer el resultado
        model.setObjective(
            gp.quicksum(
                params.bin_cost[k] * x[j, k]
                for j in open_j for k in range(n_k)
            ),
            GRB.MINIMIZE
        )

    # ── (2) Asignación completa ───────────────────────────
    # Cada edificio asignado a exactamente un punto abierto por tipo
    for i in range(n_i):
        for k in range(n_k):
            cands = valid_open[i][k]
            if not cands:
                # Edificio sin cobertura → problema infactible
                model.dispose()
                return None
            model.addConstr(
                gp.quicksum(y[i, j, k] for j in cands) == 1
            )

    # ── (3) Orgánica y resto al mismo punto ───────────────
    for i in range(n_i):
        for j in valid_open[i][0]:
            if (i, j, 1) in y:
                model.addConstr(y[i, j, 0] == y[i, j, 1])

    # ── (4) Capacidad ─────────────────────────────────────
    for j in open_j:
        for k in range(n_k):
            assigned = [i for i in range(n_i) if (i, j, k) in y]
            if assigned:
                model.addConstr(
                    gp.quicksum(
                        compute_demand(instance.I[i].h_i, params, k) * y[i, j, k]
                        for i in assigned
                    ) <= params.bin_capacity[k] * x[j, k]
                )

    # ── (5) Límite físico de bins por punto ───────────────
    if allow_closures:
        # z variable → Σₖ x[j,k] ≤ N_j · z[j]: si z[j]=0 el punto cierra
        # y no admite bins (fuerza x=0), enlazando bins con apertura.
        for j in open_j:
            model.addConstr(
                gp.quicksum(x[j, k] for k in range(n_k)) <= params.max_bins * z[j]
            )
    else:
        # z está fijo a 1 para open_j → simplifica a Σₖ x[j,k] ≤ N_j
        for j in open_j:
            model.addConstr(
                gp.quicksum(x[j, k] for k in range(n_k)) <= params.max_bins
            )

    # ── (8) Relación x-w ─────────────────────────────────
    for j in open_j:
        for k in range(n_k):
            model.addConstr(x[j, k] <= params.max_bins * w[j, k])

    # ── (7) Enlace w-z (solo allow_closures) ──────────────
    # Si el punto está cerrado (z[j]=0) no puede haber presencia de bin:
    # w[j,k] ≤ z[j]. Es imprescindible para que la nearest-allocation (9)
    # —que usa w para bloquear asignaciones a puntos más lejanos— no se
    # apoye en bins fantasma de puntos cerrados.
    if allow_closures:
        for j in range(n_j):
            for k in range(n_k):
                model.addConstr(w[j, k] <= z[j])

    # ── (9) Nearest-allocation ────────────────────────────
    # Si j' está más cerca que j y tiene bin tipo k, i no puede ir a j
    for i in range(n_i):
        for k in range(n_k):
            cands = valid_open[i][k]
            for p in range(1, len(cands)):
                j_far = cands[p]
                for q in range(p):
                    j_close = cands[q]
                    model.addConstr(y[i, j_far, k] + w[j_close, k] <= 1)

    # ── Resolver ──────────────────────────────────────────
    model.optimize()

    if model.SolCount == 0:
        model.dispose()
        return None

    # ── Extraer solución ──────────────────────────────────
    x_sol = np.zeros((n_j, n_k), dtype=int)
    for j in open_j:
        for k in range(n_k):
            x_sol[j, k] = int(round(x[j, k].X))

    w_sol = np.zeros((n_j, n_k), dtype=bool)
    for j in open_j:
        for k in range(n_k):
            w_sol[j, k] = w[j, k].X > 0.5

    y_assign = np.full((n_i, n_k), -1, dtype=int)
    for i in range(n_i):
        for k in range(n_k):
            for j in valid_open[i][k]:
                if y[i, j, k].X > 0.5:
                    y_assign[i, k] = j
                    break

    # Coste total y z resultante según el modo
    if allow_closures:
        # z es variable: la leemos de la solución y el objetivo ya
        # incluye coste fijo + variable, así que ObjVal es el coste total.
        z_sol = np.array([z[j].X > 0.5 for j in range(n_j)], dtype=bool)
        total_cost = float(model.ObjVal)
    else:
        # z fija: coste total = coste fijo (constante) + variable (ObjVal)
        z_sol = z_fixed.copy()
        fixed_cost = float(np.sum(z_fixed * np.array([instance.J[j].opening_cost for j in range(n_j)])))
        variable_cost = float(model.ObjVal)
        total_cost = fixed_cost + variable_cost

    model.dispose()

    return FeasibleSolution(
        z=z_sol,
        x=x_sol,
        y_assign=y_assign,
        w=w_sol,
        cost=total_cost,
    )


def solve_lagrangian(
    instance: Instance,
    max_iterations: int = 2000,
    gap_tolerance: float = 0.01,
    no_improve_limit: int = 200,
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
    best_ub = float(1e9)  # una cota superior grande (p.ej. coste de abrir todo con max_bins)
    best_feasible: FeasibleSolution | None = None

    lb_history = []
    ub_history = []

    # Precómputo de candidatos válidos para cada edificio y tipo de residuo
    valid_candidates = precompute_valid_candidates(instance)

    # Precómputo de la demanda q_ik (constante: no depende de multiplicadores)
    n_i = instance.n_buildings
    demand_matrix = np.zeros((n_i, n_k))
    for i in range(n_i):
        for k in range(n_k):
            demand_matrix[i, k] = compute_demand(instance.I[i].h_i, instance.params, k)

    # Construcción de los modelos Gurobi reutilizables (restricciones fijas)
    model_z, z_vars = build_plr_z_model(instance, valid_candidates)
    model_x, x_vars = build_plr_x_model(instance, demand_matrix)

    # contador de iteraciones sin mejora de la cota inferior
    iters_without_improvement = 0

    # ── Pool de soluciones para multi-polish ────────────────────
    # Guardamos las N mejores soluciones factibles con z DIVERSAS
    # (no solo la mejor) para pulir varias al final con el MIP exacto
    # solve_fixed_locations y quedarnos con el mejor UB resultante.
    # Motivo: dos z con coste de repair parecido pueden dar UBs muy
    # distintos tras el polish, porque el repair es heurístico y el
    # polish reoptimiza exactamente bins/asignaciones sobre cada z.
    top_k_solutions = []  # lista de (cost, z_array)
    MAX_TOP_K = 5

    # ── Control del polish intermedio (mid-polish) ──────────────
    # Pulir la mejor z con el MIP exacto DURANTE el bucle (no solo al
    # final) cuando el UB se estanca. Un mejor UB reduce |UB − LB| en el
    # step size σ = θ·|UB − LB|/‖φ‖², lo que afina la convergencia de las
    # iteraciones restantes. Para no malgastar tiempo, solo se dispara
    # tras 500 iters sin mejora de UB y se desactiva si el polish no
    # mejora; se reactiva en cuanto el UB baje por otra vía.
    ub_stagnation_counter = 0
    ub_last_improved = best_ub
    polish_exhausted = False


    # ── Step 2-5: Iterar hasta convergencia ─────────────────────
    for iteration in range(max_iterations):

        # ── Step 2: Solve P_LR(γᵗ) → lower bound ──────────────
        z, obj_plrz = solve_plr_z(model_z, z_vars, instance, multipliers)
        x, obj_plrx = solve_plr_x(model_x, x_vars, instance, multipliers)
        y_assign, w, obj_plryw = solve_plr_yw(instance, multipliers, valid_candidates, demand_matrix)

        current_lb = obj_plrz + obj_plrx + obj_plryw


        # Después de las 3 líneas solve_plr_*:
        if iteration < 5 or iteration % 50 == 0:
            print(f"    [descomposición iter {iteration}] "
                  f"z={obj_plrz:>10,.1f}  x={obj_plrx:>10,.1f}  "
                  f"yw={obj_plryw:>10,.1f}  total={current_lb:>10,.1f}")


        if current_lb > best_lb:
            best_lb = current_lb
            iters_without_improvement = 0
        else:
            iters_without_improvement += 1

        lb_history.append(best_lb)


        # ── Step 3: Subgradient + feasibility → upper bound ────
        phi_mu, phi_lbd, phi_nu = compute_subgradient(
            instance, z, x, y_assign, w, demand_matrix
        )

        feasible = repair_solution(instance, z, valid_candidates, demand_matrix)

        if feasible.cost < best_ub:
            best_ub = feasible.cost
            best_feasible = feasible

        # ── Guardar z para multi-polish si es DIVERSA ──────────
        # "Diversa" = difiere de TODAS las z ya guardadas en al menos
        # 2 puntos abiertos/cerrados (distancia de Hamming ≥ 2). Así el
        # pool cubre configuraciones realmente distintas y no clones de
        # la misma solución, maximizando la exploración del polish final.
        z_copy = feasible.z.copy()
        is_diverse = all(
            np.sum(z_copy ^ existing_z) >= 2
            for _, existing_z in top_k_solutions
        )
        if is_diverse:
            top_k_solutions.append((feasible.cost, z_copy))
            # Mantener solo las MAX_TOP_K mejores por coste de repair
            top_k_solutions.sort(key=lambda x: x[0])
            top_k_solutions = top_k_solutions[:MAX_TOP_K]

        ub_history.append(best_ub)

        # ── Polish intermedio (mid-polish) ─────────────────────
        # Contador de estancamiento del UB: cuántas iteraciones
        # consecutivas llevamos sin bajar el mejor upper bound.
        if best_ub < ub_last_improved:
            # El UB ha mejorado por la vía heurística (repair): reiniciamos
            # el contador y rehabilitamos el polish (puede volver a aportar
            # sobre la nueva z).
            ub_stagnation_counter = 0
            ub_last_improved = best_ub
            polish_exhausted = False  # UB cambió, permitir polish de nuevo
        else:
            ub_stagnation_counter += 1

        # Cuando el UB lleva 500 iters sin mejorar, lanzamos un polish
        # exacto sobre la mejor z para intentar bajarlo. Un UB menor
        # reduce |UB − LB| en σ = θ·|UB − LB|/‖φ‖², afinando la
        # convergencia de las iteraciones que quedan. Solo se dispara con
        # estancamiento real para no pagar el coste del MIP cada iter.
        if ub_stagnation_counter >= 500 and not polish_exhausted:
            polished = solve_fixed_locations(
                instance, best_feasible.z, valid_candidates, time_limit=120.0
            )
            if polished is not None and polished.cost < best_ub:
                best_ub = polished.cost
                best_feasible = polished
                gap = (best_ub - best_lb) / best_ub if best_ub > 0 else 0.0
                ub_stagnation_counter = 0
                ub_last_improved = best_ub
                # Registrar mejora en historial
                ub_history[-1] = best_ub
                if verbose:
                    print(f"  [mid-polish] UB mejorado: {best_ub:,.1f} (gap {gap:.2%})")
            else:
                # El polish no mejoró: desactivamos hasta que el UB cambie
                # por otra vía (evita repetir un MIP costoso sin ganancia).
                polish_exhausted = True
                if verbose:
                    print(f"  [mid-polish] Sin mejora, desactivado hasta que UB cambie")


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
                 - multipliers.mu)
                < 0
            ))
            n_open = int(np.sum(feasible.z))
            print(
                f"  iter {iteration:4d} | "
                f"curLB {current_lb:12.1f} | "
                f"bestLB {best_lb:12.1f} | "
                f"UB {best_ub:12.1f} | "
                f"gap {gap:7.2%} | "
                f"θ {theta:.4f} | "
                f"neg_coefs {n_neg_coefs:4d} | "
                f"open {n_open:3d}"
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

    # ══════════════════════════════════════════════════════════
    # MULTI-POLISH — después del bucle, antes del return
    # ══════════════════════════════════════════════════════════
    # En lugar de pulir solo la mejor z, pulimos las MAX_TOP_K mejores
    # z diversas guardadas durante el bucle. Cada solve_fixed_locations
    # reoptimiza exactamente bins/asignaciones para una z fija; como
    # distintas z pueden dar UBs muy diferentes tras el polish, probar
    # varias suele encontrar un UB mejor que pulir solo una. El coste
    # extra es acotado: ~MAX_TOP_K × 120s = ~10 min en el peor caso,
    # asumible como paso final único de refinamiento.
    if top_k_solutions and valid_candidates is not None:
        if verbose:
            print(f"  [multi-polish] Puliendo {len(top_k_solutions)} soluciones...")
        for idx, (cost, z_candidate) in enumerate(top_k_solutions):
            polished = solve_fixed_locations(
                instance, z_candidate, valid_candidates, time_limit=120.0
            )
            if polished is not None and polished.cost < best_ub:
                best_ub = polished.cost
                best_feasible = polished
                gap = (best_ub - best_lb) / best_ub if best_ub > 0 else 0.0
                if verbose:
                    print(f"    z#{idx+1} (repair={cost:,.0f}) → polish={polished.cost:,.1f} ✓")
            elif verbose:
                ub_str = f"{polished.cost:,.1f}" if polished else "infeasible"
                print(f"    z#{idx+1} (repair={cost:,.0f}) → polish={ub_str}")

        if verbose:
            print(f"  [multi-polish] Mejor UB: {best_ub:,.1f} (gap {gap:.2%})")

    # ══════════════════════════════════════════════════════════
    # WARM-START POLISH — paso final con cierre de puntos
    # ══════════════════════════════════════════════════════════
    # Resolvemos el HDM completo (z como variable) pero arrancando desde
    # la mejor solución encontrada (z_fixed como warm-start). Gurobi parte
    # de esa solución factible del repair y la mejora CERRANDO puntos
    # innecesarios, algo que los polish con z fija no pueden hacer. Es el
    # paso más caro (time_limit=300s) pero único y final.
    if best_feasible is not None:
        # Guardamos la z de partida para contar cuántos puntos se cierran
        z_fixed_before = best_feasible.z.copy()
        if verbose:
            print(f"  [warm-start] Optimizando con cierre de puntos (time_limit=300s)...")
        warmstart = solve_fixed_locations(
            instance, best_feasible.z, valid_candidates,
            time_limit=300.0,
            allow_closures=True,
        )
        if warmstart is not None and warmstart.cost < best_ub:
            old_ub = best_ub
            best_ub = warmstart.cost
            best_feasible = warmstart
            gap = (best_ub - best_lb) / best_ub if best_ub > 0 else 0.0
            n_closed = int(np.sum(z_fixed_before) - np.sum(warmstart.z))
            if verbose:
                print(f"  [warm-start] UB: {old_ub:,.1f} → {best_ub:,.1f} "
                      f"(gap {gap:.2%}) — {n_closed} puntos cerrados")
        elif verbose:
            print(f"  [warm-start] Sin mejora")

    # Registrar estado final en historial
    ub_history.append(best_ub)
    lb_history.append(best_lb)
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
    import sys
    import os
    import re

    instance_path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/instancia_laguna_500m.json"

    radius_match = re.search(r'(\d+)m', instance_path)
    radius_tag = radius_match.group(1) + "m" if radius_match else "default"

    output_dir = f"output/lagrangiana_{radius_tag}"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 65)
    print("  Lagrangian Relaxation — HDM")
    print("=" * 65)

    inst = load_instance(instance_path)
    print(f"  Instance: {inst.study_case}")
    print(f"  Buildings: {inst.n_buildings}  Candidates: {inst.n_candidates}")
    print(f"  Waste types: {inst.n_waste_types}")
    print("-" * 65)

    start = time.time()
    result = solve_lagrangian(
        inst,
        max_iterations=4000,
        gap_tolerance=0.01,
        no_improve_limit=200,     # ← más paciencia antes de reducir θ
        verbose=True,
        print_every=50,
    )
    elapsed = time.time() - start

    print("-" * 65)
    print(f"  Completed in {result.n_iterations} iterations ({elapsed:.1f}s)")
    print(f"  Best LB:  {result.best_lb:,.1f}")
    print(f"  Best UB:  {result.best_ub:,.1f}")
    print(f"  Gap:      {result.gap:.2%}")
    print(f"  Points open: {int(np.sum(result.best_feasible.z))}")
    print(f"  Total bins:  {int(np.sum(result.best_feasible.x))}")
    print("=" * 65)

    # ── Guardar resultados para análisis posterior ─────────
    import pickle, json

    bf = result.best_feasible

    # 1. Pickle completo (preserva arrays numpy y todo el historial)
    pkl_path = f"{output_dir}/solucion_lagrangiana.pkl"
    payload = {
        "best_lb": result.best_lb,
        "best_ub": result.best_ub,
        "gap": result.gap,
        "lb_history": result.lb_history,
        "ub_history": result.ub_history,
        "best_feasible": {
            "z": bf.z,
            "x": bf.x,
            "y_assign": bf.y_assign,
            "w": bf.w,
            "cost": bf.cost,
        },
        "n_iterations": result.n_iterations,
        "runtime_seconds": elapsed,
    }
    with open(pkl_path, "wb") as f:
        pickle.dump(payload, f)

    # 2. Verificar que se guardó correctamente
    with open(pkl_path, "rb") as f:
        check = pickle.load(f)
    assert check["best_ub"] == result.best_ub, "ERROR: pickle corrupto"
    assert len(check["lb_history"]) == len(result.lb_history), "ERROR: historial perdido"

    # 3. Resumen legible en JSON (backup sin arrays)
    json_path = f"{output_dir}/solucion_lagrangiana_resumen.json"
    resumen = {
        "best_lb": result.best_lb,
        "best_ub": result.best_ub,
        "gap": result.gap,
        "n_iterations": result.n_iterations,
        "runtime_seconds": elapsed,
        "n_points_open": int(np.sum(bf.z)),
        "total_bins": int(np.sum(bf.x)),
        "bins_per_type": {
            k: int(np.sum(bf.x[:, k])) for k in range(inst.n_waste_types)
        },
        "open_points": [int(j) for j in np.where(bf.z)[0]],
    }
    with open(json_path, "w") as f:
        json.dump(resumen, f, indent=2)

    # 4. Confirmación
    print(f"  ✅ Pickle guardado y verificado: {pkl_path}")
    print(f"  ✅ Resumen JSON:                 {json_path}")
    print("=" * 65)