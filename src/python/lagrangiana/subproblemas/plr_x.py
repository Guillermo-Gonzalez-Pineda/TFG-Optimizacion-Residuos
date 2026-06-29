"""
P_LR_x subproblem: bin placement (x) variables.
"""

from __future__ import annotations

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from instancia import Instance, compute_demand
from ..tipos import Multipliers


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

    # NO se llama model.reset(): solo cambian los coeficientes del objetivo
    # entre iteraciones; conservar el estado permite warm-start de Gurobi
    # (paso 4b — aceleración).

    return x_sol, obj_plrx
