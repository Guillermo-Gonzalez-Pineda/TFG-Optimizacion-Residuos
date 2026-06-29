"""
P_LR_z subproblem: facility location (z) variables.
"""

from __future__ import annotations

import numpy as np
import gurobipy as gp

from instancia import Instance
from ..tipos import Multipliers


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
    model.ModelSense = gp.GRB.MINIMIZE
    model.Params.OutputFlag = 0  # Silenciar salida de Gurobi
    model.Params.Threads    = 6

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

    # NO se llama model.reset(): entre iteraciones solo cambian los
    # coeficientes del objetivo (las restricciones de cobertura son fijas),
    # así que conservar el estado permite a Gurobi warm-startear desde la
    # solución previa (paso 4b — aceleración).

    return z_sol, obj_plrz
