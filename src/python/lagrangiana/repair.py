"""
Feasibility repair and MIP polish for the Lagrangian relaxation.
"""

from __future__ import annotations

import warnings

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from instancia import Instance, compute_demand
from .tipos import FeasibleSolution


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
    #  Diagnóstico final + coste HONESTO
    # ────────────────────────────────────────────────────────────
    #  repair NO debe mentir: si la solución viola N_j (5) o deja
    #  edificios sin cubrir (2), su coste es +inf. Solo se reporta un
    #  coste finito sobre una solución factible. La factibilidad real
    #  la verifica is_feasible() de forma independiente; aquí solo
    #  evitamos devolver un número engañoso al orquestador.
    w = (x > 0).astype(bool)
    bin_cost_arr = np.array([instance.params.bin_cost[k] for k in instance.K])
    fixed_cost = float(np.sum(z_rep * C_j))
    variable_cost = float(np.sum(x * bin_cost_arr))

    nj_violations = sum(
        1 for j in range(n_candidates)
        if z_rep[j] and np.sum(x[j, :]) > N_j
    )
    # Edificios sin asignar pese a tener candidatos válidos abiertos
    uncovered = sum(
        1 for i in range(n_buildings) for k in range(n_waste_types)
        if y_assign[i, k] == -1 and valid_candidates[i][k]
    )

    if nj_violations > 0 or uncovered > 0:
        cost = float('inf')
        warnings.warn(
            f"repair: solución INFACTIBLE → coste=inf "
            f"({nj_violations} violaciones N_j, {uncovered} sin cubrir) "
            f"tras {repair_iter + 1} iteraciones"
        )
    else:
        cost = fixed_cost + variable_cost

    return FeasibleSolution(z=z_rep, x=x, y_assign=y_assign, w=w, cost=cost)


def solve_fixed_locations(
    instance: Instance,
    z_fixed: np.ndarray,
    valid_candidates: list[list[list[int]]],
    time_limit: float = 60.0,
    allow_closures: bool = False,
    solution_limit: int | None = None,
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
    solution_limit   : if set, Gurobi stops at the first N feasible
                       incumbents (SolutionLimit). With 1 → "first feasible"
                       mode, used by the feasibility safety net to obtain a
                       guaranteed-feasible UB fast (no optimisation).

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
    if solution_limit is not None:
        model.Params.SolutionLimit = solution_limit
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
