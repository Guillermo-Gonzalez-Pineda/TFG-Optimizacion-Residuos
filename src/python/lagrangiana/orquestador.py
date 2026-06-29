"""
Main orchestrator: precompute_valid_candidates and solve_lagrangian.
"""

from __future__ import annotations

import time

import numpy as np

from instancia import Instance, compute_demand
from .tipos import Multipliers, FeasibleSolution, LagrangianResult
from .subproblemas import (
    build_plr_z_model, solve_plr_z,
    build_plr_x_model, solve_plr_x,
    solve_plr_yw, solve_plr_x_greedy,
)
from .subgradiente import compute_subgradient, compute_step_length, update_multipliers
from .repair import repair_solution, solve_fixed_locations
from .factibilidad import is_feasible


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


def _safety_net_feasible(
    instance: Instance,
    z_warm: np.ndarray,
    valid_candidates: list[list[list[int]]],
    time_limit: float,
    solution_limit: int | None = None,
) -> FeasibleSolution | None:
    """
    Feasibility safety net (paso 2c).

    Guarantees a feasible UB via the FULL HDM (allow_closures=True),
    warm-started with z_warm: z becomes a variable so Gurobi can OPEN the
    extra points that dense zones need — the freedom repair and the
    z-fixed MIP (mode False) lack. With solution_limit=1 it stops at the
    first feasible incumbent ("primer factible"), used to seed a UB fast.

    Returns a verified-feasible solution, or None only if the instance
    admits no feasible solution within the time budget.
    """
    sol = solve_fixed_locations(
        instance, z_warm, valid_candidates,
        time_limit=time_limit, allow_closures=True, solution_limit=solution_limit,
    )
    if sol is not None and is_feasible(instance, sol, valid_candidates)[0]:
        return sol
    return None


def solve_lagrangian(
    instance: Instance,
    max_iterations: int = 60000,
    gap_tolerance: float = 0.01,
    no_improve_limit: int = 250,
    no_improve_stop: int = 10000,
    time_limit: float = 3600*2.0,
    verbose: bool = True,
    print_every: int = 250,
    seed_time_limit: float = 45.0,
    net_time_limit: float = 180.0,
    x_subproblem: str = "greedy",
) -> LagrangianResult:
    """
    Solve the HDM using Lagrangian relaxation with subgradient
    optimization, as described in Li et al. (2026), Section 4.

    Implements the 5-step algorithm:
      Step 1: Initialize multipliers, theta, bounds
      Step 2: Solve P_LR (three subproblems) → lower bound
      Step 3: Compute subgradient, repair → upper bound
      Step 4: Update multipliers, reduce theta if stalled
      Step 5: Check stopping criterion

    Stopping hierarchy (first triggered wins):
      1. gap <= gap_tolerance          — primal-dual convergence
      2. iters_no_lb_improve >= no_improve_stop  — dual stagnation
      3. wall_time >= time_limit       — safety time budget
      4. iteration >= max_iterations   — safety iteration budget

    Returns a LagrangianResult with the best feasible solution,
    bounds, gap, iteration count, convergence history, timing, and
    the stop reason.
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
    best_ub = float(1e9)
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
    # El subproblema x admite dos solvers: "greedy" (heurístico, ~3x más rápido,
    # ignora el acoplamiento Σ_k x[j,k] ≤ N_j) o "exact" (Gurobi, modelo reusable).
    use_greedy_x = (x_subproblem == "greedy")
    model_x, x_vars = (None, None) if use_greedy_x else build_plr_x_model(instance, demand_matrix)

    # ── Dos contadores independientes ───────────────────────────
    # iters_without_improvement: para el halving de theta (se resetea con
    #   el halving Y cuando best_lb mejora).
    # iters_no_lb_improve: para el criterio de parada no_improve_stop.
    #   Se resetea SOLO cuando best_lb mejora de verdad; el halving de
    #   theta NO lo toca.
    iters_without_improvement = 0
    iters_no_lb_improve = 0

    # ── Tiempos ─────────────────────────────────────────────────
    t_start = time.perf_counter()
    time_to_best_lb = 0.0   # instante en que se alcanzó el best_lb final

    # ── Pool de soluciones para multi-polish ────────────────────
    top_k_solutions = []  # lista de (cost, z_array)
    MAX_TOP_K = 5

    # ── Control del polish intermedio (mid-polish) ──────────────
    ub_stagnation_counter = 0
    ub_last_improved = best_ub
    polish_exhausted = False

    # ── Red de seguridad de factibilidad (paso 2c) ──────────────
    # Se siembra UNA vez, la primera iteración en que repair no logra una
    # solución factible (zona densa). Garantiza un best_ub factible para
    # que el step-length de Polyak no opere con 1e9.
    seeded = False

    # ── Estado de parada ────────────────────────────────────────
    stop_reason = "max_iters"
    gap = 0.0


    # ── Step 2-5: Iterar hasta convergencia ─────────────────────
    for iteration in range(max_iterations):

        # ── Step 2: Solve P_LR(γᵗ) → lower bound ──────────────
        z, obj_plrz = solve_plr_z(model_z, z_vars, instance, multipliers)
        if use_greedy_x:
            x, obj_plrx = solve_plr_x_greedy(instance, multipliers)
        else:
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
            # Ambos contadores se resetean cuando best_lb mejora
            iters_without_improvement = 0
            iters_no_lb_improve = 0
            time_to_best_lb = time.perf_counter() - t_start
        else:
            iters_without_improvement += 1
            iters_no_lb_improve += 1

        lb_history.append(best_lb)


        # ── Step 3: Subgradient + feasibility → upper bound ────
        phi_mu, phi_lbd, phi_nu = compute_subgradient(
            instance, z, x, y_assign, w, demand_matrix
        )

        feasible = repair_solution(instance, z, valid_candidates, demand_matrix)

        # Solo se acepta como UB si MEJORA y es FACTIBLE de verdad
        # (is_feasible, verificación independiente). repair ya devuelve
        # inf cuando es infactible, así que is_feasible solo corre sobre
        # candidatos que mejoran → coste despreciable.
        if feasible.cost < best_ub:
            ok, viols = is_feasible(instance, feasible, valid_candidates)
            if ok:
                best_ub = feasible.cost
                best_feasible = feasible
            elif verbose:
                print(f"  [factib] iter {iteration}: repair coste finito pero "
                      f"INFACTIBLE, descartado → {viols[:2]}")

        # ── Semilla de factibilidad (paso 2c): primera vez que NO hay UB ──
        # factible. Garantiza un best_ub finito y factible desde el principio.
        if best_feasible is None and not seeded:
            seeded = True
            seed = _safety_net_feasible(
                instance, z, valid_candidates,
                time_limit=seed_time_limit, solution_limit=1,
            )
            if seed is not None:
                best_ub = seed.cost
                best_feasible = seed
                if verbose:
                    print(f"  [seed 2c] iter {iteration}: UB factible sembrado "
                          f"(HDM primer-factible) = {best_ub:,.1f}")
            elif verbose:
                print(f"  [seed 2c] iter {iteration}: la red de seguridad no "
                      f"halló factible en {seed_time_limit:.0f}s")

        # ── Guardar z para multi-polish si es DIVERSA y FACTIBLE ───
        # No metemos en el pool z's contaminados (coste inf): pulir un z
        # infactible con modo False solo devuelve None (paso 2d).
        if feasible.cost < float('inf'):
            z_copy = feasible.z.copy()
            is_diverse = all(
                np.sum(z_copy ^ existing_z) >= 2
                for _, existing_z in top_k_solutions
            )
            if is_diverse:
                top_k_solutions.append((feasible.cost, z_copy))
                top_k_solutions.sort(key=lambda x: x[0])
                top_k_solutions = top_k_solutions[:MAX_TOP_K]

        ub_history.append(best_ub)

        # ── Polish intermedio (mid-polish) ─────────────────────
        if best_ub < ub_last_improved:
            ub_stagnation_counter = 0
            ub_last_improved = best_ub
            polish_exhausted = False
        else:
            ub_stagnation_counter += 1

        if best_feasible is not None and ub_stagnation_counter >= 500 and not polish_exhausted:
            polished = solve_fixed_locations(
                instance, best_feasible.z, valid_candidates, time_limit=120.0
            )
            if (polished is not None and polished.cost < best_ub
                    and is_feasible(instance, polished, valid_candidates)[0]):
                best_ub = polished.cost
                best_feasible = polished
                gap = (best_ub - best_lb) / best_ub if best_ub > 0 else 0.0
                ub_stagnation_counter = 0
                ub_last_improved = best_ub
                ub_history[-1] = best_ub
                if verbose:
                    print(f"  [mid-polish] UB mejorado: {best_ub:,.1f} (gap {gap:.2%})")
            else:
                polish_exhausted = True
                if verbose:
                    print(f"  [mid-polish] Sin mejora, desactivado hasta que UB cambie")


        # ── Step 5: Stopping criterion ──────────────────────────
        # Jerarquía: gap → no_improve → time_limit → max_iters
        gap = (best_ub - best_lb) / best_ub if best_ub > 0 else 0.0

        if gap <= gap_tolerance:
            stop_reason = "gap"
            break

        if iters_no_lb_improve >= no_improve_stop:
            stop_reason = "no_improve"
            break

        if time.perf_counter() - t_start >= time_limit:
            stop_reason = "time_limit"
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

        # Theta halving: usa solo iters_without_improvement (se resetea aquí)
        # iters_no_lb_improve NO se resetea con el halving
        if iters_without_improvement >= no_improve_limit:
            theta /= 2
            iters_without_improvement = 0

    # ── Línea resumen de parada ─────────────────────────────────
    t_loop = time.perf_counter() - t_start
    print(
        f"[STOP] reason={stop_reason}  iters={iteration + 1}  "
        f"best_lb={best_lb:,.1f}  best_ub={best_ub:,.1f}  gap={gap:.2%}\n"
        f"       wall={t_loop:.1f}s  t_best_lb={time_to_best_lb:.1f}s"
    )

    # ══════════════════════════════════════════════════════════
    # MULTI-POLISH — después del bucle, antes del return
    # ══════════════════════════════════════════════════════════
    if top_k_solutions and valid_candidates is not None:
        if verbose:
            print(f"  [multi-polish] Puliendo {len(top_k_solutions)} soluciones...")
        for idx, (cost, z_candidate) in enumerate(top_k_solutions):
            polished = solve_fixed_locations(
                instance, z_candidate, valid_candidates, time_limit=120.0
            )
            if (polished is not None and polished.cost < best_ub
                    and is_feasible(instance, polished, valid_candidates)[0]):
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
    if best_feasible is not None:
        z_fixed_before = best_feasible.z.copy()
        if verbose:
            print(f"  [warm-start] Optimizando con cierre de puntos (time_limit=300s)...")
        warmstart = solve_fixed_locations(
            instance, best_feasible.z, valid_candidates,
            time_limit=300.0,
            allow_closures=True,
        )
        if (warmstart is not None and warmstart.cost < best_ub
                and is_feasible(instance, warmstart, valid_candidates)[0]):
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

    # ══════════════════════════════════════════════════════════
    # VERIFICACIÓN FINAL (paso 1d + red 2c): nunca un UB infactible
    # ══════════════════════════════════════════════════════════
    # Si pese a todo no hay UB factible (la semilla 2c falló o nunca se
    # disparó), último intento con la red de seguridad y más tiempo.
    if best_feasible is None:
        if verbose:
            print(f"  [red 2c-final] Sin UB factible; último intento "
                  f"(HDM, time_limit={net_time_limit:.0f}s)...")
        # Warm-start con el último z relajado disponible (cobertura garantizada)
        net = _safety_net_feasible(instance, z, valid_candidates, time_limit=net_time_limit)
        if net is not None:
            best_ub = net.cost
            best_feasible = net
            gap = (best_ub - best_lb) / best_ub if best_ub > 0 else 0.0

    # Verificación: nunca entregar una solución infactible como UB
    if best_feasible is None:
        best_ub = float('inf')
        gap = float('inf')
        print("  [FACTIBILIDAD] ⚠️  Sin UB factible: la instancia no admitió "
              "solución factible en el tiempo dado.")
    else:
        ok, viols = is_feasible(instance, best_feasible, valid_candidates)
        assert ok, f"BUG: best_feasible entregado es INFACTIBLE → {viols}"
        if verbose:
            print(f"  [FACTIBILIDAD] ✅ UB entregado verificado factible: {best_ub:,.1f}")

    # Registrar estado final en historial
    ub_history.append(best_ub)
    lb_history.append(best_lb)

    wall_time = time.perf_counter() - t_start

    return LagrangianResult(
        best_feasible=best_feasible,
        best_lb=best_lb,
        best_ub=best_ub,
        gap=gap,
        n_iterations=iteration + 1,
        lb_history=lb_history,
        ub_history=ub_history,
        wall_time=wall_time,
        time_to_best_lb=time_to_best_lb,
        stop_reason=stop_reason,
    )
