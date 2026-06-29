"""
Independent feasibility checker for HDM solutions.

is_feasible() verifies a FeasibleSolution against the HDM constraints
WITHOUT trusting how it was built (it does not look at the cost field,
nor assume any invariant from repair/solve_fixed_locations). It is the
single source of truth before accepting any upper bound.
"""

from __future__ import annotations

import numpy as np

from instancia import Instance, compute_demand
from .tipos import FeasibleSolution


def is_feasible(
    instance: Instance,
    solution: FeasibleSolution,
    valid_candidates: list[list[list[int]]],
    max_report: int = 10,
) -> tuple[bool, list[str]]:
    """
    Check whether `solution` is a feasible HDM solution.

    Verifies, independently of construction:
      (5) Σ_k x[j,k] ≤ N_j           for every open point j
      (2) every (i,k) with valid candidates is assigned to an open,
          valid candidate (y_assign != -1, in range, z[j]=1)
      (3) organic (k=0) and resto (k=1) share the same point
      (4) Σ_i q_ik·[y=j] ≤ Q_k·x[j,k] for every (j,k)
      z–x consistency: x[j,k]>0 ⟹ z[j]=1
      w coherence:     w[j,k] == (x[j,k]>0)
      (9) nearest-allocation: each building uses the closest open point
          that actually holds a bin of that type

    Returns (feasible, violations). `violations` lists human-readable
    reasons (truncated to `max_report` per category) for logging.
    """

    n_i = len(instance.I)
    n_j = len(instance.J)
    n_k = len(instance.K)
    N_j = instance.params.max_bins
    Q = np.array([instance.params.bin_capacity[k] for k in range(n_k)])

    z = solution.z
    x = solution.x
    y = solution.y_assign
    w = solution.w

    violations: list[str] = []

    def add(cat: str, items: list[str]) -> None:
        if items:
            extra = f" (+{len(items) - max_report} más)" if len(items) > max_report else ""
            violations.append(f"[{cat}] {len(items)}: " + "; ".join(items[:max_report]) + extra)

    # ── (5) límite de bins por punto ──────────────────────────────
    add("(5) N_j", [
        f"j={j} Σx={int(np.sum(x[j, :]))}>{N_j}"
        for j in range(n_j) if z[j] and np.sum(x[j, :]) > N_j
    ])

    # ── z–x consistency: bins en punto cerrado ────────────────────
    add("z-x", [
        f"j={j},k={k} x={int(x[j, k])} pero z={int(z[j])}"
        for j in range(n_j) for k in range(n_k)
        if x[j, k] > 0 and not z[j]
    ])

    # ── (8) x>0 ⟹ w=1 ─────────────────────────────────────────────
    #  Solo se exige la implicación del modelo (8): si hay bins, hay
    #  presencia. El recíproco NO es restricción del HDM: w=1 con x=0
    #  (presencia sin bins) es admisible — w no tiene coste y la
    #  asignación ya respeta (9) con esa w, así que no falsea nada.
    add("(8) x>0⟹w", [
        f"j={j},k={k} x={int(x[j, k])} pero w=0"
        for j in range(n_j) for k in range(n_k)
        if x[j, k] > 0 and not w[j, k]
    ])

    # ── (2) asignación a candidato válido y abierto ───────────────
    cov = []
    for i in range(n_i):
        for k in range(n_k):
            if not valid_candidates[i][k]:
                continue  # (i,k) sin cobertura posible: no genera restricción (2)
            j = y[i, k]
            if j == -1:
                cov.append(f"i={i},k={k} sin asignar")
            elif j not in valid_candidates[i][k]:
                cov.append(f"i={i},k={k} j={j} fuera de radio")
            elif not z[j]:
                cov.append(f"i={i},k={k} j={j} cerrado")
    add("(2) cobertura", cov)

    # ── (3) orgánica (0) y resto (1) al mismo punto ───────────────
    add("(3) org=resto", [
        f"i={i} y0={y[i, 0]} y1={y[i, 1]}"
        for i in range(n_i)
        if valid_candidates[i][0] and valid_candidates[i][1]
        and y[i, 0] != y[i, 1]
    ])

    # ── (4) capacidad: demanda asignada ≤ Q_k·x[j,k] ──────────────
    assigned = np.zeros((n_j, n_k))
    for i in range(n_i):
        for k in range(n_k):
            j = y[i, k]
            if j != -1:
                assigned[j, k] += compute_demand(instance.I[i].h_i, instance.params, k)
    cap = []
    for j in range(n_j):
        for k in range(n_k):
            if assigned[j, k] > Q[k] * x[j, k] + 1e-6:
                cap.append(f"j={j},k={k} dem={assigned[j, k]:.0f}>{Q[k] * x[j, k]:.0f}")
    add("(4) capacidad", cap)

    # ── (9) nearest-allocation ────────────────────────────────────
    near = []
    for i in range(n_i):
        for k in range(n_k):
            j = y[i, k]
            if j == -1:
                continue
            for jc in valid_candidates[i][k]:
                if jc == j:
                    break  # llegamos al asignado sin encontrar uno más cercano con bin
                if w[jc, k]:
                    near.append(f"i={i},k={k} usa j={j} habiendo j'={jc} más cerca con bin")
                    break
    add("(9) nearest", near)

    return (len(violations) == 0, violations)
