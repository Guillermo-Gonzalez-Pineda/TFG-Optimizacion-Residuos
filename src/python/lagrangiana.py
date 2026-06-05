"""
Lagrangian relaxation heuristic for the HDM/HVM waste collection
location problem.
"""

from __future__ import annotations

import numpy as np
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

            

    
