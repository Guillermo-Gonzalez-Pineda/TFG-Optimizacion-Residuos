"""
P_LR_yw subproblem: assignment (y) and presence (w) variables.
"""

from __future__ import annotations

import numpy as np

from instancia import Instance
from ..tipos import Multipliers


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
