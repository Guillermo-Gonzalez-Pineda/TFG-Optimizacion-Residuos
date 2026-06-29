"""
Shared dataclasses for the Lagrangian relaxation module.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


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
    # Fields with defaults (must come last for backward compat with positional construction)
    wall_time:       float = 0.0    # total wall-clock time of solve_lagrangian
    time_to_best_lb: float = 0.0    # wall-clock instant when best_lb was last improved
    stop_reason:     str   = "unknown"  # "gap" | "no_improve" | "time_limit" | "max_iters"
