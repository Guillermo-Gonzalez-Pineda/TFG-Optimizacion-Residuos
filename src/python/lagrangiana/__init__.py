"""
Lagrangian relaxation heuristic for the HDM/HVM waste collection
location problem.

Public API — identical to the original monolithic lagrangiana.py so
existing imports continue to work without any changes.
"""

from .tipos import Multipliers, LRSolution, FeasibleSolution, LagrangianResult
from .orquestador import solve_lagrangian, precompute_valid_candidates
from .repair import repair_solution, solve_fixed_locations
from .factibilidad import is_feasible
from .persistencia import guardar_solucion
from .subproblemas import (
    solve_plr_yw,
    build_plr_x_model,
    solve_plr_x,
    solve_plr_x_greedy,
    build_plr_z_model,
    solve_plr_z,
)
from .subgradiente import compute_subgradient, compute_step_length, update_multipliers

__all__ = [
    "Multipliers",
    "LRSolution",
    "FeasibleSolution",
    "LagrangianResult",
    "precompute_valid_candidates",
    "solve_lagrangian",
    "repair_solution",
    "solve_fixed_locations",
    "is_feasible",
    "guardar_solucion",
    "solve_plr_yw",
    "build_plr_x_model",
    "solve_plr_x",
    "solve_plr_x_greedy",
    "build_plr_z_model",
    "solve_plr_z",
    "compute_subgradient",
    "compute_step_length",
    "update_multipliers",
]
