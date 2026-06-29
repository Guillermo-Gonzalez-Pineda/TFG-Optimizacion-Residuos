"""
Subgradient computation and multiplier update for Lagrangian relaxation.
"""

from __future__ import annotations

import numpy as np

from instancia import Instance
from .tipos import Multipliers


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
