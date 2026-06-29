"""
Persistencia de resultados de la relajación lagrangiana.

Replica la ESTRUCTURA del modelo exacto (modelo_exacto.py) para permitir
comparación fichero-a-fichero:

    output/lagrangiana_{R}m/
        solucion_lagrangiana.pkl          (dicts z/x/w/y_assign + metadatos)
        solucion_lagrangiana_resumen.json (mismas claves que el exacto + extras)

Las estructuras de solución (z, x, w, y_assign) se guardan en el MISMO
formato de dict con claves-tupla que el exacto. Se añaden los campos
propios de una relajación (LB, gap dual, iteraciones, motivo de parada,
desglose de coste fijo/variable) que el exacto no tiene.
"""

from __future__ import annotations

import os
import json
import pickle

import numpy as np

from instancia import Instance
from .tipos import LagrangianResult


def _solucion_a_dicts(fs, n_i: int, n_j: int, n_k: int) -> dict:
    """Convierte una FeasibleSolution (arrays numpy) al formato dict del
    exacto: z {j:0/1}, x {(j,k):int}, w {(j,k):0/1}, y_assign {(i,k):j}."""
    z = {j: int(fs.z[j]) for j in range(n_j)}
    x = {(j, k): int(fs.x[j, k]) for j in range(n_j) for k in range(n_k)}
    w = {(j, k): int(bool(fs.w[j, k])) for j in range(n_j) for k in range(n_k)}
    y_assign = {
        (i, k): int(fs.y_assign[i, k])
        for i in range(n_i) for k in range(n_k)
        if fs.y_assign[i, k] != -1
    }
    return {"z": z, "x": x, "w": w, "y_assign": y_assign}


def _desglose_coste(fs, instance: Instance) -> tuple[float, float]:
    """Coste fijo (apertura) y variable (bins) de una FeasibleSolution."""
    n_j = len(instance.J)
    n_k = len(instance.K)
    C_j = np.array([instance.J[j].opening_cost for j in range(n_j)])
    bin_cost = np.array([instance.params.bin_cost[k] for k in range(n_k)])
    fixed = float(np.sum(fs.z * C_j))
    variable = float(np.sum(fs.x * bin_cost))
    return fixed, variable


def guardar_solucion(
    result: LagrangianResult,
    instance: Instance,
    output_dir: str,
    runtime_seconds: float,
    verbose: bool = True,
) -> tuple[str, str]:
    """
    Guarda la solución de la relajación en `output_dir`, en formato espejo
    del exacto. Devuelve (ruta_pkl, ruta_json).

    Si no hay solución factible (best_feasible None), guarda igualmente el
    resumen con cost=null y feasible=False, para no perder LB/diagnóstico.
    """
    os.makedirs(output_dir, exist_ok=True)
    n_i, n_j, n_k = instance.n_buildings, instance.n_candidates, instance.n_waste_types
    bf = result.best_feasible

    # ── Estructuras de solución (formato exacto) ──────────────────
    if bf is not None:
        sol_dicts = _solucion_a_dicts(bf, n_i, n_j, n_k)
        fixed, variable = _desglose_coste(bf, instance)
        n_open = int(np.sum(bf.z))
        total_bins = int(np.sum(bf.x))
        bins_per_type = {k: int(np.sum(bf.x[:, k])) for k in range(n_k)}
        open_points = [int(j) for j in np.where(bf.z)[0]]
        cost = float(bf.cost)
        feasible = bool(np.isfinite(cost))
    else:
        sol_dicts = {"z": {}, "x": {}, "w": {}, "y_assign": {}}
        fixed = variable = None
        n_open = total_bins = 0
        bins_per_type = {k: 0 for k in range(n_k)}
        open_points = []
        cost = None
        feasible = False

    # ── 1. Pickle completo ────────────────────────────────────────
    pkl_path = os.path.join(output_dir, "solucion_lagrangiana.pkl")
    payload = {
        **sol_dicts,                       # z, x, w, y_assign (formato exacto)
        "cost": cost,                      # = best_ub (UB factible entregado)
        "cost_fixed": fixed,
        "cost_variable": variable,
        "best_lb": result.best_lb,
        "best_ub": result.best_ub,
        "gap": result.gap,                 # gap dual (basado en LB)
        "n_iterations": result.n_iterations,
        "stop_reason": result.stop_reason,
        "runtime": runtime_seconds,
        "wall_time": result.wall_time,
        "time_to_best_lb": result.time_to_best_lb,
        "feasible": feasible,
        "lb_history": result.lb_history,
        "ub_history": result.ub_history,
    }
    with open(pkl_path, "wb") as f:
        pickle.dump(payload, f)

    # Verificación de integridad (como el exacto)
    with open(pkl_path, "rb") as f:
        check = pickle.load(f)
    assert check["best_ub"] == result.best_ub, "ERROR: pickle corrupto"
    assert len(check["lb_history"]) == len(result.lb_history), "ERROR: historial perdido"

    # ── 2. Resumen JSON (claves del exacto + extras) ──────────────
    json_path = os.path.join(output_dir, "solucion_lagrangiana_resumen.json")
    resumen = {
        # --- claves compartidas con el exacto ---
        "cost": cost,
        "gap": result.gap,
        "runtime": runtime_seconds,
        "stop_reason": result.stop_reason,
        "n_points_open": n_open,
        "total_bins": total_bins,
        "bins_per_type": bins_per_type,
        "open_points": open_points,
        # --- extras propios de la relajación ---
        "best_lb": result.best_lb,
        "best_ub": result.best_ub,
        "cost_fixed": fixed,
        "cost_variable": variable,
        "n_iterations": result.n_iterations,
        "time_to_best_lb": result.time_to_best_lb,
        "feasible": feasible,
    }
    with open(json_path, "w") as f:
        json.dump(resumen, f, indent=2)

    if verbose:
        print(f"  Pickle guardado y verificado: {pkl_path}")
        print(f"  Resumen JSON:                 {json_path}")

    return pkl_path, json_path
