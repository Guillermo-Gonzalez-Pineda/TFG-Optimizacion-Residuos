"""
Exact MIP formulation of the HDM using Gurobi.
Solves the complete model with all constraints intact.
"""

from __future__ import annotations

import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB

from instancia import Instance, compute_demand
from lagrangiana import precompute_valid_candidates


def solve_exact_hdm(
    instance: Instance,
    valid_candidates: list[list[list[int]]],
    time_limit: float = 3600.0,
    verbose: bool = True,
) -> dict:

    n_i = instance.n_buildings
    n_j = instance.n_candidates
    n_k = instance.n_waste_types
    params = instance.params

    model = gp.Model("HDM_exact")
    if not verbose:
        model.Params.OutputFlag = 0
    model.Params.TimeLimit = time_limit
    model.Params.LogFile = "output/gurobi_run.log"


    # ── Variables ──────────────────────────────────────────
    z = model.addVars(n_j, vtype=GRB.BINARY, name="z")
    x = model.addVars(n_j, n_k, vtype=GRB.INTEGER, lb=0, name="x")
    w = model.addVars(n_j, n_k, vtype=GRB.BINARY, name="w")

    # y solo para triples válidos
    y = {}
    for i in range(n_i):
        for k in range(n_k):
            for j in valid_candidates[i][k]:
                y[i, j, k] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}_{k}")

    # ── Objetivo (1) ──────────────────────────────────────
    model.setObjective(
        gp.quicksum(instance.J[j].opening_cost * z[j] for j in range(n_j))
        + gp.quicksum(params.bin_cost[k] * x[j, k]
                       for j in range(n_j) for k in range(n_k)),
        GRB.MINIMIZE
    )

    # ── (2) Asignación completa ───────────────────────────
    for i in range(n_i):
        for k in range(n_k):
            model.addConstr(
                gp.quicksum(y[i, j, k] for j in valid_candidates[i][k]) == 1
            )

    # ── (3) Orgánica y resto juntos ───────────────────────
    for i in range(n_i):
        for j in valid_candidates[i][0]:  # r_0 == r_1, mismos candidatos
            if (i, j, 1) in y:
                model.addConstr(y[i, j, 0] == y[i, j, 1])

    # ── (4) Capacidad ─────────────────────────────────────
    for j in range(n_j):
        for k in range(n_k):
            assigned = [i for i in range(n_i) if (i, j, k) in y]
            if assigned:
                model.addConstr(
                    gp.quicksum(
                        compute_demand(instance.I[i].h_i, params, k) * y[i, j, k]
                        for i in assigned
                    ) <= params.bin_capacity[k] * x[j, k]
                )

    # ── (5) Límite físico ─────────────────────────────────
    for j in range(n_j):
        model.addConstr(
            gp.quicksum(x[j, k] for k in range(n_k)) <= params.max_bins * z[j]
        )

    # ── (8) Relación x-w ─────────────────────────────────
    for j in range(n_j):
        for k in range(n_k):
            model.addConstr(x[j, k] <= params.max_bins * w[j, k])

    # ── (9) Nearest-allocation ────────────────────────────
    for i in range(n_i):
        for k in range(n_k):
            cands = valid_candidates[i][k]
            for p in range(1, len(cands)):
                j = cands[p]                    # candidato más lejano
                for q in range(p):
                    j_closer = cands[q]         # candidato más cercano
                    model.addConstr(y[i, j, k] + w[j_closer, k] <= 1)

    # ── Resolver ──────────────────────────────────────────
    model.optimize()

    # ── Extraer solución ──────────────────────────────────
    # ── Extraer solución ──────────────────────────────────
    if model.Status == GRB.OPTIMAL or model.SolCount > 0:
        solution = {
            "z": {j: int(z[j].X > 0.5) for j in range(n_j)},
            "x": {(j, k): int(round(x[j, k].X)) for j in range(n_j) for k in range(n_k)},
            "w": {(j, k): int(w[j, k].X > 0.5) for j in range(n_j) for k in range(n_k)},
            "y_assign": {},
            "cost": model.ObjVal,
            "gap_gurobi": model.MIPGap,
            "runtime": model.Runtime,
            "status": model.Status,
        }
        # Extraer asignaciones en formato y_assign[i,k] = j
        for i in range(n_i):
            for k in range(n_k):
                for j in valid_candidates[i][k]:
                    if y[i, j, k].X > 0.5:
                        solution["y_assign"][(i, k)] = j
                        break
        return solution
    else:
        print(f"Gurobi status: {model.Status} — no solution found")
        return {"status": model.Status, "cost": None}
    

if __name__ == "__main__":
    from instancia import load_instance
    import sys
    import os
    import re

    instance_path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/instancia_laguna_500m.json"

    radius_match = re.search(r'(\d+)m', instance_path)
    radius_tag = radius_match.group(1) + "m" if radius_match else "default"

    output_dir = f"output/exacto_{radius_tag}"
    os.makedirs(output_dir, exist_ok=True)
    log_path = f"{output_dir}/gurobi_run.log"

    # Redirigir stdout y stderr al fichero y pantalla simultáneamente
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()

    log_file = open(log_path, "w")
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)

    print("=" * 65)
    print("  Exact HDM — Gurobi MIP")
    print("=" * 65)

    inst = load_instance(instance_path)
    vc = precompute_valid_candidates(inst)

    print(f"  Instance: {inst.study_case}")
    print(f"  Buildings: {inst.n_buildings}  Candidates: {inst.n_candidates}")
    print(f"  Valid y-variables: {sum(len(vc[i][k]) for i in range(inst.n_buildings) for k in range(inst.n_waste_types))}")
    print("-" * 65)

    result = solve_exact_hdm(inst, vc, time_limit=10800, verbose=True)

    print("-" * 65)
    if result["cost"] is not None:
        n_open = sum(result["z"].values())
        total_bins = sum(result["x"].values())
        print(f"  Optimal cost:  {result['cost']:,.1f}")
        print(f"  Gurobi gap:    {result['gap_gurobi']:.4%}")
        print(f"  Runtime:       {result['runtime']:.1f}s")
        print(f"  Points open:   {n_open}")
        print(f"  Total bins:    {total_bins}")
    else:
        print(f"  No solution found. Status: {result['status']}")

    # ── Guardar solución para análisis posterior ──────────
    if result["cost"] is not None:
        import pickle, json

        os.makedirs(output_dir, exist_ok=True)

        # 1. Pickle completo (preserva claves tupla)
        pkl_path = f"{output_dir}/solucion_exacta.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(result, f)

        # 2. Verificar que se guardó correctamente
        with open(pkl_path, "rb") as f:
            check = pickle.load(f)
        assert check["cost"] == result["cost"], "ERROR: pickle corrupto"
        assert len(check["y_assign"]) == len(result["y_assign"]), "ERROR: asignaciones perdidas"

        # 3. Resumen legible en JSON (backup sin claves tupla)
        json_path = f"{output_dir}/solucion_exacta_resumen.json"
        resumen = {
            "cost": result["cost"],
            "gap_gurobi": result["gap_gurobi"],
            "runtime": result["runtime"],
            "status": result["status"],
            "n_points_open": sum(result["z"].values()),
            "total_bins": sum(result["x"].values()),
            "bins_per_type": {
                k: sum(v for (j, kk), v in result["x"].items() if kk == k)
                for k in range(inst.n_waste_types)
            },
            "open_points": [j for j, v in result["z"].items() if v == 1],
        }
        with open(json_path, "w") as f:
            json.dump(resumen, f, indent=2)

        print(f"  ✅ Pickle guardado y verificado: {pkl_path}")
        print(f"  ✅ Resumen JSON:                 {json_path}")
        print(f"     Contenido del pickle:")
        print(f"       z:        {len(result['z'])} entries")
        print(f"       x:        {len(result['x'])} entries")
        print(f"       w:        {len(result['w'])} entries")
        print(f"       y_assign: {len(result['y_assign'])} entries")
    else:
        print("  ⚠️  No solution to save.")

    print("=" * 65)

