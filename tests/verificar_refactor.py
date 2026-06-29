"""
Verification script for the lagrangiana refactor.

Usage
-----
    # Save baseline from the ORIGINAL monolithic lagrangiana.py (run BEFORE refactor):
    python tests/verificar_refactor.py --save-baseline [instance_path] [max_iter]

    # Compare against baseline from the NEW package (run AFTER refactor):
    python tests/verificar_refactor.py --compare [instance_path] [max_iter]

    # Run both in one shot (use only AFTER refactor — baseline must exist):
    python tests/verificar_refactor.py --compare data/processed/instancia_laguna_1500m.json 2000

Defaults
--------
    instance_path : data/processed/instancia_laguna_1500m.json
    max_iter      : 2000

Compared fields
---------------
    best_lb, best_ub, gap, n_iterations  (bit-exact float equality)

Note: the multi-polish and warm-start steps depend on Gurobi's internal
state and time limits; for a deterministic check use max_iter=N with
gap_tolerance=0.0 so the run stops exactly at N iterations.
"""

from __future__ import annotations

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "python"))

from instancia import load_instance
from lagrangiana import solve_lagrangian

BASELINE_PATH = os.path.join(os.path.dirname(__file__), "verificar_refactor_baseline.json")

DEFAULT_INSTANCE = "data/processed/instancia_laguna_1500m.json"
DEFAULT_MAX_ITER = 6000


def run(instance_path: str, max_iter: int) -> dict:
    inst = load_instance(instance_path)
    print(f"  Instancia : {inst.study_case}  "
          f"({inst.n_buildings} edificios, {inst.n_candidates} candidatos)")
    print(f"  max_iter  : {max_iter}")
    result = solve_lagrangian(
        inst,
        max_iterations=max_iter,
        gap_tolerance=0.0,   # sin stop anticipado → determinista en nº iters
        no_improve_limit=200,
        verbose=False,
        print_every=9999,
    )
    return {
        "best_lb":     result.best_lb,
        "best_ub":     result.best_ub,
        "gap":         result.gap,
        "n_iterations": result.n_iterations,
    }


def save_baseline(instance_path: str, max_iter: int) -> None:
    print("=== GUARDANDO BASELINE ===")
    data = run(instance_path, max_iter)
    with open(BASELINE_PATH, "w") as f:
        json.dump({"instance_path": instance_path, "max_iter": max_iter, "result": data}, f, indent=2)
    print(f"  Guardado en {BASELINE_PATH}")
    print(f"  best_lb={data['best_lb']:,.1f}  best_ub={data['best_ub']:,.1f}  "
          f"gap={data['gap']:.4%}  n_iter={data['n_iterations']}")


def compare(instance_path: str, max_iter: int) -> None:
    print("=== COMPARANDO CON BASELINE ===")
    if not os.path.exists(BASELINE_PATH):
        print(f"  ERROR: no existe baseline en {BASELINE_PATH}")
        print("  Ejecuta primero --save-baseline con el código ORIGINAL.")
        sys.exit(1)

    with open(BASELINE_PATH) as f:
        saved = json.load(f)

    if saved["instance_path"] != instance_path or saved["max_iter"] != max_iter:
        print(f"  AVISO: parámetros distintos al baseline guardado "
              f"({saved['instance_path']}, iter={saved['max_iter']})")

    ref = saved["result"]
    current = run(instance_path, max_iter)

    fields = ["best_lb", "best_ub", "gap", "n_iterations"]
    all_ok = True
    print(f"\n  {'Campo':<15} {'Baseline':>20} {'Refactored':>20} {'OK?':>6}")
    print("  " + "-" * 65)
    for field in fields:
        ref_val = ref[field]
        cur_val = current[field]
        ok = ref_val == cur_val
        all_ok = all_ok and ok
        status = "OK" if ok else "FAIL"
        print(f"  {field:<15} {ref_val:>20} {cur_val:>20} {status:>6}")

    print()
    if all_ok:
        print("  RESULTADO: OK — resultados bit-a-bit idénticos.")
    else:
        print("  RESULTADO: FAIL — hay diferencias. Revisar el refactor.")
        sys.exit(1)


if __name__ == "__main__":
    args = sys.argv[1:]

    mode = "--compare"
    instance_path = DEFAULT_INSTANCE
    max_iter = DEFAULT_MAX_ITER

    if args and args[0] in ("--save-baseline", "--compare"):
        mode = args[0]
        args = args[1:]
    if args:
        instance_path = args[0]
        args = args[1:]
    if args:
        max_iter = int(args[0])

    if mode == "--save-baseline":
        save_baseline(instance_path, max_iter)
    else:
        compare(instance_path, max_iter)
