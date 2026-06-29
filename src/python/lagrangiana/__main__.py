"""
Entry point: python -m lagrangiana <tamaño|ruta>

Parametrizado por TAMAÑO de instancia (espejo de modelo_exacto.py):

    python -m lagrangiana 1500          # → data/processed/instancia_laguna_1500m.json
    python -m lagrangiana 500
    python -m lagrangiana data/processed/instancia_laguna_700m.json   # ruta directa

Lee la instancia de su directorio y guarda los resultados en
output/lagrangiana_{R}m/, en formato espejo del exacto (output/exacto_{R}m/)
para comparación fichero-a-fichero.
"""

from __future__ import annotations

import os
import re
import sys
import time

import numpy as np

from instancia import load_instance
from lagrangiana import solve_lagrangian
from lagrangiana.persistencia import guardar_solucion


def _resolver_instancia(arg: str) -> tuple[str, str]:
    """Devuelve (ruta_instancia, radius_tag) a partir de un tamaño (1500) o
    una ruta. Acepta '1500', '1500m' o una ruta .json."""
    if arg.endswith(".json") or os.sep in arg or "/" in arg:
        path = arg
    else:
        size = re.sub(r"[^0-9]", "", arg)   # '1500m' → '1500'
        path = f"../../data/processed/instancia_laguna_{size}m.json"
    m = re.search(r"(\d+)m", path)
    radius_tag = (m.group(1) + "m") if m else "default"
    return path, radius_tag


class _Tee:
    """Duplica stdout/stderr a fichero y pantalla (como el exacto)."""
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj); f.flush()
    def flush(self):
        for f in self.files:
            f.flush()


def main() -> None:
    arg = sys.argv[1] if len(sys.argv) > 1 else "500"
    instance_path, radius_tag = _resolver_instancia(arg)

    output_dir = f"output/lagrangiana_{radius_tag}"
    os.makedirs(output_dir, exist_ok=True)

    log_file = open(os.path.join(output_dir, "lagrangiana_run.log"), "w")
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)

    print("=" * 65)
    print("  Lagrangian Relaxation — HDM")
    print("=" * 65)

    if not os.path.exists(instance_path):
        print(f"  ERROR: instancia no encontrada: {instance_path}")
        sys.exit(1)

    inst = load_instance(instance_path)
    print(f"  Instance: {inst.study_case}  ({instance_path})")
    print(f"  Buildings: {inst.n_buildings}  Candidates: {inst.n_candidates}  "
          f"Waste types: {inst.n_waste_types}")
    print("-" * 65)

    start = time.time()
    result = solve_lagrangian(
        inst,
        max_iterations=60000,
        gap_tolerance=0.01,
        no_improve_limit=250,
        verbose=True,
        print_every=250,
    )
    elapsed = time.time() - start

    print("-" * 65)
    print(f"  Completed in {result.n_iterations} iterations ({elapsed:.1f}s)")
    print(f"  Best LB:  {result.best_lb:,.1f}")
    print(f"  Best UB:  {result.best_ub:,.1f}")
    print(f"  Gap:      {result.gap:.2%}")
    if result.best_feasible is not None:
        print(f"  Points open: {int(np.sum(result.best_feasible.z))}")
        print(f"  Total bins:  {int(np.sum(result.best_feasible.x))}")
    else:
        print("  ⚠️  Sin solución factible entregada.")
    print("=" * 65)

    guardar_solucion(result, inst, output_dir, runtime_seconds=elapsed, verbose=True)
    print("=" * 65)


if __name__ == "__main__":
    main()
