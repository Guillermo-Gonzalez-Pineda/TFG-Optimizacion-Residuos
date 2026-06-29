"""
Barrido secuencial de instancias Laguna.

Ejecutar desde la raíz del repositorio:
    python scripts/barrido_instancias.py

Procesa las instancias 250–1500 m en orden, llamando a solve_lagrangian
con los criterios del TFG (gap 1%, no_improve_stop 500, time_limit 30 min).
Guarda resultados incrementalmente en output/ para no perder trabajo si
se interrumpe.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "python"))

from instancia import load_instance
from lagrangiana import solve_lagrangian, precompute_valid_candidates
from lagrangiana.persistencia import guardar_solucion

# ── Configuración ──────────────────────────────────────────────────────────────
SIZES = [250, 300, 350, 400, 450, 500, 550, 1000, 1500]
INSTANCE_PATTERN = "data/processed/instancia_laguna_{size}m.json"

# Resumen agregado del barrido (una fila por tamaño)
SUMMARY_DIR = "output/barrido_relajacion"
OUTPUT_CSV  = os.path.join(SUMMARY_DIR, "barrido_resultados.csv")
OUTPUT_JSON = os.path.join(SUMMARY_DIR, "barrido_resultados.json")

# La solución COMPLETA de cada tamaño se persiste, en formato espejo del
# exacto, en output/lagrangiana_{size}m/ (vía guardar_solucion).
PER_SIZE_DIR = "output/lagrangiana_{size}m"

SOLVE_PARAMS: dict = dict(
    max_iterations  = 6000,
    gap_tolerance   = 0.01,
    no_improve_limit= 200,
    no_improve_stop = 500,
    time_limit      = 1800.0,
    verbose         = True,
    print_every     = 250,
)

FIELDNAMES = [
    "size_m", "n_buildings", "n_candidates", "n_valid_y",
    "best_lb", "best_ub", "gap", "n_iterations", "stop_reason",
    "wall_time", "time_to_best_lb",
]


def _n_valid_y(valid_candidates: list, n_buildings: int, n_waste_types: int) -> int:
    """Total number of (i, k, j) valid assignment triplets."""
    return sum(
        len(valid_candidates[i][k])
        for i in range(n_buildings)
        for k in range(n_waste_types)
    )


def _save(all_results: list[dict]) -> None:
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_results)


def main() -> None:
    os.makedirs(SUMMARY_DIR, exist_ok=True)

    # Cargar resultados previos para modo append
    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON) as f:
            all_results: list[dict] = json.load(f)
        done_sizes = {r["size_m"] for r in all_results}
        print(f"Resultados previos cargados: {sorted(done_sizes)}")
    else:
        all_results = []
        done_sizes: set[int] = set()

    n_total = len(SIZES)
    for idx, size in enumerate(SIZES):
        instance_path = INSTANCE_PATTERN.format(size=size)

        print(f"\n{'=' * 65}")
        print(f"  [{idx + 1}/{n_total}] {size}m — {instance_path}")
        print(f"{'=' * 65}")

        if not os.path.exists(instance_path):
            print(f"  AVISO: fichero no encontrado, saltando.")
            continue

        if size in done_sizes:
            print(f"  AVISO: {size}m ya procesado, saltando.")
            continue

        inst = load_instance(instance_path)
        print(f"  Edificios: {inst.n_buildings}  Candidatos: {inst.n_candidates}  "
              f"Tipos: {inst.n_waste_types}")

        # Precomputa candidatos válidos para n_valid_y
        # (solve_lagrangian también los precomputa internamente)
        valid_candidates = precompute_valid_candidates(inst)
        n_valid_y = _n_valid_y(valid_candidates, inst.n_buildings, inst.n_waste_types)
        print(f"  Triples válidos (i,k,j): {n_valid_y:,}")

        t_ext_start = time.perf_counter()
        result = solve_lagrangian(inst, **SOLVE_PARAMS)
        t_ext_total = time.perf_counter() - t_ext_start

        row: dict = {
            "size_m":          size,
            "n_buildings":     inst.n_buildings,
            "n_candidates":    inst.n_candidates,
            "n_valid_y":       n_valid_y,
            "best_lb":         round(result.best_lb, 2),
            "best_ub":         round(result.best_ub, 2),
            "gap":             round(result.gap, 6),
            "n_iterations":    result.n_iterations,
            "stop_reason":     result.stop_reason,
            "wall_time":       round(result.wall_time, 2),
            "time_to_best_lb": round(result.time_to_best_lb, 2),
        }
        all_results.append(row)
        done_sizes.add(size)

        # Persistir la SOLUCIÓN COMPLETA del tamaño (formato espejo del exacto)
        guardar_solucion(
            result, inst,
            output_dir=PER_SIZE_DIR.format(size=size),
            runtime_seconds=t_ext_total,
            verbose=False,
        )

        # Guardar incrementalmente — si falla la siguiente instancia no se pierde esto
        _save(all_results)

        print(
            f"\n  [{idx + 1}/{n_total}] {size}m → "
            f"gap {result.gap:.1%}, "
            f"parada por {result.stop_reason}, "
            f"{result.wall_time:.0f}s"
        )
        print(f"  Guardado en {OUTPUT_CSV} y {OUTPUT_JSON}")

    print(f"\n{'=' * 65}")
    print(f"  Barrido completo: {len(all_results)} instancias procesadas.")
    print(f"  CSV : {OUTPUT_CSV}")
    print(f"  JSON: {OUTPUT_JSON}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
