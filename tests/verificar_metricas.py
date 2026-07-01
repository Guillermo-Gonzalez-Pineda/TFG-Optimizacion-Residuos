"""
Verificación ejecutable de ``analisis.metricas`` contra los ``*_resumen.json``.

Equivalente sin pytest de ``tests/test_metricas.py`` (pytest no está instalado en
el entorno; el repo ya usa este patrón dual, cf. ``tests/verificar_refactor.py``).

Carga cada solución exacta con ``analisis.carga.cargar_solucion``, calcula las
métricas y las compara con el resumen. Imprime una tabla y termina con código de
salida 1 si algo no coincide.

Uso (desde la raíz del repo):
    venv/bin/python tests/verificar_metricas.py
"""

from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "python"))

from instancia import load_instance

from analisis import metricas, rutas
from analisis.carga import cargar_solucion


RAIZ = rutas.raiz_repo()

# Valores fijados por el enunciado (regresión dura).
FIJADOS = {1500: (2265500.0, 403), 500: (343070.0, 64)}


def _dir_viejo(tam: int):
    return RAIZ / "output" / f"exacto_{tam}m"


def _tams_disponibles() -> list[int]:
    tams = []
    for carpeta in (RAIZ / "output").glob("exacto_*m"):
        casa = re.match(r"exacto_(\d+)m$", carpeta.name)
        if casa and (carpeta / "solucion_exacta.json").exists() \
                and (carpeta / "solucion_exacta_resumen.json").exists():
            tams.append(int(casa.group(1)))
    return sorted(tams)


def main() -> None:
    tams = _tams_disponibles()
    if not tams:
        print("No hay soluciones exactas en el esquema actual (output/exacto_*m/).")
        sys.exit(1)

    print("=" * 78)
    print("  VERIFICACIÓN metricas.py  vs  *_resumen.json   (exacto)")
    print("=" * 78)
    cabecera = f"  {'tam':>5} | {'coste':>12} {'≟':<2}| {'puntos':>6} {'≟':<2}| {'bins':>5} {'≟':<2}| {'bins_por_tipo':>20} {'≟':<2}| gap"
    print(cabecera)
    print("  " + "-" * 74)

    fallos: list[str] = []

    for tam in tams:
        sol = cargar_solucion(_dir_viejo(tam) / "solucion_exacta.json")
        with open(_dir_viejo(tam) / "solucion_exacta_resumen.json", encoding="utf-8") as f:
            res = json.load(f)

        coste = metricas.coste(sol)
        n_pts = metricas.n_puntos_abiertos(sol)
        n_bins = metricas.total_bins(sol)
        bpt = {str(k): v for k, v in metricas.bins_por_tipo(sol).items()}
        g = metricas.gap(sol)

        ok_coste = coste == res["cost"]
        ok_pts = n_pts == res["n_points_open"] and metricas.puntos_abiertos(sol) == res["open_points"]
        ok_bins = n_bins == res["total_bins"]
        ok_bpt = bpt == res["bins_per_type"]
        ok_gap = g == res["gap_gurobi"]

        def marca(b: bool) -> str:
            return "✅" if b else "❌"

        print(f"  {tam:>5} | {coste:>12,.1f} {marca(ok_coste):<2}| "
              f"{n_pts:>6} {marca(ok_pts):<2}| {n_bins:>5} {marca(ok_bins):<2}| "
              f"{str(list(bpt.values())):>20} {marca(ok_bpt):<2}| {g:.2e}")

        for nombre, ok in [("coste", ok_coste), ("puntos", ok_pts),
                           ("bins", ok_bins), ("bins_por_tipo", ok_bpt), ("gap", ok_gap)]:
            if not ok:
                fallos.append(f"{tam}m: {nombre}")

        # Valores fijados del enunciado
        if tam in FIJADOS:
            coste_esp, n_esp = FIJADOS[tam]
            if coste != coste_esp:
                fallos.append(f"{tam}m: coste fijado {coste} != {coste_esp}")
            if n_pts != n_esp:
                fallos.append(f"{tam}m: puntos fijado {n_pts} != {n_esp}")

    # ── Desglose de coste y violaciones (requieren la instancia) ──
    print("\n  Desglose de coste (apertura + contenedores == coste) y violaciones:")
    for tam in [t for t in (500, 1500) if t in tams]:
        sol = cargar_solucion(_dir_viejo(tam) / "solucion_exacta.json")
        inst = load_instance(str(rutas.ruta_instancia(tam)))
        fija, variable = metricas.desglose_coste(sol, inst)
        viol = metricas.violaciones_capacidad(sol, inst)
        suma_ok = abs((fija + variable) - metricas.coste(sol)) < 1e-6
        print(f"    {tam:>5}m | fija={fija:>12,.1f} + variable={variable:>11,.1f} "
              f"= {fija + variable:>12,.1f}  {'✅' if suma_ok else '❌'} | "
              f"violaciones_capacidad={len(viol)} {'✅' if not viol else '❌'}")
        if not suma_ok:
            fallos.append(f"{tam}m: fija+variable != coste")
        if viol:
            fallos.append(f"{tam}m: {len(viol)} violaciones de capacidad")

    print("\n" + "=" * 78)
    if fallos:
        print(f"  ❌ FALLOS ({len(fallos)}):")
        for f in fallos:
            print(f"      - {f}")
        sys.exit(1)
    print(f"  ✅ TODO COINCIDE — {len(tams)} instancias exactas verificadas contra su resumen.")
    print("=" * 78)


if __name__ == "__main__":
    main()
