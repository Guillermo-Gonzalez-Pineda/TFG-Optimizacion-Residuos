"""
Verificación ejecutable de ``analisis.geo`` (Fase D): densidad_convexhull y
validar_grafo. Patrón del repo (sin pytest).

Comprueba además el AISLAMIENTO de la pila geo: ``import analisis`` no debe
cargar geopandas; solo ``from analisis import geo`` lo hace.

Uso (desde la raíz del repo):
    venv/bin/python tests/verificar_geo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "python"))

# ── 3 (parte 1). Aislamiento: importar el paquete NO arrastra la pila geo ──
import analisis                                    # paquete ligero
from instancia import load_instance
from analisis import rutas                         # ligero (sin geopandas)
GEO_ANTES = "geopandas" in sys.modules             # esperado: False

from analisis import geo                           # AQUÍ se paga la pila geo
GEO_DESPUES = "geopandas" in sys.modules           # esperado: True

TAM = 1500
fallos: list[str] = []


def marca(ok: bool) -> str:
    return "✅" if ok else "❌"


def main() -> None:
    inst = load_instance(str(rutas.ruta_instancia(TAM)))

    print("=" * 74)
    print(f"  VERIFICACIÓN analisis.geo  (tam = {TAM} m)")
    print("=" * 74)

    # ── 1. densidad_convexhull ──
    d = geo.densidad_convexhull(inst)
    area, dens = d["area_km2"], d["densidad_hab_km2"]
    recomputo = inst.total_population / area if area > 0 else float("nan")
    ok_area = area > 0
    ok_dens = abs(dens - recomputo) < 1e-6
    ok_sane = 1000.0 <= dens <= 50000.0
    ok1 = ok_area and ok_dens and ok_sane
    print(f"\n  1. densidad_convexhull:")
    print(f"       area_km2 = {area:.4f}  (>0 {marca(ok_area)})")
    print(f"       densidad_hab_km2 = {dens:.2f}  "
          f"(== total_population/area {marca(ok_dens)} | rango urbano plausible {marca(ok_sane)})")
    if not ok1:
        fallos.append("1 densidad_convexhull")

    # ── 2. validar_grafo ──
    g = geo.validar_grafo(inst)
    coherente = (not g["es_conexo"]) or (g["n_componentes_conexas"] == 1 and g["n_nodos_aislados"] == 0)
    ok2 = g["n_nodos"] > 0 and g["n_aristas"] > 0 and coherente
    print(f"\n  2. validar_grafo (dict completo):")
    for k, v in g.items():
        print(f"       {k:>22} : {v}")
    print(f"       n_nodos>0 & n_aristas>0 & coherencia conexo↔(1 comp, 0 aislados): {marca(ok2)}")
    if not ok2:
        fallos.append("2 validar_grafo")

    # ── 3 (parte 2). Aislamiento de la pila geo ──
    ok3 = (GEO_ANTES is False) and (GEO_DESPUES is True)
    print(f"\n  3. Aislamiento pila geo:")
    print(f"       'geopandas' en sys.modules tras 'import analisis' = {GEO_ANTES}  (esperado False)")
    print(f"       'geopandas' en sys.modules tras 'from analisis import geo' = {GEO_DESPUES}  (esperado True)")
    print(f"       {marca(ok3)}")
    if not ok3:
        fallos.append("3 aislamiento geopandas")

    print("\n" + "=" * 74)
    if fallos:
        print(f"  ❌ FALLOS ({len(fallos)}):")
        for f in fallos:
            print(f"      - {f}")
        sys.exit(1)
    print("  ✅ TODO OK — analisis.geo verificado (densidad, grafo, aislamiento).")
    print("=" * 74)


if __name__ == "__main__":
    main()
