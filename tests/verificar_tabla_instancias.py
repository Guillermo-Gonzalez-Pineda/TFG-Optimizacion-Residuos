"""
Verificación ejecutable de ``comparativas.tabla_instancias`` (Fase D). Patrón del
repo (sin pytest).

Demuestra el reparto de responsabilidades: el VERIFICADOR (haciendo de cuaderno)
llama a ``metricas.*`` y ``geo.*`` para calcular; ``comparativas`` solo FORMATEA
(geo-free). Se comprueba además que importar ``comparativas``+``metricas`` NO
arrastra la pila geo (geopandas/osmnx) — eso solo lo trae ``geo``.

Uso (desde la raíz del repo):
    venv/bin/python tests/verificar_tabla_instancias.py
"""

from __future__ import annotations

import inspect
import math
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "python"))

# ── comparativas + metricas primero: deben ser geo-free ──
from analisis import comparativas, metricas
GEO_LIBRE_TRAS_COMPARATIVAS = ("geopandas" not in sys.modules
                               and "osmnx" not in sys.modules)

from instancia import load_instance
from analisis import rutas
from analisis import geo               # AQUÍ (en el verificador) se paga la pila geo

TAMS = [500, 1000, 1500]
fallos: list[str] = []


def marca(ok: bool) -> str:
    return "✅" if ok else "❌"


def _fila_de(inst, tam: int) -> dict:
    """Arma el dict-fila llamando a las funciones REALES (rol del cuaderno)."""
    return {
        "tam": tam,
        "n_buildings": inst.n_buildings,
        "n_candidates": inst.n_candidates,
        "total_population": inst.total_population,
        "cobertura": metricas.cobertura_por_tipo(inst),
        "outliers": metricas.outliers_demanda_iqr(inst),
        "candidatos": metricas.candidatos_reales_vs_artificiales(inst),
        "distancias": metricas.resumen_distancias(inst),
        "densidad": geo.densidad_convexhull(inst),
        "grafo": geo.validar_grafo(inst),
    }


def main() -> None:
    print("=" * 78)
    print(f"  VERIFICACIÓN comparativas.tabla_instancias  (tams = {TAMS})")
    print("=" * 78)

    # ── 1. Construir filas con cómputos reales y armar la tabla ──
    filas = []
    for tam in TAMS:
        inst = load_instance(str(rutas.ruta_instancia(tam)))
        filas.append(_fila_de(inst, tam))
    df = comparativas.tabla_instancias(filas)

    # ── 2. shape + print ──
    n_cols = len(comparativas._COLS_INSTANCIAS)
    ok_shape = df.shape == (len(TAMS), n_cols)
    print(f"\n  2. df.shape = {df.shape}  (esperado ({len(TAMS)}, {n_cols}))  {marca(ok_shape)}\n")
    import pandas as pd
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)
    print(df.to_string(index=False))
    if not ok_shape:
        fallos.append("2 shape")

    # ── 3. Coherencia por fila ──
    print("\n  3. Coherencia por fila:")
    num_cols = ["tam", "n_edificios", "n_candidatos", "poblacion", "reales",
                "artificiales", "sin_cobertura_algun_tipo", "n_outliers",
                "pct_demanda_outliers", "dist_media", "dist_max", "area_km2",
                "densidad_hab_km2", "aristas_largas"]
    for _, r in df.iterrows():
        cand_ok = (r["reales"] + r["artificiales"]) == r["n_candidatos"]
        dens_ok = math.isclose(r["densidad_hab_km2"], r["poblacion"] / r["area_km2"],
                               rel_tol=1e-3)
        nan_ok = not any(isinstance(r[c], float) and math.isnan(r[c]) for c in num_cols)
        fila_ok = cand_ok and dens_ok and nan_ok
        print(f"     tam={int(r['tam']):>5} | reales+artif==n_cand {marca(cand_ok)} | "
              f"densidad≈pobl/area {marca(dens_ok)} | sin NaN inesperado {marca(nan_ok)}  {marca(fila_ok)}")
        if not fila_ok:
            fallos.append(f"3 coherencia tam={int(r['tam'])}")

    # ── 4. comparativas es geo-free ──
    mod = sys.modules[comparativas.tabla_instancias.__module__]
    fuente = inspect.getsource(mod)
    prohibidos = [ln.strip() for ln in fuente.splitlines()
                  if re.match(r"\s*(import|from)\s", ln)
                  and re.search(r"\b(osmnx|geopandas|shapely|pyproj)\b|\.(geo|mapas)\b", ln)]
    ok_modulo = comparativas.tabla_instancias.__module__ == "analisis.comparativas"
    ok4 = not prohibidos and ok_modulo and GEO_LIBRE_TRAS_COMPARATIVAS
    print(f"\n  4. comparativas geo-free:")
    print(f"       __module__ = {comparativas.tabla_instancias.__module__}  {marca(ok_modulo)}")
    print(f"       imports pesados en el módulo: {prohibidos or 'ninguno'}  {marca(not prohibidos)}")
    print(f"       'geopandas'/'osmnx' ausentes tras importar comparativas+metricas: "
          f"{GEO_LIBRE_TRAS_COMPARATIVAS}  {marca(GEO_LIBRE_TRAS_COMPARATIVAS)}")
    if not ok4:
        fallos.append("4 geo-free comparativas")

    print("\n" + "=" * 78)
    if fallos:
        print(f"  ❌ FALLOS ({len(fallos)}):")
        for f in fallos:
            print(f"      - {f}")
        sys.exit(1)
    print("  ✅ TODO OK — tabla_instancias verificada (shape, coherencia, geo-free).")
    print("=" * 78)


if __name__ == "__main__":
    main()
