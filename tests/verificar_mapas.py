"""
Verificación ejecutable de ``analisis.mapas`` (Fase C.2). Patrón del repo (sin
pytest, cf. verificar_comparativas.py / verificar_metricas.py).

Sobre 500m:
  1. cargar_fondo(500): nº de nodos del grafo y nº de polígonos del gdf.
  2. emparejar_demanda: emparejados / sin-match (esperado casi todos).
  3. Genera y guarda en output/figuras/ (vía rutas.ruta_figura), sin error:
     mapa_instancia, mapa_demanda, mapa_solucion(rich=False) y (rich=True) sobre
     la solución EXACTA de 500m.
  4. Si hay lagrangiana 500m, mapa_solucion(rich=True) sobre ella (MISMA función).
  5. Ninguna figura vacía (nº de artistas > 0).
  6. Color-por-demanda no plano: demanda_por_punto sobre los abiertos con min != max.

Uso (desde la raíz del repo):
    venv/bin/python tests/verificar_mapas.py
"""

from __future__ import annotations

import os
import sys

import matplotlib                       # backend sin display ANTES de pyplot
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "python"))

from instancia import load_instance

from analisis import metricas, rutas
from analisis.carga import cargar_solucion
from analisis import mapas

TAM = 500
RAIZ = rutas.raiz_repo()

fallos: list[str] = []


def _n_artistas(ax) -> int:
    return (len(ax.collections) + len(ax.lines) + len(ax.patches) + len(ax.images))


def _guardar(ax, nombre: str):
    """Guarda la figura de ``ax`` en output/figuras/<nombre>.png y la cierra.
    Devuelve (ruta, nº_artistas)."""
    n = _n_artistas(ax)
    ruta = rutas.ruta_figura(nombre)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ax.figure.savefig(ruta, dpi=120, bbox_inches="tight")
    plt.close(ax.figure)
    if n <= 0:
        fallos.append(f"figura vacía: {nombre}")
    return ruta, n


def main() -> None:
    print("=" * 72)
    print(f"  VERIFICACIÓN analisis.mapas  (tam = {TAM} m)")
    print("=" * 72)

    inst = load_instance(str(rutas.ruta_instancia(TAM)))

    # ── 1. cargar_fondo ──
    grafo, gdf = mapas.cargar_fondo(TAM)
    print(f"\n  1. cargar_fondo({TAM}): "
          f"nodos_grafo={grafo.number_of_nodes()}  polígonos_gdf={len(gdf)}")

    # ── 2. emparejar_demanda ──
    gdf_h = mapas.emparejar_demanda(gdf, inst)
    emparejados = int(gdf_h["h_i"].notna().sum())
    total_edif = len(inst.I)
    sin_match = total_edif - emparejados
    tasa = emparejados / total_edif if total_edif else 0.0
    print(f"\n  2. emparejar_demanda: {emparejados}/{total_edif} emparejados "
          f"({tasa:.1%}) · sin-match={sin_match}")
    if tasa < 0.95:
        fallos.append(f"match rate bajo: {tasa:.1%} ({sin_match} sin match)")

    # ── 3. figuras sobre la EXACTA de 500m ──
    sol_ex = cargar_solucion(RAIZ / "output" / f"exacto_{TAM}m" / "solucion_exacta.json")
    print("\n  3. Figuras (output/figuras/):")
    for nombre, ax in [
        (f"mapa_instancia_{TAM}m",            mapas.mapa_instancia(inst, TAM)),
        (f"mapa_demanda_{TAM}m",              mapas.mapa_demanda(inst, TAM)),
        (f"mapa_solucion_exacta_{TAM}m",      mapas.mapa_solucion(sol_ex, inst, rich=False)),
        (f"mapa_solucion_rich_exacta_{TAM}m", mapas.mapa_solucion(sol_ex, inst, rich=True)),
    ]:
        ruta, n = _guardar(ax, nombre)
        print(f"       {'✅' if n > 0 else '❌'} {ruta.relative_to(RAIZ)}  (artistas={n})")

    # ── 4. lagrangiana 500m con la MISMA función ──
    lag_json = RAIZ / "output" / f"lagrangiana_{TAM}m" / "solucion_lagrangiana.json"
    if lag_json.exists():
        sol_lag = cargar_solucion(lag_json)
        ax = mapas.mapa_solucion(sol_lag, inst, rich=True)
        ruta, n = _guardar(ax, f"mapa_solucion_rich_lagrangiana_{TAM}m")
        print(f"\n  4. MISMA función sobre lagrangiana:")
        print(f"       {'✅' if n > 0 else '❌'} {ruta.relative_to(RAIZ)}  (artistas={n})")
    else:
        print(f"\n  4. (sin lagrangiana en {lag_json.relative_to(RAIZ)}; se omite)")

    # ── 6. color-por-demanda no plano (sobre la exacta) ──
    dpp = metricas.demanda_por_punto(sol_ex, inst, 0)
    vals = list(dpp.values())
    vmin, vmax = (min(vals), max(vals)) if vals else (0.0, 0.0)
    varia = len(set(round(v, 6) for v in vals)) > 1 and vmin != vmax
    print(f"\n  6. Color-por-demanda: {len(vals)} puntos abiertos · "
          f"min={vmin:.2f}  max={vmax:.2f}  → {'varía ✅' if varia else 'PLANO ❌'}")
    if not varia:
        fallos.append("color-por-demanda plano (min == max)")

    print("\n" + "=" * 72)
    if fallos:
        print(f"  ❌ FALLOS ({len(fallos)}):")
        for f in fallos:
            print(f"      - {f}")
        sys.exit(1)
    print("  ✅ TODO OK — analisis.mapas verificado (fondo, emparejamiento, figuras, color).")
    print("=" * 72)


if __name__ == "__main__":
    main()
