"""
Verificación ejecutable de ``analisis.comparativas`` y ``analisis.estilo``.

Patrón del repo (sin pytest, cf. tests/verificar_refactor.py / verificar_metricas.py).

Carga las soluciones del layout VIEJO (output/exacto_<tam>m/..., output/lagrangiana_500m/...)
—el glob del layout viejo vive AQUÍ, no en la librería: ``rutas.tamaños_disponibles``
devuelve [] a propósito hasta la Fase E— construye la tabla resumen, cruza sus
valores con ``metricas`` y genera las figuras comparativas.

Uso (desde la raíz del repo):
    venv/bin/python tests/verificar_comparativas.py
"""

from __future__ import annotations

import os
import re
import sys

import matplotlib            # backend sin display ANTES de importar pyplot
matplotlib.use("Agg")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "python"))

from instancia import load_instance

from analisis import estilo, metricas, rutas
from analisis.carga import cargar_solucion
from analisis.comparativas import (
    Solucion,
    tabla_resumen,
    tabla_matplotlib,
    grafico_escalabilidad,
    comparar_metodos,
    grafico_convergencia,
)

RAIZ = rutas.raiz_repo()
FIJADOS = {("exacto", 1500): (2265500.0, 403), ("exacto", 500): (343070.0, 64)}

fallos: list[str] = []


def _inst(tam: int):
    ruta = rutas.ruta_instancia(tam)
    return load_instance(str(ruta)) if ruta.exists() else None


def _cargar_soluciones() -> tuple[list[Solucion], Solucion]:
    """(soluciones exactas del layout viejo, solución lagrangiana 500m)."""
    exactos: list[Solucion] = []
    for carpeta in sorted((RAIZ / "output").glob("exacto_*m")):
        casa = re.match(r"exacto_(\d+)m$", carpeta.name)
        sol_json = carpeta / "solucion_exacta.json"
        if casa and sol_json.exists():
            tam = int(casa.group(1))
            exactos.append(Solucion("exacto", tam, cargar_solucion(sol_json), _inst(tam)))

    lag_json = RAIZ / "output" / "lagrangiana_500m" / "solucion_lagrangiana.json"
    lag500 = Solucion("lagrangiana", 500, cargar_solucion(lag_json), _inst(500))
    return exactos, lag500


def _guardar(obj, nombre: str):
    """Guarda la figura de `obj` (Axes o array de Axes) en output/figuras/<nombre>.png."""
    import numpy as np
    eje0 = np.atleast_1d(obj).ravel()[0]
    ruta = rutas.ruta_figura(nombre)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    eje0.figure.savefig(ruta, dpi=120, bbox_inches="tight")
    matplotlib.pyplot.close(eje0.figure)
    return ruta


def main() -> None:
    # ── 0. estilo: PALETA_METODOS debe cubrir las claves del REGISTRO ──
    if set(estilo.PALETA_METODOS) != set(rutas.REGISTRO):
        fallos.append(
            f"PALETA_METODOS {set(estilo.PALETA_METODOS)} != REGISTRO {set(rutas.REGISTRO)}"
        )
    print(f"  estilo.PALETA_METODOS claves == rutas.REGISTRO claves: "
          f"{set(estilo.PALETA_METODOS) == set(rutas.REGISTRO)}")

    # ── 1. Cargar soluciones (layout viejo) ──
    exactos, lag500 = _cargar_soluciones()
    sols = exactos + [lag500]
    print(f"  Soluciones cargadas: {len(exactos)} exactas + 1 lagrangiana = {len(sols)}")

    # ── 2. tabla_resumen ──
    df = tabla_resumen(sols)
    print("\n  tabla_resumen(sols).head():")
    print(df.head().to_string(index=False))
    print(f"\n  tabla_resumen(sols).shape: {df.shape}")

    # ── 3. Cruce tabla vs metricas (deben casar) ──
    indice = {(r.metodo, r.tam): r for r in df.itertuples(index=False)}
    for s in sols:
        fila = indice.get((s.metodo, s.tam))
        if fila is None:
            fallos.append(f"falta fila {s.metodo} {s.tam}")
            continue
        if fila.coste != metricas.coste(s.datos):
            fallos.append(f"{s.metodo} {s.tam}: coste tabla {fila.coste} != metricas {metricas.coste(s.datos)}")
        if fila.n_puntos != metricas.n_puntos_abiertos(s.datos):
            fallos.append(f"{s.metodo} {s.tam}: n_puntos discrepa")
        if fila.n_bins != metricas.total_bins(s.datos):
            fallos.append(f"{s.metodo} {s.tam}: n_bins discrepa")
        # Valores fijados
        if (s.metodo, s.tam) in FIJADOS:
            coste_esp, n_esp = FIJADOS[(s.metodo, s.tam)]
            if fila.coste != coste_esp or fila.n_puntos != n_esp:
                fallos.append(f"{s.metodo} {s.tam}: fijado ({fila.coste},{fila.n_puntos}) != ({coste_esp},{n_esp})")
    print("  Cruce tabla_resumen ↔ metricas (coste/n_puntos/n_bins): "
          f"{'✅ casa' if not fallos else '❌ ver fallos'}")

    # ── 4. comparar_metodos: distancia al óptimo exacto ──
    dfc = comparar_metodos({"exacto": exactos, "lagrangiana": [lag500]}, referencia="exacto")
    print("\n  comparar_metodos(referencia='exacto'):")
    print(dfc.to_string(index=False))
    fila_500 = dfc[(dfc["metodo"] == "lagrangiana") & (dfc["tam"] == 500)]
    if not fila_500.empty:
        print(f"\n  → gap_vs_ref lagrangiana 500m = {fila_500.iloc[0]['gap_vs_ref']:.6f}")

    # ── 5. Figuras a output/figuras/ ──
    print("\n  Generando figuras:")
    try:
        r1 = _guardar(grafico_escalabilidad(sols), "escalabilidad_comparativa")
        print(f"    ✅ {r1.relative_to(RAIZ)}")
        r2 = _guardar(tabla_matplotlib(df, titulo="Resumen comparativo de soluciones"),
                      "tabla_resumen")
        print(f"    ✅ {r2.relative_to(RAIZ)}")
        r3 = _guardar(grafico_convergencia(lag500), "convergencia_lagrangiana")
        print(f"    ✅ {r3.relative_to(RAIZ)}")
    except Exception as exc:                       # pragma: no cover
        fallos.append(f"generación de figuras: {exc!r}")
        print(f"    ❌ {exc!r}")

    print("\n" + "=" * 70)
    if fallos:
        print(f"  ❌ FALLOS ({len(fallos)}):")
        for f in fallos:
            print(f"      - {f}")
        sys.exit(1)
    print("  ✅ TODO OK — comparativas y estilo verificados.")
    print("=" * 70)


if __name__ == "__main__":
    main()
