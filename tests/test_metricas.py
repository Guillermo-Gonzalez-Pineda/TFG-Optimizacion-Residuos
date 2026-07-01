"""
Tests de ``analisis.metricas`` validados contra los ``*_resumen.json`` existentes,
que ya contienen los valores correctos calculados por el solver.

Estrategia: cargar la solución COMPLETA con ``analisis.carga.cargar_solucion``,
calcular las métricas con ``analisis.metricas.*`` y comparar con los campos del
resumen (``cost``, ``n_points_open``, ``total_bins``, ``bins_per_type``,
``open_points``, ``gap_gurobi``).

Nota sobre rutas: los ficheros siguen en el esquema VIEJO
(``output/exacto_<tam>m/solucion_exacta.json``); la migración al esquema nuevo es
la Fase E. Por eso aquí se construyen las rutas viejas a mano, ancladas a
``rutas.raiz_repo()`` para ser independientes del cwd.
"""

from __future__ import annotations

import json
import re

import pytest

from instancia import load_instance

from analisis import metricas, rutas
from analisis.carga import cargar_solucion


RAIZ = rutas.raiz_repo()


# ── Helpers de carga (esquema viejo, ubicación física actual) ────────────────
def _dir_viejo(tam: int):
    return RAIZ / "output" / f"exacto_{tam}m"


def _cargar_sol(tam: int) -> dict:
    return cargar_solucion(_dir_viejo(tam) / "solucion_exacta.json")


def _cargar_resumen(tam: int) -> dict:
    with open(_dir_viejo(tam) / "solucion_exacta_resumen.json", encoding="utf-8") as f:
        return json.load(f)


def _tams_exactos_disponibles() -> list[int]:
    """Tamaños con solución completa + resumen presentes (esquema viejo)."""
    tams: list[int] = []
    for carpeta in (RAIZ / "output").glob("exacto_*m"):
        casa = re.match(r"exacto_(\d+)m$", carpeta.name)
        if (
            casa
            and (carpeta / "solucion_exacta.json").exists()
            and (carpeta / "solucion_exacta_resumen.json").exists()
        ):
            tams.append(int(casa.group(1)))
    return sorted(tams)


# ── 1. Valores fijados (regresión exacta del enunciado) ──────────────────────
@pytest.mark.parametrize(
    "tam, coste_esperado, n_puntos_esperado",
    [
        (1500, 2265500.0, 403),
        (500, 343070.0, 64),
    ],
)
def test_valores_fijados(tam, coste_esperado, n_puntos_esperado):
    sol = _cargar_sol(tam)
    assert metricas.coste(sol) == coste_esperado
    assert metricas.n_puntos_abiertos(sol) == n_puntos_esperado
    # consistencia interna de las dos vías de contar puntos
    assert len(metricas.puntos_abiertos(sol)) == n_puntos_esperado


# ── 2. Coincidencia con el resumen para TODOS los tamaños disponibles ────────
@pytest.mark.parametrize("tam", _tams_exactos_disponibles())
def test_coincide_con_resumen(tam):
    sol = _cargar_sol(tam)
    res = _cargar_resumen(tam)

    assert metricas.coste(sol) == res["cost"]
    assert metricas.n_puntos_abiertos(sol) == res["n_points_open"]
    assert metricas.puntos_abiertos(sol) == res["open_points"]   # mismo conjunto y orden
    assert metricas.total_bins(sol) == res["total_bins"]
    assert metricas.gap(sol) == res["gap_gurobi"]

    # bins_por_tipo usa claves int; el resumen usa claves string ("0".."3")
    bpt = metricas.bins_por_tipo(sol)
    assert {str(k): v for k, v in bpt.items()} == res["bins_per_type"]


# ── 3. Métricas que requieren la instancia (desglose y violaciones) ──────────
@pytest.mark.parametrize("tam", [500, 1500])
def test_desglose_y_violaciones(tam):
    sol = _cargar_sol(tam)
    inst = load_instance(str(rutas.ruta_instancia(tam)))

    fija, variable = metricas.desglose_coste(sol, inst)
    # El coste del modelo es exactamente apertura fija + contenedores variables.
    assert fija + variable == pytest.approx(metricas.coste(sol))

    # Una solución exacta válida no viola el límite físico de contenedores.
    assert metricas.violaciones_capacidad(sol, inst) == []

    fila = metricas.resumen(sol, inst)
    assert fila["coste"] == metricas.coste(sol)
    assert fila["total_bins"] == metricas.total_bins(sol)
    assert fila["n_violaciones_capacidad"] == 0
