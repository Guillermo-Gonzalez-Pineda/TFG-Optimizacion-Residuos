"""
Métricas puras sobre una solución del problema de localización de contenedores.

Funciones SIN efectos visuales (no dibujan nada) y SIN dependencias de gurobipy,
osmnx ni matplotlib. Operan sobre:

  - ``sol``: el diccionario que devuelve ``analisis.carga.cargar_solucion``, con
    ``z`` {j: int}, ``x``/``w``/``y_assign`` {(·, ·): int} y campos escalares
    (``cost``, ``gap`` o ``gap_gurobi``...).
  - ``inst``: una ``Instance`` de ``instancia.load_instance`` (solo donde hace
    falta el coste de apertura, el coste de contenedor o el límite físico).

Todo está diseñado AGNÓSTICO al método: las mismas funciones sirven para exacto,
lagrangiana, greedy y tabú, porque consumen el formato común de solución.

Factoriza las métricas que hoy están inline y duplicadas en los cuadernos 02 y 03
(sección 1.7 del plan).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:                       # evita acoplar el import en tiempo de ejecución
    from instancia import Instance


def puntos_abiertos(sol: dict) -> list[int]:
    """Índices ``j`` de los puntos abiertos (``z[j] == 1``), en orden ascendente."""
    return sorted(j for j, abierto in sol["z"].items() if abierto == 1)


def n_puntos_abiertos(sol: dict) -> int:
    """Número de puntos abiertos."""
    return sum(1 for abierto in sol["z"].values() if abierto == 1)


def total_bins(sol: dict) -> int:
    """Número total de contenedores instalados (suma de ``x[(j, k)]``)."""
    return int(sum(sol["x"].values()))


def bins_por_tipo(sol: dict, n_k: int = 4) -> dict[int, int]:
    """Contenedores instalados por tipo de residuo: ``{k: Σ_j x[(j, k)]}``."""
    conteo = {k: 0 for k in range(n_k)}
    for (_, k), bins in sol["x"].items():
        if k in conteo:
            conteo[k] += bins
    return conteo


def coste(sol: dict) -> float | None:
    """Coste total de la solución (``None`` si no hay solución factible)."""
    return sol.get("cost")


def gap(sol: dict) -> float | None:
    """Gap de optimalidad, agnóstico al método: usa ``gap`` (lagrangiana) o
    ``gap_gurobi`` (exacto). Devuelve ``None`` si la solución no reporta gap."""
    if sol.get("gap") is not None:
        return sol["gap"]
    return sol.get("gap_gurobi")


def desglose_coste(sol: dict, inst: "Instance") -> tuple[float, float]:
    """Descompone el coste en (apertura_fija, contenedores_variable).

    - fija     = Σ_{j abierto} coste_apertura(j)
    - variable = Σ_{(j, k)} x[(j, k)] · coste_contenedor(k)

    Para una solución factible del modelo, ``fija + variable == coste(sol)``."""
    fija = sum(inst.J[j].opening_cost for j in puntos_abiertos(sol))
    bin_cost = inst.params.bin_cost
    variable = sum(bins * bin_cost[k] for (j, k), bins in sol["x"].items())
    return float(fija), float(variable)


def violaciones_capacidad(sol: dict, inst: "Instance") -> list[dict]:
    """Puntos que violan el límite físico de contenedores: ``Σ_k x[(j, k)] >
    max_bins``. Devuelve una lista de dicts (uno por punto infractor), en orden
    ascendente de punto.

    Útil sobre todo para greedy/tabú, que pueden entregar soluciones infactibles;
    para una solución exacta/lagrangiana válida la lista está vacía."""
    max_bins = inst.params.max_bins

    totales: dict[int, int] = {}
    for (j, _), bins in sol["x"].items():
        totales[j] = totales.get(j, 0) + bins

    violaciones = [
        {
            "punto": j,
            "bins_total": total,
            "max_bins": max_bins,
            "abierto": bool(sol["z"].get(j, 0)),
        }
        for j, total in totales.items()
        if total > max_bins
    ]
    return sorted(violaciones, key=lambda v: v["punto"])


def resumen(sol: dict, inst: "Instance" | None = None) -> dict[str, Any]:
    """Una fila de métricas para tablas comparativas, agnóstica al método:
    coste, nº de puntos abiertos, total de contenedores, desglose por tipo, gap y
    nº de violaciones de capacidad.

    Si ``inst`` es ``None``, omite las métricas que requieren la instancia
    (desglose por tipo usa ``n_k=4`` por defecto y no se calculan violaciones)."""
    n_k = inst.n_waste_types if inst is not None else 4
    violaciones = violaciones_capacidad(sol, inst) if inst is not None else []
    return {
        "coste": coste(sol),
        "n_puntos_abiertos": n_puntos_abiertos(sol),
        "total_bins": total_bins(sol),
        "bins_por_tipo": bins_por_tipo(sol, n_k),
        "gap": gap(sol),
        "n_violaciones_capacidad": len(violaciones),
    }
