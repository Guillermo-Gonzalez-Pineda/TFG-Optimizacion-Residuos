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

import numpy as np                      # geo-free: solo cálculo numérico (percentiles, stats)

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


def demanda_por_punto(sol: dict, inst: "Instance", k: int = 0) -> dict[int, float]:
    """Demanda (población) que cada punto recibe del tipo de residuo ``k``.

    Para cada edificio ``i`` asignado al punto ``j`` en el tipo ``k``
    (``y_assign[(i, k)] == j``), acumula su población ``inst.I[i].h_i`` en ``j``.
    Devuelve ``{j: demanda}`` solo con los puntos que reciben algo del tipo ``k``.

    Salta las asignaciones no factibles (``j < 0``, p. ej. ``-1`` = sin asignar).
    En una solución factible completa cada edificio asigna su tipo ``k`` a un único
    punto, de modo que ``Σ_j demanda_por_punto(sol, inst, k)[j]`` es la población
    total de la instancia, igual para todo ``k`` (invariante de conservación)."""
    demanda: dict[int, float] = {}
    for (i, kk), j in sol["y_assign"].items():
        if kk != k or j < 0:
            continue
        demanda[j] = demanda.get(j, 0.0) + inst.I[i].h_i
    return demanda


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


# ─────────────────── Métricas de INSTANCIA (geo-free) ───────────────────────
# Operan solo sobre la ``Instance`` (I/J/dij/params), SIN tocar la pila
# geográfica (osmnx/geopandas/shapely/pyproj): eso vive en el módulo ``mapas``.
# Las consume el cuaderno 01 (validación de instancias).
#
# Orientación de ``inst.dij``: ``dij[j][i]`` = distancia del candidato ``j`` al
# edificio ``i`` (clave externa = candidato, interna = edificio; cf. instancia.py).
# Un candidato puede tener su dict interno VACÍO (sin edificios en rango): se
# tolera sin romper (no aporta cobertura ni distancias).


def outliers_demanda_iqr(inst: "Instance", factor: float = 1.5) -> dict[str, Any]:
    """Outliers de demanda por la regla del rango intercuartílico (IQR).

    Marca como outlier todo edificio con ``h_i > Q3 + factor·(Q3 − Q1)`` sobre el
    conjunto ``{h_i}``. Devuelve el umbral, los índices de edificio outlier y el
    peso que su demanda representa sobre el total."""
    h = np.array([edif.h_i for edif in inst.I.values()], dtype=float)
    q1, q3 = np.percentile(h, [25, 75])
    umbral = float(q3 + factor * (q3 - q1))

    indices = [i for i, edif in inst.I.items() if edif.h_i > umbral]
    demanda_total = float(h.sum())
    demanda_outliers = float(sum(inst.I[i].h_i for i in indices))
    pct = demanda_outliers / demanda_total if demanda_total else 0.0

    return {
        "umbral": umbral,
        "indices": indices,
        "n_outliers": len(indices),
        "demanda_total": demanda_total,
        "demanda_outliers": demanda_outliers,
        "pct_demanda_outliers": pct,
    }


def cobertura_por_tipo(inst: "Instance") -> dict[str, Any]:
    """Cobertura por tipo de residuo: cuántos candidatos alcanzan cada edificio
    dentro de ``coverage_radius[k]``.

    Para cada tipo ``k`` cuenta, por edificio, el nº de candidatos ``j`` con
    ``dij[j][i] ≤ coverage_radius[k]`` y resume ``acc_media/acc_min/acc_max`` y
    cuántos edificios quedan sin ningún candidato. Como ``d ≤ r_pequeño`` implica
    ``d ≤ r_grande``, ``edificios_sin_cobertura`` es NO CRECIENTE al crecer el
    radio (invariante que valida de paso la orientación de ``dij``)."""
    cobertura = inst.params.coverage_radius
    # conteo[k][i] = nº de candidatos que cubren el edificio i para el tipo k.
    conteo = {k: {i: 0 for i in inst.I} for k in inst.K}
    for inner in inst.dij.values():                 # inner = {edificio_i: distancia}
        for i, d in inner.items():
            for k in inst.K:
                if d <= cobertura[k]:
                    conteo[k][i] += 1

    por_tipo: dict[int, dict] = {}
    sin_cobertura_algun: set[int] = set()
    for k in inst.K:
        valores = list(conteo[k].values())
        sin_cob = [i for i, c in conteo[k].items() if c == 0]
        por_tipo[k] = {
            "acc_media": sum(valores) / len(valores) if valores else 0.0,
            "acc_min": min(valores) if valores else 0,
            "acc_max": max(valores) if valores else 0,
            "edificios_sin_cobertura": len(sin_cob),
        }
        sin_cobertura_algun.update(sin_cob)

    return {"por_tipo": por_tipo, "sin_cobertura_algun_tipo": len(sin_cobertura_algun)}


def candidatos_reales_vs_artificiales(inst: "Instance") -> dict[str, int]:
    """Reparte los candidatos en reales vs artificiales.

    Un candidato es ARTIFICIAL si su ``osm_id`` empieza por ``"-"`` (id negativo
    insertado sobre una arista para garantizar cobertura); el resto son reales."""
    artificiales = sum(1 for cand in inst.J.values() if str(cand.osm_id).startswith("-"))
    total = len(inst.J)
    return {"total": total, "reales": total - artificiales, "artificiales": artificiales}


def resumen_distancias(inst: "Instance") -> dict[str, Any]:
    """Estadística de las distancias edificio↔candidato de ``inst.dij``.

    ``n_sobre_cutoff`` usa el radio de Dijkstra de la instancia
    (``inst.dijkstra_radius_m``) como cutoff: ninguna distancia debería superarlo."""
    dists = np.array([d for inner in inst.dij.values() for d in inner.values()],
                     dtype=float)
    cutoff = inst.dijkstra_radius_m
    if dists.size == 0:
        return {"n": 0, "min": None, "media": None, "mediana": None, "max": None,
                "n_cero": 0, "n_sub5": 0, "n_sobre_cutoff": 0}
    return {
        "n": int(dists.size),
        "min": float(dists.min()),
        "media": float(dists.mean()),
        "mediana": float(np.median(dists)),
        "max": float(dists.max()),
        "n_cero": int(np.sum(dists == 0)),
        "n_sub5": int(np.sum(dists < 5)),
        "n_sobre_cutoff": int(np.sum(dists > cutoff)),
    }
