"""
Verificador de factibilidad de soluciones del HDM, INDEPENDIENTE del solver.

Port de ``lagrangiana.factibilidad.is_feasible`` a la capa de análisis:
  - opera sobre el DICT CANÓNICO que devuelve ``analisis.carga.cargar_solucion``
    (``z {j:int}``, ``x``/``w`` ``{(j,k):int}``, ``y_assign {(i,k):j}``), no sobre los
    arrays ``FeasibleSolution`` del solver;
  - es SOLVER-FREE y GEO-FREE: no importa gurobipy ni osmnx, y ``compute_demand`` se
    importa de forma PEREZOSA dentro de la función (así ``import analisis`` no arrastra
    ``instancia``);
  - separa explícitamente **(6) cobertura** y **(7) NIMBY**, que ``is_feasible`` funde en
    "candidato válido" (pertenencia a ``valid_candidates``).

Comprueba una solución contra las restricciones del HDM SIN fiarse de cómo se construyó
(no mira el coste ni asume invariantes de reparación): es la verificación que blinda el
análisis de cualquier método (exacto, lagrangiana, greedy, tabú).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:                       # el hint no acopla el import en tiempo de ejecución
    from instancia import Instance


# Restricciones verificadas, en orden de modelo (claves de salida ESTABLES).
CATEGORIAS = (
    "(2) asignación",
    "(3) org=resto",
    "(4) capacidad",
    "(5) límite físico",
    "(6) cobertura",
    "(7) NIMBY",
    "(8) x–w",
    "(9) nearest-alloc",
    "z–x",
)


def candidatos_validos(inst: "Instance") -> dict[int, dict[int, list[int]]]:
    """Para cada edificio ``i`` y tipo ``k``, lista de candidatos ``j`` que cumplen NIMBY
    y cobertura —``nimby_distance ≤ dij[j][i] ≤ coverage_radius[k]``—, ORDENADA por
    distancia ascendente.

    Port puro (sin gurobi) de ``lagrangiana.precompute_valid_candidates``. El orden por
    distancia es lo que usa la restricción (9) nearest-allocation."""
    n_k = inst.n_waste_types
    cobertura = inst.params.coverage_radius
    r0 = inst.params.nimby_distance

    temp: dict[int, dict[int, list[tuple[float, int]]]] = {
        i: {k: [] for k in range(n_k)} for i in inst.I
    }
    for j, inner in inst.dij.items():           # dij[j][i] = distancia candidato→edificio
        for i, d in inner.items():
            for k in range(n_k):
                if r0 <= d <= cobertura[k]:
                    temp[i][k].append((d, j))

    return {
        i: {k: [j for _, j in sorted(temp[i][k])] for k in range(n_k)}
        for i in inst.I
    }


def verificar_hdm(sol: dict, inst: "Instance", max_detalle: int = 10) -> dict[str, Any]:
    """Verifica ``sol`` (dict canónico de ``cargar_solucion``) contra las restricciones
    del HDM. Devuelve::

        {"factible": bool,
         "violaciones": {categoría: nº_violaciones, ...},   # todas las CATEGORIAS
         "detalle":     {categoría: [motivos legibles, ...]}}  # solo las no vacías

    Fiel a ``lagrangiana.is_feasible`` (verifica (2),(3),(4),(5),(8),(9), z–x) con (6) y
    (7) desglosadas. ``max_detalle`` trunca los motivos por categoría."""
    from instancia import compute_demand        # perezoso: mantiene 'import analisis' ligero

    n_k = inst.n_waste_types
    N_j = inst.params.max_bins
    Q = inst.params.bin_capacity
    cobertura = inst.params.coverage_radius
    r0 = inst.params.nimby_distance
    valid = candidatos_validos(inst)

    z = sol["z"]
    x = sol["x"]
    y = sol["y_assign"]
    w = sol.get("w", {})

    det: dict[str, list[str]] = {cat: [] for cat in CATEGORIAS}

    # Bins por punto (para el límite físico (5)).
    bins_punto: dict[int, int] = {}
    for (j, k), v in x.items():
        bins_punto[j] = bins_punto.get(j, 0) + v

    # (5) límite físico: Σ_k x[j,k] ≤ N_j en cada punto ABIERTO.
    for j in inst.J:
        if z.get(j, 0) and bins_punto.get(j, 0) > N_j:
            det["(5) límite físico"].append(f"j={j} Σx={bins_punto[j]}>{N_j}")

    # z–x (bins en punto cerrado) y (8) x>0 ⟹ w=1.
    for (j, k), v in x.items():
        if v > 0 and not z.get(j, 0):
            det["z–x"].append(f"j={j},k={k} x={v} pero z=0")
        if v > 0 and not w.get((j, k), 0):
            det["(8) x–w"].append(f"j={j},k={k} x={v} pero w=0")

    # (2) asignación a candidato ABIERTO y VÁLIDO; (6)/(7) según por qué es inválido.
    for i in inst.I:
        for k in range(n_k):
            vc = valid[i][k]
            if not vc:
                continue                        # (i,k) sin cobertura posible: sin restricción (2)
            j = y.get((i, k), -1)
            if j is None or j < 0:
                det["(2) asignación"].append(f"i={i},k={k} sin asignar")
            elif not z.get(j, 0):
                det["(2) asignación"].append(f"i={i},k={k} j={j} cerrado")
            elif j not in vc:                   # asignado fuera de los válidos: ¿por qué?
                d = inst.dij.get(j, {}).get(i)
                if d is None or d > cobertura[k]:
                    det["(6) cobertura"].append(
                        f"i={i},k={k} j={j} fuera de cobertura (r_{k}={cobertura[k]:.0f} m)")
                else:                           # d < r0
                    det["(7) NIMBY"].append(
                        f"i={i},k={k} j={j} d={d:.0f}<nimby={r0:.0f} m")

    # (3) orgánica (0) y resto (1) al mismo punto.
    for i in inst.I:
        if valid[i][0] and valid[i][1] and y.get((i, 0), -1) != y.get((i, 1), -1):
            det["(3) org=resto"].append(f"i={i} y0={y.get((i, 0))} y1={y.get((i, 1))}")

    # (4) capacidad: demanda asignada ≤ Q_k·x[j,k].
    asignada: dict[tuple[int, int], float] = {}
    for (i, k), j in y.items():
        if j is None or j < 0:
            continue
        asignada[(j, k)] = asignada.get((j, k), 0.0) + compute_demand(inst.I[i].h_i, inst.params, k)
    for (j, k), dem in asignada.items():
        cap = Q[k] * x.get((j, k), 0)
        if dem > cap + 1e-6:
            det["(4) capacidad"].append(f"j={j},k={k} dem={dem:.0f}>{cap:.0f}")

    # (9) nearest-allocation: ningún candidato MÁS CERCANO con bin de ese tipo (w=1).
    for i in inst.I:
        for k in range(n_k):
            j = y.get((i, k), -1)
            if j is None or j < 0:
                continue
            for jc in valid[i][k]:
                if jc == j:
                    break
                if w.get((jc, k), 0):
                    det["(9) nearest-alloc"].append(
                        f"i={i},k={k} usa j={j} habiendo j'={jc} más cerca con bin")
                    break

    violaciones = {cat: len(det[cat]) for cat in CATEGORIAS}
    factible = all(n == 0 for n in violaciones.values())
    detalle = {cat: det[cat][:max_detalle] for cat in CATEGORIAS if det[cat]}
    return {"factible": factible, "violaciones": violaciones, "detalle": detalle}
