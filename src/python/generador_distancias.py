"""
Distancias y garantía de cobertura (módulo 3/4 del generador).

¿POR QUÉ EXISTE? El modelo necesita, para cada candidato j y cada edificio i,
la distancia peatonal d(j, i) (si está dentro del radio de servicio). Aquí:

    - compute_distances : calcula esas distancias con Dijkstra disperso sobre el
                          grafo UTM, interpolando sobre la arista a la que se
                          proyecta cada edificio.
    - evaluate_coverage : detecta qué edificios quedan SIN un contenedor dentro de
                          su radio para algún tipo de residuo.
    - ensure_coverage   : garantiza que todo edificio cubrible reciba un candidato
                          a su alcance, insertando contenedores artificiales
                          justo enfrente de los edificios que se quedan sin uno.

Todo opera sobre graph_utm (metros). Las coordenadas geográficas de los
candidatos artificiales se obtienen reproyectando su punto UTM a lat/lon.

Referencias:
    Li et al. (2026). Waste Management 209, 115211.
"""

from __future__ import annotations

import geopandas as gpd
import networkx as nx
import osmnx as ox
from pyproj import Transformer
from shapely.geometry import LineString, Point

from instancia import (
    BuildingData,
    CandidateContext,
    CandidateData,
    ModelParameters,
)
from generador_grafo import insert_point_on_edge
from generador_extraccion import cost_by_degree


# ══════════════════════════════════════════════════════════════════════
#  Cálculo de distancias edificio-candidato
# ══════════════════════════════════════════════════════════════════════

def compute_distances(
    graph_utm: nx.MultiDiGraph,
    buildings: dict[int, BuildingData],
    candidates: dict[int, CandidateData],
    idx_to_j: dict[int, int],
    cutoff_m: float,
) -> dict[int, dict[int, float]]:
    """Compute sparse Dijkstra distances with edge interpolation (UTM).

    For each building, the connection point P is the projection onto the
    nearest edge. Walking distance = min(d(j,u)+along_u, d(j,v)+along_v) + d_perp,
    where along_u/along_v are the metres along the edge from each endpoint to P.
    """
    # ── Proyectar centroides de edificios a UTM ───────────
    crs_utm = graph_utm.graph["crs"]
    building_points_utm = gpd.GeoSeries(
        [Point(b.longitude, b.latitude) for b in buildings.values()],
        crs="EPSG:4326",
    ).to_crs(crs_utm)
    bpts = list(building_points_utm)
    building_ids = list(buildings.keys())

    # ── Arista más cercana + posición P (precalculado) ────
    building_edges: dict[int, tuple[int, int, float, float, float]] = {}
    for pos, i in enumerate(building_ids):
        pt = bpts[pos]
        u, v, key = ox.distance.nearest_edges(graph_utm, pt.x, pt.y)

        edge_data = graph_utm[u][v][key]
        if "geometry" in edge_data:
            geom = edge_data["geometry"]
        else:
            pu = (graph_utm.nodes[u]["x"], graph_utm.nodes[u]["y"])
            pv = (graph_utm.nodes[v]["x"], graph_utm.nodes[v]["y"])
            geom = LineString([pu, pv])

        along_u = geom.project(pt)         # metros u → P
        along_v = geom.length - along_u    # metros v → P
        d_perp = geom.distance(pt)         # perpendicular real
        building_edges[i] = (u, v, along_u, along_v, d_perp)

    # ── Dijkstra desde cada candidato + fórmula interpolada ──
    distances: dict[int, dict[int, float]] = {idx: {} for idx in candidates.keys()}
    for c_idx, c_node in idx_to_j.items():
        reachable = nx.single_source_dijkstra_path_length(
            graph_utm, c_node, cutoff=cutoff_m, weight="length",
        )
        for b_idx, (u, v, along_u, along_v, d_perp) in building_edges.items():
            dist_via_u = reachable.get(u, float("inf")) + along_u
            dist_via_v = reachable.get(v, float("inf")) + along_v
            best = min(dist_via_u, dist_via_v) + d_perp
            if best <= cutoff_m:
                distances[c_idx][b_idx] = best
    return distances


def evaluate_coverage(
    buildings: dict[int, BuildingData],
    dij: dict[int, dict[int, float]],
    params: ModelParameters,
) -> dict[int, list[int]]:
    """Devuelve {idx_edificio: [tipos de residuo sin cobertura]}.

    Un edificio está cubierto para el tipo k si ALGÚN candidato lo alcanza dentro
    del radio r_k. Un dict vacío significa cobertura total (no hay que insertar
    contenedores artificiales).

    Recibe edificios, las distancias dij y los parámetros (con coverage_radius).
    Devuelve solo los edificios con algún tipo sin cubrir.
    """
    # ── BLOQUE 1 — Invertir dij a {edificio: {candidato: dist}} ───────
    # dij está indexado por candidato; para razonar "por edificio" lo invertimos.
    reachable_from_building: dict[int, dict[int, float]] = {i: {} for i in buildings}
    for j_idx, building_distances in dij.items():
        for i_idx, dist in building_distances.items():
            reachable_from_building[i_idx][j_idx] = dist

    # ── BLOQUE 2 — Por edificio, qué tipos quedan sin cubrir ──────────
    uncovered: dict[int, list[int]] = {}
    for i_idx in buildings:
        missing_types: list[int] = []
        for k, r_k in params.coverage_radius.items():
            covered = any(
                dist <= r_k for dist in reachable_from_building[i_idx].values()
            )
            if not covered:
                missing_types.append(k)
        if missing_types:
            uncovered[i_idx] = missing_types

    return uncovered


# ══════════════════════════════════════════════════════════════════════
#  Garantía de cobertura: contenedores artificiales
# ══════════════════════════════════════════════════════════════════════

# Margen de seguridad (metros) para decidir si un edificio ya está "cubierto de
# sobra" por un contenedor artificial previo y no necesita el suyo. Evita colocar
# contenedores justo en el límite del radio, donde el recálculo de distancias
# (que usa la geometría ya partida) podría dejar el edificio marginalmente fuera.
_COVERAGE_SAFETY_M: float = 2.0


def ensure_coverage(
    graph_utm: nx.MultiDiGraph,
    buildings: dict[int, BuildingData],
    candidates: dict[int, CandidateData],
    dij: dict[int, dict[int, float]],
    params: ModelParameters,
    idx_to_j: dict[int, int],
    j_to_idx: dict[int, int],
    cutoff_m: float,
) -> tuple[
    dict[int, BuildingData],
    dict[int, CandidateData],
    dict[int, dict[int, float]],
    dict[int, int],
    dict[int, int],
]:
    """Garantiza que todo edificio CUBRIBLE reciba un candidato a su alcance.

    ESTRATEGIA. Para un edificio que ningún contenedor alcanza dentro de su radio,
    NO basta con poner un candidato en el cruce más cercano (puede quedar a
    100-200 m calle abajo). Colocamos un contenedor artificial SOBRE la arista, en
    el PIE DE PERPENDICULAR del propio edificio: así su distancia al contenedor es
    exactamente d_perp (la perpendicular a la calle), que por construcción es
    menor que el radio. Esto da un margen real y robusto frente al recálculo de
    distancias —el fallo de la versión anterior, que colocaba el contenedor en el
    extremo del intervalo de cobertura (a distancia EXACTA = radio) y, al
    recalcular sobre la geometría ya partida, lo dejaba marginalmente fuera y
    acababa eliminando edificios perfectamente cubribles.

    Para no multiplicar contenedores, varios edificios cercanos sobre la misma
    calle COMPARTEN uno: solo se inserta un contenedor nuevo para un edificio si
    ninguno de los ya colocados lo cubre con margen (_COVERAGE_SAFETY_M).

    Solo se eliminan al final los edificios genuinamente inalcanzables (d_perp ≥
    radio incluso proyectando justo enfrente).

    Recibe el grafo UTM (se modifica in situ), edificios, candidatos, distancias,
    parámetros, los mapas de índices y el cutoff de Dijkstra.
    Devuelve (buildings, candidates, dij, idx_to_j, j_to_idx) actualizados.
    """
    # ══════════════════════════════════════════════════════════════════
    #  BLOQUE 1 — ¿Hay edificios sin cobertura?
    # ══════════════════════════════════════════════════════════════════
    # Si todos están cubiertos, no tocamos el grafo ni recomputamos nada.
    uncovered = evaluate_coverage(buildings, dij, params)
    if not uncovered:
        return buildings, candidates, dij, idx_to_j, j_to_idx

    crs_utm = graph_utm.graph["crs"]
    # Transformador UTM → lat/lon para las coordenadas de los candidatos nuevos.
    # always_xy → transform(x, y) devuelve (lon, lat).
    transformer = Transformer.from_crs(crs_utm, "EPSG:4326", always_xy=True)

    # ══════════════════════════════════════════════════════════════════
    #  BLOQUE 2 — Arista más cercana y proyección de cada huérfano
    # ══════════════════════════════════════════════════════════════════
    # Trabajamos sobre la ARISTA (no el nodo) porque el contenedor debe caer justo
    # enfrente del edificio. nearest_edges se llama UNA vez en lote (mucho más
    # rápido que por edificio, que era un cuello de botella de la versión previa).
    orphan_ids = list(uncovered.keys())
    orphan_points_utm = list(
        gpd.GeoSeries(
            [Point(buildings[i].longitude, buildings[i].latitude) for i in orphan_ids],
            crs="EPSG:4326",
        ).to_crs(crs_utm)
    )
    nearest = ox.distance.nearest_edges(
        graph_utm,
        [p.x for p in orphan_points_utm],
        [p.y for p in orphan_points_utm],
    )

    # Por huérfano guardamos su arista más cercana, su punto UTM y el radio MÁS
    # ESTRICTO que debe satisfacer (el menor radio entre sus tipos sin cubrir).
    orphan_info: dict[int, tuple] = {}
    for pos, b_idx in enumerate(orphan_ids):
        u, v, key = nearest[pos]
        pt = orphan_points_utm[pos]
        r_strict = min(params.coverage_radius[k] for k in uncovered[b_idx])
        orphan_info[b_idx] = (u, v, key, pt, r_strict)

    # ══════════════════════════════════════════════════════════════════
    #  BLOQUE 3 — Agrupar huérfanos por arista compartida
    # ══════════════════════════════════════════════════════════════════
    # Edificios que proyectan sobre la MISMA calle pueden compartir contenedor.
    # Agrupamos por frozenset({u, v}) (insensible al sentido): en una calle
    # bidireccional nearest_edges puede devolver (u,v) a un edificio y (v,u) a su
    # vecino, pero físicamente es el mismo tramo. Tomamos como referencia la
    # arista (y geometría) del PRIMER huérfano del grupo y reproyectamos a todos
    # sobre ella para medir un 'along' homogéneo.
    groups: dict[frozenset, dict] = {}
    for b_idx, (u, v, key, pt, r_strict) in orphan_info.items():
        edge_id = frozenset((u, v))
        if edge_id not in groups:
            geom = _edge_geometry(graph_utm, u, v, key)
            groups[edge_id] = {"ref": (u, v, key, geom), "members": []}
        groups[edge_id]["members"].append((b_idx, pt, r_strict))

    # ══════════════════════════════════════════════════════════════════
    #  BLOQUE 4 — Decidir posiciones (greedy anclado al pie de cada edificio)
    # ══════════════════════════════════════════════════════════════════
    next_j_idx = max(candidates.keys()) + 1 if candidates else 0
    next_point_id = -1   # ids negativos: nunca colisionan con osmids (positivos)
    added = 0

    for group in groups.values():
        u0, v0, key0, geom0 = group["ref"]
        L0 = geom0.length

        # Proyectar cada miembro sobre la geometría de referencia: along (metros
        # desde u0) y d_perp (perpendicular real). Descartar los inalcanzables
        # (d_perp ≥ radio: ni justo enfrente entrarían; se eliminan en BLOQUE 7).
        members = []
        for b_idx, pt, r_strict in group["members"]:
            along = geom0.project(pt)
            d_perp = geom0.distance(pt)
            if d_perp < r_strict:
                members.append((along, d_perp, r_strict))
        if not members:
            continue

        # Greedy: recorriendo los edificios por posición, colocamos un contenedor
        # en el PIE de cada uno salvo que algún contenedor ya colocado lo cubra
        # con margen (|Δalong| + d_perp ≤ radio − seguridad). Anclar al pie da a
        # cada edificio distancia ≈ d_perp < radio, con margen frente al recálculo.
        members.sort(key=lambda m: m[0])
        placed: list[float] = []
        for along, d_perp, r_strict in members:
            already = any(
                abs(p - along) + d_perp <= r_strict - _COVERAGE_SAFETY_M
                for p in placed
            )
            if not already:
                placed.append(along)

        # Separar las posiciones de los nodos extremos y entre sí (evita sub-tramos
        # degenerados al partir). El set deduplica posiciones que coincidan.
        eps = min(0.5, L0 / 2)
        positions = sorted({min(max(p, eps), L0 - eps) for p in placed})

        # ══════════════════════════════════════════════════════════════
        #  BLOQUE 5 — Insertar cada contenedor y registrar su candidato
        # ══════════════════════════════════════════════════════════════
        # Insertamos en orden DESCENDENTE de 'along'. Al partir (u0→right_node) en
        # P, el sub-tramo izquierdo (u0→P) conserva la parametrización desde u0,
        # así que la siguiente posición (menor) sigue siendo un 'along' válido
        # sobre él y basta encadenar la inserción sobre el nuevo extremo derecho.
        right_node = v0
        cur_key = key0
        for p in reversed(positions):
            pid = next_point_id
            next_point_id -= 1
            insert_point_on_edge(graph_utm, u0, right_node, cur_key, p, pid)

            # Coordenadas lat/lon reproyectando el punto P de UTM a EPSG:4326.
            x, y = graph_utm.nodes[pid]["x"], graph_utm.nodes[pid]["y"]
            lon, lat = transformer.transform(x, y)
            candidates[next_j_idx] = CandidateData(
                osm_id=str(pid),
                latitude=lat,
                longitude=lon,
                context=CandidateContext.STREET,
                # Artificial = extremo de calle (grado 1): coste de grado 1.
                opening_cost=cost_by_degree(1, params.opening_cost),
            )
            idx_to_j[next_j_idx] = pid
            j_to_idx[pid] = next_j_idx
            next_j_idx += 1
            added += 1

            # El sub-tramo izquierdo pasa a ser (u0 → pid) con su nueva key.
            right_node = pid
            cur_key = next(iter(graph_utm[u0][pid]))

    # ══════════════════════════════════════════════════════════════════
    #  BLOQUE 6 — Recomputar distancias si se añadieron contenedores
    # ══════════════════════════════════════════════════════════════════
    # compute_distances re-localiza cada edificio sobre el grafo YA modificado:
    # los cercanos a un P insertado proyectan sobre su sub-arista y obtienen
    # reachable[P] = 0 → distancia ≈ d_perp (exacta).
    if added > 0:
        dij = compute_distances(graph_utm, buildings, candidates, idx_to_j, cutoff_m)

    # ══════════════════════════════════════════════════════════════════
    #  BLOQUE 7 — Eliminar edificios aún inalcanzables y re-indexar
    # ══════════════════════════════════════════════════════════════════
    # Lo que quede sin cubrir es genuinamente inalcanzable (sin calle a ≤ radio).
    uncovered = evaluate_coverage(buildings, dij, params)
    if uncovered:
        # Re-indexar contiguamente para no dejar huecos en los índices de edificio.
        old_to_new: dict[int, int] = {
            old_idx: new_idx
            for new_idx, old_idx in enumerate(
                sorted(i for i in buildings if i not in uncovered)
            )
        }
        new_buildings: dict[int, BuildingData] = {
            new_idx: buildings[old_idx] for old_idx, new_idx in old_to_new.items()
        }
        dij = {
            j: {
                old_to_new[i]: dist
                for i, dist in building_distances.items()
                if i in old_to_new
            }
            for j, building_distances in dij.items()
        }
        buildings = new_buildings
        print(f"Removed {len(uncovered)} unreachable buildings from instance")

    return buildings, candidates, dij, idx_to_j, j_to_idx


# ── Helper interno ─────────────────────────────────────────────────────

def _edge_geometry(
    graph_utm: nx.MultiDiGraph,
    u: int,
    v: int,
    key: int,
) -> LineString:
    """Geometría de la arista (u,v,key): la explícita o la recta u→v."""
    edge_data = graph_utm[u][v][key]
    if "geometry" in edge_data:
        return edge_data["geometry"]
    pu = (graph_utm.nodes[u]["x"], graph_utm.nodes[u]["y"])
    pv = (graph_utm.nodes[v]["x"], graph_utm.nodes[v]["y"])
    return LineString([pu, pv])
