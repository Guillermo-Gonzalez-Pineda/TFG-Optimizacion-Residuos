"""
Extracción de candidatos y edificios (módulo 2/4 del generador).

¿POR QUÉ EXISTE? Una vez tenemos el grafo de calles (módulo generador_grafo),
hay que sacar de OSM los dos conjuntos del problema de localización:

    - CANDIDATOS (J): puntos donde PODRÍA ir un contenedor. Salen de los nodos de
      la red (cruces y extremos de calle). Funciones: extract_candidates,
      classify_candidate_context, consolidate_candidates, cost_by_degree.
    - EDIFICIOS (I): puntos de demanda. Salen de las huellas (polígonos) de los
      edificios; su "población" h_i se estima por superficie. Función:
      extract_buildings.

Todo aquí trabaja en lat/lon (EPSG:4326); las distancias en metros son cosa del
módulo generador_distancias.

Referencias:
    Li et al. (2026). Waste Management 209, 115211.
"""

from __future__ import annotations

import math
from dataclasses import replace

import geopandas as gpd
import networkx as nx
import osmnx as ox
from shapely import STRtree
from shapely.geometry import Point
from shapely.ops import unary_union

from instancia import (
    BuildingData,
    CandidateContext,
    CandidateData,
    GeographicConfig,
)


# ══════════════════════════════════════════════════════════════════════
#  Coste de apertura por grado del nodo
# ══════════════════════════════════════════════════════════════════════

def cost_by_degree(street_count: int, base_cost: float) -> float:
    """Coste de apertura según el grado del nodo (nº de calles que confluyen).

    Cruces (grado alto) son más baratos: más espacio de acera, mejor acceso del
    camión. Tramos y artificiales son más caros. Romper la homogeneidad de costes
    también mejora la convergencia de la relajación lagrangiana.

    Recibe el grado del nodo y el coste base; devuelve el coste ajustado.
    """
    if street_count >= 4:
        factor = 0.95
    elif street_count == 3:
        factor = 1.00
    elif street_count == 2:
        factor = 1.15
    else:  # grado 1 o artificial
        factor = 1.30
    return base_cost * factor


# ══════════════════════════════════════════════════════════════════════
#  Candidatos: extracción desde los nodos de la red
# ══════════════════════════════════════════════════════════════════════

def extract_candidates(
    graph: nx.MultiDiGraph,
    config: GeographicConfig,
) -> tuple[dict[int, CandidateData], dict[int, int], dict[int, int]]:
    """Extrae candidatos a contenedor desde los nodos del grafo.

    ¿POR QUÉ filtrar por grado? Un nodo de grado 1 es un fondo de saco o un
    extremo de calle; preferimos cruces (min_node_degree, normalmente 2+) donde
    cabe físicamente un contenedor y el acceso es mejor.

    Recibe el grafo (lat/lon) y la configuración.
    Devuelve:
        candidates : {idx → CandidateData}
        idx_to_j   : {idx interno → id de nodo OSM}
        j_to_idx   : {id de nodo OSM → idx interno}
    """
    candidates: dict[int, CandidateData] = {}
    idx_to_j: dict[int, int] = {}
    j_to_idx: dict[int, int] = {}

    new_index = 0
    for node, data in graph.nodes.items():
        if data.get("street_count", 0) >= config.min_node_degree:
            candidates[new_index] = CandidateData(
                osm_id=str(node),
                latitude=data["y"],
                longitude=data["x"],
                opening_cost=cost_by_degree(
                    data.get("street_count", 0), config.base_opening_cost
                ),
            )
            idx_to_j[new_index] = node
            j_to_idx[node] = new_index
            new_index += 1

    return candidates, idx_to_j, j_to_idx


def classify_candidate_context(
    config: GeographicConfig,
    candidates: dict[int, CandidateData],
    graph: nx.MultiDiGraph,
    idx_to_j: dict[int, int],
) -> dict[int, CandidateData]:
    """Clasifica el contexto urbano de cada candidato (calle, plaza, parque…).

    ¿POR QUÉ? El contexto se usa luego en consolidate_candidates (los candidatos
    en parques/plazas se descartan; en rotondas se agrupan con más holgura) y es
    información útil para el análisis posterior de las soluciones.

    Recibe configuración, candidatos, grafo y el mapa idx→nodo OSM.
    Devuelve un nuevo dict de candidatos con el campo `context` relleno.
    """
    # ── BLOQUE 1 — Descargar polígonos de parques y plazas ────────────
    parks_gdf = ox.features_from_address(
        config.place, tags={"leisure": "park"}, dist=config.radius
    )
    squares_gdf = ox.features_from_address(
        config.place, tags={"place": "square"}, dist=config.radius
    )
    parks_gdf = parks_gdf[parks_gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    squares_gdf = squares_gdf[squares_gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()

    # unary_union fusiona todos los polígonos en una sola geometría para poder
    # preguntar "¿este punto está dentro?" de una vez.
    park_union = unary_union(parks_gdf.geometry) if len(parks_gdf) > 0 else None
    square_union = unary_union(squares_gdf.geometry) if len(squares_gdf) > 0 else None

    # ── BLOQUE 2 — Detectar nodos de rotonda ──────────────────────────
    roundabout_nodes: set[int] = set()
    for u, v, data in graph.edges(data=True):
        if data.get("junction") == "roundabout":
            roundabout_nodes.add(u)
            roundabout_nodes.add(v)

    # ── BLOQUE 3 — Clasificar cada candidato por prioridad ────────────
    # Orden de prioridad: rotonda > parque > plaza > calle.
    classified: dict[int, CandidateData] = {}
    for idx, candidate in candidates.items():
        point = Point(candidate.longitude, candidate.latitude)
        if idx_to_j[idx] in roundabout_nodes:
            context = CandidateContext.ROUNDABOUT
        elif park_union is not None and park_union.contains(point):
            context = CandidateContext.PARK
        elif square_union is not None and square_union.contains(point):
            context = CandidateContext.SQUARE
        else:
            context = CandidateContext.STREET
        classified[idx] = replace(candidate, context=context)

    return classified


# Umbrales de fusión (metros) según el contexto del candidato. Dos candidatos a
# menos de este umbral se consideran "el mismo sitio" y se consolidan en uno.
_CONSOLIDATION_THRESHOLDS: dict[str, float] = {
    CandidateContext.STREET:        10.0,
    CandidateContext.ROUNDABOUT:    20.0,
    CandidateContext.DENSE_CLUSTER: 15.0,
}
_DEFAULT_CONSOLIDATION_THRESHOLD: float = 15.0

_EXCLUDED_CONTEXTS: frozenset[CandidateContext] = frozenset({
    CandidateContext.PARK,
    CandidateContext.SQUARE,
})


def consolidate_candidates(
    candidates: dict[int, CandidateData],
    graph: nx.MultiDiGraph,
    idx_to_j: dict[int, int],
) -> tuple[dict[int, CandidateData], dict[int, int], dict[int, int]]:
    """Fusiona candidatos espacialmente redundantes (mismo sitio físico).

    ¿POR QUÉ? OSM modela un cruce grande con varios nodos muy próximos. Sin
    consolidar tendríamos contenedores "duplicados" a pocos metros. Agrupamos los
    que están a menos de un umbral (según contexto) y nos quedamos con un
    representante por grupo: el de mayor grado (mejor acceso).

    También se descartan de entrada los candidatos en parque/plaza (no son sitio
    para un contenedor de calle).

    OPTIMIZACIÓN: en lugar de comparar los O(n²) pares posibles, se usa un índice
    espacial (STRtree) que solo devuelve los pares realmente cercanos.

    Recibe candidatos, grafo (para el grado) y el mapa idx→nodo OSM.
    Devuelve (candidatos, idx_to_j, j_to_idx) re-indexados de forma contigua.
    """
    # ── BLOQUE 1 — Descartar contextos no aptos (parque/plaza) ────────
    candidates = {
        idx: c for idx, c in candidates.items()
        if c.context not in _EXCLUDED_CONTEXTS
    }
    if not candidates:
        return {}, {}, {}

    # ── BLOQUE 2 — Grafo de proximidad vía índice espacial ────────────
    # Proyectamos a UTM (metros) para que los umbrales tengan sentido métrico.
    idx_list = list(candidates.keys())
    pts_latlon = gpd.GeoSeries(
        [Point(candidates[i].longitude, candidates[i].latitude) for i in idx_list],
        crs="EPSG:4326",
    )
    pts_utm = pts_latlon.to_crs(pts_latlon.estimate_utm_crs())
    geoms = [Point(p.x, p.y) for p in pts_utm]

    proximity_graph = nx.Graph()
    proximity_graph.add_nodes_from(idx_list)

    # query con el umbral MÁXIMO devuelve todos los pares candidatos a fusionarse;
    # luego cada par se filtra con su umbral concreto (que depende del contexto).
    tree = STRtree(geoms)
    max_threshold = max(_CONSOLIDATION_THRESHOLDS.values())
    pair_array = tree.query(geoms, predicate="dwithin", distance=max_threshold)
    for a, b in zip(pair_array[0], pair_array[1]):
        if a >= b:  # el índice devuelve (a,b) y (b,a) y los auto-pares; nos basta uno
            continue
        i1, i2 = idx_list[a], idx_list[b]
        dist = geoms[a].distance(geoms[b])
        threshold = max(
            _CONSOLIDATION_THRESHOLDS.get(candidates[i1].context, _DEFAULT_CONSOLIDATION_THRESHOLD),
            _CONSOLIDATION_THRESHOLDS.get(candidates[i2].context, _DEFAULT_CONSOLIDATION_THRESHOLD),
        )
        if dist < threshold:
            proximity_graph.add_edge(i1, i2)

    # ── BLOQUE 3 — Un representante por componente conexa ──────────────
    # Cada componente = un "mismo sitio". Representante = mayor grado (desempate
    # por menor idx para que el resultado sea determinista).
    representatives: list[int] = []
    for component in nx.connected_components(proximity_graph):
        representative = max(
            component,
            key=lambda idx: (graph.degree(idx_to_j[idx]), -idx),
        )
        representatives.append(representative)

    # ── BLOQUE 4 — Re-indexar de forma contigua ───────────────────────
    new_candidates: dict[int, CandidateData] = {}
    new_idx_to_j: dict[int, int] = {}
    new_j_to_idx: dict[int, int] = {}
    for new_idx, rep_idx in enumerate(sorted(representatives)):
        osm_node = idx_to_j[rep_idx]
        new_candidates[new_idx] = candidates[rep_idx]
        new_idx_to_j[new_idx] = osm_node
        new_j_to_idx[osm_node] = new_idx

    return new_candidates, new_idx_to_j, new_j_to_idx


# ══════════════════════════════════════════════════════════════════════
#  Edificios: extracción y estimación de población
# ══════════════════════════════════════════════════════════════════════

def extract_buildings(
    config: GeographicConfig,
    ref_surface_m2: float = 30.0,
) -> tuple[dict[int, BuildingData], dict[str, int], dict[int, str], gpd.GeoDataFrame]:
    """Extrae los edificios de OSM y estima su población h_i por superficie.

    ¿POR QUÉ por superficie? No hay censo por edificio; aproximamos la demanda
    con el área construida: h_i = area / ref_surface_m2 (mínimo 1 habitante). El
    área se mide en UTM (m²); el centroide se reproyecta a lat/lon para el mapa.

    Recibe configuración y la superficie de referencia por habitante.
    Devuelve:
        buildings  : {idx → BuildingData}
        idx_to_i   : {idx interno → id OSM}
        i_to_idx   : {id OSM → idx interno}
        buildings_gdf : GeoDataFrame original (para guardar el GeoJSON de mapas).
    """
    # ── BLOQUE 1 — Descargar huellas de edificios ─────────────────────
    buildings_gdf: gpd.GeoDataFrame = ox.features_from_address(
        config.place, tags={"building": True}, dist=config.radius
    )
    buildings_gdf = buildings_gdf[
        buildings_gdf.geometry.type.isin(["Polygon", "MultiPolygon"])
    ].copy()

    # ── BLOQUE 2 — Área (UTM) → h_i, y centroide (lat/lon) ────────────
    # El área debe medirse en metros, así que pasamos a UTM. El centroide se
    # calcula en UTM (evita el aviso de shapely sobre centroides en lat/lon) y se
    # reproyecta a EPSG:4326 para tener su coordenada geográfica.
    buildings_utm = buildings_gdf.to_crs(buildings_gdf.estimate_utm_crs())
    buildings_utm["centroid"] = buildings_utm.geometry.centroid
    buildings_gdf["area_m2"] = buildings_utm.geometry.area
    buildings_gdf["h_i"] = (buildings_gdf["area_m2"] / ref_surface_m2).clip(lower=1.0)
    buildings_gdf["centroid"] = buildings_utm["centroid"].to_crs("EPSG:4326")

    # ── BLOQUE 3 — Empaquetar en BuildingData con índices internos ────
    buildings: dict[int, BuildingData] = {}
    i_to_idx: dict[str, int] = {}
    idx_to_i: dict[int, str] = {}
    for i, (osm_id, row) in enumerate(buildings_gdf.iterrows()):
        buildings[i] = BuildingData(
            osm_id=str(osm_id),
            latitude=row["centroid"].y,
            longitude=row["centroid"].x,
            h_i=row.h_i,
        )
        idx_to_i[i] = str(osm_id)
        i_to_idx[str(osm_id)] = i

    return buildings, idx_to_i, i_to_idx, buildings_gdf
