"""
Construcción y manipulación del grafo peatonal (módulo 1/4 del generador).

¿POR QUÉ EXISTE? Todo el pipeline de generación de instancias parte de un grafo
de calles descargado de OpenStreetMap vía OSMnx. Este módulo concentra TODO lo
relativo al grafo en sí:

    1. download_graph      — descarga la red peatonal y filtra las vías no
                             caminables (autopistas, pistas, accesos privados…).
    2. project_to_utm      — proyecta el grafo a coordenadas UTM (metros), que es
                             el sistema en el que se miden distancias.
    3. insert_point_on_edge— "materializa" un punto en mitad de una calle como un
                             nodo nuevo del grafo, partiendo la arista que lo
                             contiene. Lo usa el módulo distancias para colocar
                             contenedores artificiales justo enfrente de un
                             edificio.

CONVENCIÓN DE DOS GRAFOS. El pipeline maneja dos copias del mismo grafo:
    - graph     : en lat/lon (EPSG:4326). Sirve para coordenadas y mapas.
    - graph_utm : en metros (UTM). Sirve SOLO para medir distancias (Dijkstra,
                  proyecciones, longitudes de arista).
insert_point_on_edge opera sobre graph_utm porque necesita longitudes en metros.

Referencias:
    Boeing (2025). Geographical Analysis 57, 567-577.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import osmnx as ox
from shapely.geometry import LineString
from shapely.ops import substring

from instancia import GeographicConfig

# Cache de OSMnx activado: descargar el mismo lugar dos veces no vuelve a la red.
ox.settings.use_cache = True
ox.settings.log_console = False


# ══════════════════════════════════════════════════════════════════════
#  Filtrado de vías no caminables
# ══════════════════════════════════════════════════════════════════════
# Aunque pedimos a OSMnx la red "walk", arrastra vías por las que un vecino no
# iría a tirar la basura (autopistas, enlaces, pistas de tierra) o accesos que no
# son calle pública (caminos de garaje, pasillos de aparcamiento). Las quitamos
# para que las distancias peatonales sean realistas.

_NON_WALKABLE_HIGHWAY_TAGS: frozenset[str] = frozenset({
    "motorway", "motorway_link",
    "trunk", "trunk_link",
    "track",
})

_NON_WALKABLE_SERVICE_SUBTYPES: frozenset[str] = frozenset({
    "driveway",
    "parking_aisle",
})


def _highway_tag(data: dict[str, Any]) -> str:
    """Normaliza el atributo 'highway' de OSMnx a un único string.

    OSMnx a veces guarda 'highway' como lista (cuando una arista fusiona varias
    vías OSM con etiquetas distintas). Nos quedamos con la primera para poder
    compararla contra los conjuntos de exclusión.
    """
    tag = data.get("highway", "")
    return tag[0] if isinstance(tag, list) else tag


def _is_non_walkable(data: dict[str, Any]) -> bool:
    """Devuelve True si la arista debe excluirse de la red peatonal.

    Recibe el diccionario de atributos de una arista de OSMnx.
    Devuelve True (excluir) / False (conservar).
    """
    tag = _highway_tag(data)

    # 1) Tipos de vía nunca caminables (autopistas, pistas…).
    if tag in _NON_WALKABLE_HIGHWAY_TAGS:
        return True

    # 2) Vías "service": solo se excluyen los subtipos que no son calle pública
    #    (caminos de garaje, pasillos de parking). El resto de 'service' (p. ej.
    #    callejones de acceso) se conservan.
    if tag == "service":
        subtype = data.get("service", "")
        if isinstance(subtype, list):
            subtype = subtype[0]
        if subtype in _NON_WALKABLE_SERVICE_SUBTYPES:
            return True

    return False


def download_graph(config: GeographicConfig) -> nx.MultiDiGraph:
    """Descarga la red peatonal de OSMnx y elimina las vías no caminables.

    ¿POR QUÉ el margen? Pedimos un radio mayor (radius + graph_margin) que el de
    estudio para que las calles del borde no queden "cortadas": un edificio justo
    en el límite necesita poder proyectarse sobre una calle que continúe fuera.

    Recibe:
        config : configuración geográfica (lugar, radio, margen, tipo de red).
    Devuelve:
        Grafo dirigido (MultiDiGraph) en lat/lon, ya filtrado.
    """
    # ── BLOQUE 1 — Descargar la red dentro del radio + margen ──────────
    graph: nx.MultiDiGraph = ox.graph_from_address(
        config.place,
        dist=config.radius + config.graph_margin,
        network_type=config.network_type,
    )

    # ── BLOQUE 2 — Eliminar aristas no caminables ─────────────────────
    # Recolectamos primero y borramos después para no mutar el grafo mientras se
    # itera sobre sus aristas.
    edges_to_remove = [
        (u, v, k)
        for u, v, k, data in graph.edges(keys=True, data=True)
        if _is_non_walkable(data)
    ]
    graph.remove_edges_from(edges_to_remove)

    # ── Eliminar nodos aislados (grado 0) ─────────────────
    # Tras filtrar calles no caminables, algunos nodos quedan sin ninguna
    # arista (puntos fantasma de OSM sin acceso viario). No son candidatos
    # válidos ni los usa Dijkstra, así que los retiramos para que el grafo
    # quede conexo y limpio.
    nodos_aislados = [n for n in graph.nodes if graph.degree(n) == 0]
    graph.remove_nodes_from(nodos_aislados)

    return graph


def project_to_utm(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Proyecta el grafo lat/lon a UTM (metros).

    ¿POR QUÉ? Dijkstra, las proyecciones de edificios y las longitudes de arista
    solo tienen sentido en metros. El grafo lat/lon se conserva aparte para
    coordenadas y mapas (ver convención de dos grafos en la cabecera del módulo).

    Recibe un grafo en EPSG:4326; devuelve una copia nueva en la zona UTM que
    OSMnx estima automáticamente para el lugar.
    """
    return ox.project_graph(graph)


def insert_point_on_edge(
    graph_utm: nx.MultiDiGraph,
    u: int,
    v: int,
    key: int,
    along_u: float,
    point_id: int,
) -> int:
    """Inserta un nodo P sobre la arista (u, v) partiéndola en u→P y P→v.

    ¿POR QUÉ EXISTE? Un contenedor colocado SOBRE una calle (no en un cruce) no
    coincide con ningún nodo del grafo. Dijkstra solo conoce distancias entre
    nodos existentes, así que para que el grafo "sepa" que hay un punto alcanzable
    en mitad de la calle hay que materializarlo como un nodo nuevo P y reconectar
    la arista que lo contiene como u→P→v. Así los edificios que proyectan sobre
    esa calle tendrán a P como EXTREMO de su sub-arista y su distancia al
    contenedor saldrá exacta (reachable[P] = 0 en compute_distances), no inflada
    por tener que rodear hasta u o v.

    Recibe:
        graph_utm : grafo proyectado a UTM. Se modifica IN SITU.
        u, v, key : identifican la arista dirigida que se va a partir.
        along_u   : metros desde u, a lo largo de la geometría, donde cae P.
        point_id  : id (entero, único) que tendrá el nuevo nodo P.
    Devuelve:
        point_id (el id del nodo recién insertado).
    """
    # ══════════════════════════════════════════════════════════════════
    #  BLOQUE 1 — Recuperar la geometría de la arista
    # ══════════════════════════════════════════════════════════════════
    # La arista puede tener una geometría curva explícita ("geometry") o ser una
    # recta implícita entre u y v. En el segundo caso la construimos con las
    # coordenadas UTM de ambos nodos.
    edge_data = graph_utm[u][v][key]
    geom = _edge_geometry(graph_utm, u, v, edge_data)
    length = geom.length

    # Mantener P estrictamente DENTRO de la arista evita sub-tramos de longitud
    # cero (que romperían Dijkstra) y que P coincida con un nodo ya existente.
    along_u = min(max(along_u, 1e-6), length - 1e-6)

    # ══════════════════════════════════════════════════════════════════
    #  BLOQUE 2 — Partir la geometría en dos sub-tramos
    # ══════════════════════════════════════════════════════════════════
    # substring corta la polilínea respetando su forma real: seg_u va de u a P y
    # seg_v de P a v. Así cada sub-arista conserva la LONGITUD verdadera de la
    # calle (no la distancia en línea recta entre extremos). p_point es P.
    seg_u = substring(geom, 0.0, along_u)        # u → P
    seg_v = substring(geom, along_u, length)     # P → v
    p_point = geom.interpolate(along_u)          # coordenadas UTM de P

    # ══════════════════════════════════════════════════════════════════
    #  BLOQUE 3 — Crear el nodo P y reconstruir el sentido directo
    # ══════════════════════════════════════════════════════════════════
    # street_count=1 marca a P como "extremo de calle" (artificial), coherente
    # con el coste de apertura de grado 1 que le asigna ensure_coverage.
    graph_utm.add_node(point_id, x=p_point.x, y=p_point.y, street_count=1)

    # Copiamos los atributos originales (highway, osmid…) salvo los que
    # recalculamos —length y geometry— para no heredar valores obsoletos.
    fwd_attrs = _attrs_without_geom(edge_data)
    # NOTA: no fijamos key=0. Dejamos que NetworkX asigne la siguiente key libre;
    # así, si por lo que fuera ya existiera una arista entre estos nodos, NO la
    # sobrescribimos (que era una de las sospechas del bug de conectividad).
    graph_utm.add_edge(u, point_id, length=seg_u.length, geometry=seg_u, **fwd_attrs)
    graph_utm.add_edge(point_id, v, length=seg_v.length, geometry=seg_v, **fwd_attrs)
    graph_utm.remove_edge(u, v, key)

    # ══════════════════════════════════════════════════════════════════
    #  BLOQUE 4 — Reconstruir el sentido inverso (si la calle es bidireccional)
    # ══════════════════════════════════════════════════════════════════
    # Si existe la arista inversa (v, u) hay que partirla TAMBIÉN. Si la dejáramos
    # intacta, nearest_edges podría devolver esa recta sin cortar para algún
    # edificio; entonces P no sería extremo de su sub-arista y la distancia al
    # contenedor saldría inflada (el edificio se daría por no cubierto).
    #
    # CLAVE (corrige el bug de longitudes cruzadas): NO proyectamos P sobre la
    # geometría inversa ni asumimos su orientación. Construimos los sub-tramos
    # inversos como el REVERSO EXACTO de los directos. Así v→P mide siempre lo
    # mismo que P→v y P→u lo mismo que u→P, sea cual sea cómo OSMnx haya
    # almacenado la geometría del sentido inverso.
    if graph_utm.has_edge(v, u):
        rev_key = next(iter(graph_utm[v][u]))
        rev_attrs = _attrs_without_geom(graph_utm[v][u][rev_key])
        seg_v_rev = LineString(seg_v.coords[::-1])   # v → P (reverso de P→v)
        seg_u_rev = LineString(seg_u.coords[::-1])   # P → u (reverso de u→P)
        graph_utm.add_edge(v, point_id, length=seg_v.length, geometry=seg_v_rev, **rev_attrs)
        graph_utm.add_edge(point_id, u, length=seg_u.length, geometry=seg_u_rev, **rev_attrs)
        graph_utm.remove_edge(v, u, rev_key)

    return point_id


# ── Helpers internos compartidos ──────────────────────────────────────

def _edge_geometry(
    graph_utm: nx.MultiDiGraph,
    u: int,
    v: int,
    edge_data: dict[str, Any],
) -> LineString:
    """Geometría de la arista (u,v): la explícita si existe, o la recta u→v."""
    if "geometry" in edge_data:
        return edge_data["geometry"]
    pu = (graph_utm.nodes[u]["x"], graph_utm.nodes[u]["y"])
    pv = (graph_utm.nodes[v]["x"], graph_utm.nodes[v]["y"])
    return LineString([pu, pv])


def _attrs_without_geom(edge_data: dict[str, Any]) -> dict[str, Any]:
    """Copia los atributos de una arista salvo 'geometry' y 'length'.

    Esos dos los recalculamos para cada sub-tramo, así que copiarlos heredaría
    valores obsoletos del tramo original.
    """
    return {k: val for k, val in edge_data.items() if k not in ("geometry", "length")}
