"""
Cómputos que NECESITAN la pila geoespacial pero devuelven NÚMEROS, no dibujos.

Cuarto cuadrante de la matriz de análisis:

    metricas   = número + geo-free   |   comparativas = visual + geo-free
    geo (este) = número + geo        |   mapas         = visual + geo

Por eso este módulo SÍ puede importar la pila pesada (shapely/pyproj vía
geopandas, networkx, osmnx), a diferencia de ``metricas`` (geo-free). PROHIBIDO
importar gurobipy. Como ``mapas``, NO se re-exporta en el ``__init__`` del
paquete: ``import analisis`` no debe arrastrar la pila geo; quien la necesite
hace ``from analisis.geo import ...`` y solo entonces la paga.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import geopandas as gpd
import networkx as nx
import osmnx as ox

from . import rutas

if TYPE_CHECKING:                       # solo type-hints: no acopla el import
    from instancia import Instance


def densidad_convexhull(inst: "Instance") -> dict[str, float]:
    """Área del convex-hull de los edificios (km²) y densidad de población.

    Toma las coordenadas lon/lat de ``inst.I``, construye su envolvente convexa y
    la REPROYECTA a UTM (``estimate_utm_crs``) ANTES de medir el área. Medir áreas
    en grados (EPSG:4326) es incorrecto: un grado no es una distancia métrica fija
    (y menos en longitud, que se acorta con la latitud), así que ``hull.area`` en
    grados no son km². En UTM las unidades son metros y el área es real.

    Es el caso OPUESTO a ``mapas.emparejar_demanda``, donde NO se reproyecta
    porque allí solo se compara CERCANÍA entre puntos del mismo CRS; aquí SÍ se
    reproyecta porque se mide SUPERFICIE.

    ``densidad = inst.total_population / area_km2``."""
    pts = gpd.GeoSeries(
        gpd.points_from_xy(
            [inst.I[i].longitude for i in inst.I],
            [inst.I[i].latitude for i in inst.I],
        ),
        crs="EPSG:4326",
    )
    pts_utm = pts.to_crs(pts.estimate_utm_crs())     # CRS métrico local (UTM)
    hull = pts_utm.union_all().convex_hull           # API vigente (no unary_union)
    area_km2 = hull.area / 1e6
    densidad = inst.total_population / area_km2 if area_km2 > 0 else float("nan")
    return {"area_km2": float(area_km2), "densidad_hab_km2": float(densidad)}


def validar_grafo(inst_o_tam) -> dict[str, Any]:
    """Salud topológica de la red de calles de un tamaño.

    Acepta una ``Instance`` (usa su ``osm_radius_m``) o un ``tam`` entero. Carga
    el graphml vía ``rutas.ruta_grafo`` y reporta:
      - nº de nodos y aristas;
      - componentes conexas y si es conexo (sobre el grafo NO dirigido: caminar no
        tiene sentido de dirección; un grafo fragmentado daría distancias ∞);
      - nº de nodos artificiales (id que empieza por ``"-"``, mismo marcador que
        los candidatos artificiales). NOTA: los graphml de este repo guardan solo
        ids OSM positivos (los candidatos artificiales viven en la instancia, no
        en el grafo), así que hoy esta cuenta es 0; se mantiene el marcador por
        fidelidad y compatibilidad hacia delante;
      - nº de aristas anormalmente largas (>500 m), señal de descarga sospechosa;
      - nº de nodos aislados (componentes de tamaño 1)."""
    tam = inst_o_tam.osm_radius_m if hasattr(inst_o_tam, "osm_radius_m") else int(inst_o_tam)
    G = ox.load_graphml(rutas.ruta_grafo(tam))

    G_u = ox.convert.to_undirected(G)
    componentes = list(nx.connected_components(G_u))
    n_componentes = len(componentes)
    n_aislados = sum(1 for c in componentes if len(c) == 1)

    n_artificiales = sum(1 for n in G.nodes if str(n).startswith("-"))

    aristas = ox.graph_to_gdfs(G, nodes=False)       # longitudes vía osmnx
    n_largas = int((aristas["length"] > 500).sum())

    return {
        "n_nodos": G.number_of_nodes(),
        "n_aristas": G.number_of_edges(),
        "n_componentes_conexas": n_componentes,
        "es_conexo": n_componentes == 1,
        "n_nodos_artificiales": n_artificiales,
        "n_aristas_largas": n_largas,
        "n_nodos_aislados": n_aislados,
    }
