"""
Instance generator for the waste collection point location problem.

Downloads the OSMnx pedestrian network, extracts buildings and candidate
collection points, computes sparse Dijkstra distances, and serialises
the result to JSON.

References:
    Li et al. (2026). Waste Management 209, 115211.
    Boeing (2025). Geographical Analysis 57, 567-577.
"""

import warnings
import math
from typing import Any

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd
from shapely.geometry import Point
from shapely.ops import unary_union
from itertools import combinations
from dataclasses import replace

from instancia import (
    BuildingData,
    CandidateData,
    GeographicConfig,
    Instance,
    ModelParameters,
    CandidateContext,
)

ox.settings.use_cache = True
ox.settings.log_console = False

_NON_WALKABLE_HIGHWAY_TAGS: frozenset[str] = frozenset({
    "motorway", "motorway_link",
    "trunk", "trunk_link",
    "track",
})

_NON_WALKABLE_SERVICE_SUBTYPES: frozenset[str] = frozenset({
    "driveway",
    "parking_aisle",
})

def _is_non_walkable(data: dict[str, Any]) -> bool:
    """Return True if the edge should be excluded from the pedestrian network."""
    tag = _highway_tag(data)
    if tag in _NON_WALKABLE_HIGHWAY_TAGS:
        return True
    if tag == "service":
        subtype = data.get("service", "")
        if isinstance(subtype, list):
            subtype = subtype[0]
        access = data.get("access", "")
        if isinstance(access, list):
            access = access[0]
    return False

def _highway_tag(data: dict[str, Any]) -> str:
    """Normalise the OSMnx highway attribute to a single string."""
    tag = data.get("highway", "")
    return tag[0] if isinstance(tag, list) else tag


def download_graph(config: GeographicConfig) -> nx.MultiDiGraph:
    """Download and filter the pedestrian street network via OSMnx."""
    graph: nx.MultiDiGraph = ox.graph_from_address(
        config.place,
        dist=config.radius,
        network_type=config.network_type,
    )

    edges_to_remove = [
        (u, v, k)
        for u, v, k, data in graph.edges(keys=True, data=True)
        if _is_non_walkable(data)
    ]
    graph.remove_edges_from(edges_to_remove)

    return graph

    


def extract_candidates(
    graph: nx.MultiDiGraph,
    config: GeographicConfig,
) -> tuple[dict[int, CandidateData], dict[int, int], dict[int, int]]:

    """Extract candidate collection points from the graph nodes."""

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
    """Classify the urban context of each candidate collection point."""
    
    parks_gdf = ox.features_from_address(
        config.place,
        tags={"leisure": "park"},
        dist=config.radius
    )
    squares_gdf = ox.features_from_address(
        config.place,
        tags={"place": "square"},
        dist=config.radius
    )

    parks_gdf = parks_gdf[
        parks_gdf.geometry.type.isin(["Polygon", "MultiPolygon"])
    ].copy()

    squares_gdf = squares_gdf[
        squares_gdf.geometry.type.isin(["Polygon", "MultiPolygon"])
    ].copy()

    park_union = unary_union(parks_gdf.geometry) if len(parks_gdf) > 0 else None
    square_union = unary_union(squares_gdf.geometry) if len(squares_gdf) > 0 else None

    roundabout_nodes: set[int] = set()
    for u, v, data in graph.edges(data=True):
        if data.get("junction") == "roundabout":
            roundabout_nodes.add(u)
            roundabout_nodes.add(v)

    classified: dict[int, CandidateData] = {}
    for idx, candidate in candidates.items():
        point = Point(candidate.longitude, candidate.latitude)
        if idx_to_j[idx] in roundabout_nodes:
            context = CandidateContext.ROUNDABOUT
        elif park_union and park_union.contains(point):
            context = CandidateContext.PARK
        elif square_union and square_union.contains(point):
            context = CandidateContext.SQUARE
        else:
            context = CandidateContext.STREET
        classified[idx] = replace(candidate, context=context)

    return classified
        
_CONSOLIDATION_THRESHOLDS: dict[str, float] = {
    CandidateContext.STREET:        10.0,
    CandidateContext.ROUNDABOUT:    20.0,
    CandidateContext.DENSE_CLUSTER: 15.0,
}

def _euclidean_distance_m(
    candidate1: CandidateData,
    candidate2: CandidateData,
) -> float:
    """Calculate the Euclidean distance in meters between two candidates."""
    latm = (candidate1.latitude - candidate2.latitude) * 111_320
    lonm = (candidate1.longitude - candidate2.longitude) * 111_320 * math.cos(math.radians(candidate1.latitude))
    return math.sqrt(latm**2 + lonm**2)


def consolidate_candidates(
    candidates: dict[int, CandidateData],
    graph: nx.MultiDiGraph,
    idx_to_j: dict[int, int],
) -> tuple[dict[int, CandidateData], dict[int, int], dict[int, int]]:
    """Consolidate spatially redundant candidates using connected components."""

    _EXCLUDED_CONTEXTS: frozenset[CandidateContext] = frozenset({
        CandidateContext.PARK,
        CandidateContext.SQUARE,
    })

    candidates = {
        idx: c for idx, c in candidates.items()
        if c.context not in _EXCLUDED_CONTEXTS
    }

    # --- Paso 1: Construir grafo de proximidad ---
    proximity_graph = nx.Graph()
    proximity_graph.add_nodes_from(candidates.keys())

    for idx1, idx2 in combinations(candidates.keys(), 2):
        dist = _euclidean_distance_m(candidates[idx1], candidates[idx2])
        threshold = max(
            _CONSOLIDATION_THRESHOLDS.get(candidates[idx1].context, 15.0),
            _CONSOLIDATION_THRESHOLDS.get(candidates[idx2].context, 15.0),
        )
        if dist < threshold:
            proximity_graph.add_edge(idx1, idx2)

    # --- Paso 2: Componentes conexas ---
    representatives: list[int] = []
    for component in nx.connected_components(proximity_graph):
        representative = max(
            component,
            key=lambda idx: (graph.degree(idx_to_j[idx]), -idx),
        )
        representatives.append(representative)

    # --- Paso 3: Reasignar índices contiguos ---
    new_candidates: dict[int, CandidateData] = {}
    new_idx_to_j: dict[int, int] = {}
    new_j_to_idx: dict[int, int] = {}

    for new_idx, rep_idx in enumerate(sorted(representatives)):
        osm_node = idx_to_j[rep_idx]
        new_candidates[new_idx] = candidates[rep_idx]
        new_idx_to_j[new_idx] = osm_node
        new_j_to_idx[osm_node] = new_idx

    return new_candidates, new_idx_to_j, new_j_to_idx



def extract_buildings(
    config: GeographicConfig,
    ref_surface_m2: float = 30.0,
) -> tuple[dict[int, BuildingData], dict[str, int], dict[int, str]]:
    """Extract building data from OSM within the specified area."""

    buildings_gdf: gpd.GeoDataFrame = ox.features_from_address(
        config.place,
        tags={"building": True},
        dist=config.radius
    )
    buildings_gdf = buildings_gdf[buildings_gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()

    buildings_utm = buildings_gdf.to_crs(buildings_gdf.estimate_utm_crs())
    buildings_utm["centroid"] = buildings_utm.geometry.centroid  # ← mueve esta línea aquí
    buildings_gdf["area_m2"] = buildings_utm.geometry.area
    buildings_gdf["h_i"] = (buildings_gdf["area_m2"] / ref_surface_m2).clip(lower=1.0)
    buildings_gdf["centroid"] = buildings_utm["centroid"].to_crs("EPSG:4326")  # ← reproyecta de vuelta a WGS84

    buildings: dict[int, BuildingData] = {}
    i_to_idx: dict[str, int] = {}
    idx_to_i : dict[int, str] = {}
    for i, (osm_id, row) in enumerate(buildings_gdf.iterrows()):
        buildings[i] = BuildingData(
            osm_id=str(osm_id),
            latitude=row["centroid"].y,
            longitude=row["centroid"].x,
            h_i=row.h_i,
        )
        idx_to_i[i] = str(osm_id)
        i_to_idx[str(osm_id)] = i
    
    return buildings, idx_to_i, i_to_idx
    