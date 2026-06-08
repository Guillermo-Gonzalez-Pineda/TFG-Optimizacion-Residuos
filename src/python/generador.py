"""
Instance generator for the waste collection point location problem.

Downloads the OSMnx pedestrian network, extracts buildings and candidate
collection points, computes sparse Dijkstra distances, and serialises
the result to JSON.

References:
    Li et al. (2026). Waste Management 209, 115211.
    Boeing (2025). Geographical Analysis 57, 567-577.
"""

import os
import warnings
import math
from typing import Any
from datetime import datetime
import json

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
    
def compute_distances(
    graph: nx.MultiDiGraph,
    buildings: dict[int, BuildingData],
    candidates: dict[int, CandidateData],
    idx_to_j: dict[int, int],
    cutoff_m: float,
) -> dict[int, dict[int, float]]:
    """Compute sparse Dijkstra distances from candidates to buildings."""
    building_edges: dict[int, tuple[tuple[int, int, int], float]] = {
        i: ox.distance.nearest_edges(graph, b.longitude, b.latitude, return_dist=True)
        for i, b in buildings.items()
    }

    distances: dict[int, dict[int, float]] = {idx: {} for idx in candidates.keys()}
    for c_idx, c_node in idx_to_j.items():
        reachable = nx.single_source_dijkstra_path_length(
            graph,
            c_node,
            cutoff=cutoff_m,
            weight="length",
        )
        for b_idx, ((u, v, _key), d_perp) in building_edges.items():
            d_perp_m = d_perp * 111_320 * math.cos(math.radians(28.48))
            dist_via_u = reachable.get(u, float("inf")) + d_perp_m
            dist_via_v = reachable.get(v, float("inf")) + d_perp_m
            best = min(dist_via_u, dist_via_v)
            if best < cutoff_m:
                distances[c_idx][b_idx] = best
    return distances


def evaluate_coverage(
    buildings: dict[int, BuildingData],
    dij: dict[int, dict[int, float]],
    params: ModelParameters,
) -> dict[int, list[int]]:
    """Return {building_idx: [waste_types without coverage]}.
    
    Empty dict means full coverage — no midpoint candidates needed.
    """
    reachable_from_building: dict[int, dict[int, float]] = {
        i: {} for i in buildings
    }

    for j_idx, building_distances in dij.items():
        for i_idx, dist in building_distances.items():
            reachable_from_building[i_idx][j_idx] = dist

    uncovered: dict[int, list[int]] = {}
    for i_idx in buildings:
        missing_types: list[int] = []
        for k, r_k in params.coverage_radius.items():
            covered = any(
                dist <= r_k
                for dist in reachable_from_building[i_idx].values()
            )
            if not covered:
                missing_types.append(k)
        if missing_types:
            uncovered[i_idx] = missing_types

    return uncovered


def add_midpoint_candidates(
    graph: nx.MultiDiGraph,
    candidates: dict[int, CandidateData],
    idx_to_j: dict[int, int],
    j_to_idx: dict[int, int],
    min_segment_length_m: float = 75.0,
) -> tuple[dict[int, CandidateData], dict[int, int], dict[int, int]]:
    """Add midpoint candidates on long segments without nearby candidates."""

    candidate_osm_nodes: set[int] = set(idx_to_j.values())

    new_candidates = dict(candidates)
    new_idx_to_j = dict(idx_to_j)
    new_j_to_idx = dict(j_to_idx)
    next_idx = max(candidates.keys()) + 1
    next_virtual_id = -1  # IDs negativos para nodos virtuales

    for u, v, data in graph.edges(data=True):
        length = data.get("length", 0.0)
        if length <= min_segment_length_m:
            continue
        if u in candidate_osm_nodes or v in candidate_osm_nodes:
            continue

        # Compute midpoint
        geom = data.get("geometry")
        if geom:
            midpoint = geom.interpolate(0.5, normalized=True)
            lat, lon = midpoint.y, midpoint.x
        else:
            lat = (graph.nodes[u]["y"] + graph.nodes[v]["y"]) / 2
            lon = (graph.nodes[u]["x"] + graph.nodes[v]["x"]) / 2

        osm_id = f"midpoint_{u}_{v}"
        new_candidates[next_idx] = CandidateData(
            osm_id=osm_id,
            latitude=lat,
            longitude=lon,
            context=CandidateContext.STREET,
        )
        new_idx_to_j[next_idx] = next_virtual_id
        new_j_to_idx[next_virtual_id] = next_idx
        next_idx += 1
        next_virtual_id -= 1

    return new_candidates, new_idx_to_j, new_j_to_idx


def build_instance(
    config: GeographicConfig,
    params: ModelParameters,
    buildings: dict[int, BuildingData],
    candidates: dict[int, CandidateData],
    i_to_idx: dict[str, int],
    idx_to_i: dict[int, str],
    j_to_idx: dict[int, int],
    idx_to_j: dict[int, int],
    dij: dict[int, dict[int, float]],
) -> Instance:
    """Assemble all components into a complete Instance object."""
    return Instance(
        # --- Metadata ---
        study_case=config.place,
        osm_radius_m=config.radius,
        dijkstra_radius_m=config.cutoff_dijkstra,
        generated_at=datetime.now().isoformat(),
        references=tuple([
            "Li et al. (2026). Waste Management 209, 115211.",
            "Boeing (2025). Geographical Analysis 57, 567-577.",
        ]),
        n_buildings=len(buildings),
        n_candidates=len(candidates),
        n_waste_types=len(params.coverage_radius),
        total_population=sum(b.h_i for b in buildings.values()),
        n_dijkstra_connections=sum(len(v) for v in dij.values()),

        # --- Index translation maps ---
        i_to_idx=i_to_idx,
        idx_to_i=idx_to_i,
        j_to_idx=j_to_idx,
        idx_to_j=idx_to_j,

        # --- Sets ---
        K=list(params.coverage_radius.keys()),
        I=buildings,
        J=candidates,

        # --- Distances ---
        dij=dij,

        # --- Parameters ---
        params=params,
    )

def save_instance(instance: Instance, path: str) -> None:
    """Serialise an Instance to JSON following the project schema."""

    data: dict[str, Any] = {
        "meta": {
            "study_case": instance.study_case,
            "osm_radius_m": instance.osm_radius_m,
            "dijkstra_radius_m": instance.dijkstra_radius_m,
            "generated_at": instance.generated_at,
            "references": list(instance.references),
            "n_buildings": instance.n_buildings,
            "n_candidates": instance.n_candidates,
            "n_waste_types": instance.n_waste_types,
            "total_population": instance.total_population,
            "n_dijkstra_connections": instance.n_dijkstra_connections,
        },
        "parameters": {
            "opening_cost": instance.params.opening_cost,
            "max_bins": instance.params.max_bins,
            "nimby_distance": instance.params.nimby_distance,
            "waste_per_capita": instance.params.waste_per_capita,
            "overflow_penalty": instance.params.overflow_penalty,
            "bin_cost": {str(k): v for k, v in instance.params.bin_cost.items()},
            "bin_capacity": {str(k): v for k, v in instance.params.bin_capacity.items()},
            "coverage_radius": {str(k): v for k, v in instance.params.coverage_radius.items()},
            "waste_proportion": {str(k): v for k, v in instance.params.waste_proportion.items()},
            "collection_frequency": {str(k): v for k, v in instance.params.collection_frequency.items()},
            "lognormal_mu": instance.params.lognormal_mu,
            "lognormal_sigma": instance.params.lognormal_sigma,
            "overflow_threshold": instance.params.overflow_threshold,
        },
        "sets": {
            "K": instance.K,
            "I": {
                str(idx): {
                    "osm_id": b.osm_id,
                    "latitude": b.latitude,
                    "longitude": b.longitude,
                    "h_i": b.h_i,
                }
                for idx, b in instance.I.items()
            },
            "J": {
                str(idx): {
                    "osm_id": c.osm_id,
                    "latitude": c.latitude,
                    "longitude": c.longitude,
                    "context": c.context.value,
                }
                for idx, c in instance.J.items()
            },
        },
        "distances": {
            str(j_idx): {
                str(i_idx): dist
                for i_idx, dist in building_dists.items()
            }
            for j_idx, building_dists in instance.dij.items()
        },
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    from instancia import ModelParameters

    config = GeographicConfig(
        place="Plaza del Cristo, San Cristóbal de La Laguna, España",
        radius=500,
        network_type="walk",
        cutoff_dijkstra=350,
        min_node_degree=3,
    )

    params = ModelParameters(
        opening_cost=4000.0,
        max_bins=8,
        nimby_distance=0.0,
        waste_per_capita=1.32,
        overflow_penalty=500.0,
        bin_cost={0: 350.0, 1: 300.0, 2: 250.0, 3: 500.0},
        bin_capacity={0: 120.0, 1: 120.0, 2: 120.0, 3: 120.0},
        coverage_radius={0: 150.0, 1: 150.0, 2: 250.0, 3: 350.0},
        waste_proportion={0: 0.5012, 1: 0.0791, 2: 0.3885, 3: 0.0312},
        collection_frequency={0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0},
        lognormal_mu=0.0,
        lognormal_sigma=0.25,
        overflow_threshold=0.05,
    )

    print("Step 1/7 — Downloading graph...")
    graph = download_graph(config)
    print(f"          Nodes: {graph.number_of_nodes()}  Edges: {graph.number_of_edges()}")

    print("Step 2/7 — Extracting candidates...")
    candidates, idx_to_j, j_to_idx = extract_candidates(graph, config)
    print(f"          Candidates: {len(candidates)}")

    print("Step 3/7 — Extracting buildings...")
    buildings, i_to_idx, idx_to_i = extract_buildings(config)
    print(f"          Buildings: {len(buildings)}")

    print("Step 4/7 — Classifying candidate context...")
    candidates = classify_candidate_context(config, candidates, graph, idx_to_j)

    print("Step 5/7 — Consolidating candidates...")
    candidates, idx_to_j, j_to_idx = consolidate_candidates(candidates, graph, idx_to_j)
    print(f"          Candidates after consolidation: {len(candidates)}")

    print("Step 6/7 — Computing distances...")
    dij = compute_distances(graph, buildings, candidates, idx_to_j, config.cutoff_dijkstra)
    print(f"          Connections: {sum(len(v) for v in dij.values())}")

    print("Step 7/7 — Building and saving instance...")
    uncovered = evaluate_coverage(buildings, dij, params)
    if uncovered:
        print(f"          ⚠ {len(uncovered)} buildings without full coverage")
    else:
        print(f"          ✓ Full coverage")

    instance = build_instance(
        config, params, buildings, candidates,
        i_to_idx, idx_to_i, j_to_idx, idx_to_j, dij,
    )

    output_path = "data/processed/instancia_laguna.json"
    save_instance(instance, output_path)
    print(f"\n✓ Instance saved to {output_path}")
    print(f"  Buildings:   {instance.n_buildings}")
    print(f"  Candidates:  {instance.n_candidates}")
    print(f"  Connections: {instance.n_dijkstra_connections}")