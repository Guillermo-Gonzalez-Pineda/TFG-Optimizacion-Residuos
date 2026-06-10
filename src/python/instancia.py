"""
Instance generation for the waste collection point location problem.

Builds the HDM instance from OSMnx data (geographic graph + building
footprints) and serialises it to JSON. Designed to extend to HVM.

References:
    Li et al. (2026). Waste Management 209, 115211.
    Boeing (2025). Geographical Analysis 57, 567-577.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from enum import Enum

class CandidateContext(str, Enum):
    """Urban context of a candidate collection point."""
    STREET        = "street"
    PARK          = "park"
    SQUARE        = "square"
    ROUNDABOUT    = "roundabout"
    DENSE_CLUSTER = "dense_cluster"


@dataclass(frozen=True)
class GeographicConfig:
    """Geographic configuration for the problem instance."""

    place: str
    radius: int
    network_type: str
    cutoff_dijkstra: int
    min_node_degree: int
    graph_margin: int


@dataclass(frozen=True)
class ModelParameters:
    """HDM/HVM parameters for the waste collection location model."""

    # --- Scalars ---
    opening_cost: float
    max_bins: int
    nimby_distance: float
    waste_per_capita: float
    overflow_penalty: float

    # --- Indexed by waste type k={0,1,2,3} ---
    # k=0: organic  k=1: general  k=2: recyclable  k=3: hazardous
    bin_cost: dict[int, float]
    bin_capacity: dict[int, float]
    coverage_radius: dict[int, float]
    waste_proportion: dict[int, float]
    collection_frequency: dict[int, float]  # Reserved for HVM

    # --- Demand variability parameters (HVM) ---
    lognormal_mu: float
    lognormal_sigma: float
    overflow_threshold: float

@dataclass(frozen=True)
class BuildingData:
    """Data for a building in the problem instance."""

    osm_id: str
    latitude: float
    longitude: float
    h_i: int


@dataclass(frozen=True)
class CandidateData:
    """Data for a candidate location in the problem instance."""

    osm_id: str
    latitude: float
    longitude: float
    context: CandidateContext = CandidateContext.STREET


@dataclass(frozen=True)
class Instance:
    """Fully built problem instance, ready for optimisation."""
    # --- Metadata ---
    study_case: str
    osm_radius_m: int
    dijkstra_radius_m: int
    generated_at: str          # ISO 8601
    references: list[str]
    n_buildings: int
    n_candidates: int
    n_waste_types: int
    total_population: int
    n_dijkstra_connections: int

    # --- Index translation maps ---
    i_to_idx: dict[str, int]   # OSM id → internal index
    idx_to_i: dict[int, str]   # internal index → OSM id
    j_to_idx: dict[int, int]
    idx_to_j: dict[int, int]

    # --- Sets (with internal indices) ---
    K: list[int]               # [0, 1, 2, 3]
    I: dict[int, BuildingData] # idx → building data
    J: dict[int, CandidateData] # idx → candidate data

    # --- Distances (sparse, internal indices) ---
    dij: dict[int, dict[int, float]]  # dij[j_idx][i_idx] = distance in meters

    # --- Model parameters ---
    params: ModelParameters


def compute_demand(
    h_i: float,
    params: ModelParameters,
    k: int,
) -> float:
    """Deterministic daily demand for waste type k at building i (HDM)."""
    return h_i * params.waste_per_capita * params.waste_proportion[k]



def load_instance(path: str) -> Instance:
    """Deserialise an Instance from a JSON file."""

    with open(path, 'r') as f:
        data = json.load(f)

    p = data["parameters"]
    params = ModelParameters(
        opening_cost=p["opening_cost"],
        max_bins=p["max_bins"],
        nimby_distance=p["nimby_distance"],
        waste_per_capita=p["waste_per_capita"],
        overflow_penalty=p["overflow_penalty"],
        bin_cost={int(k): v for k, v in p["bin_cost"].items()},
        bin_capacity={int(k): v for k, v in p["bin_capacity"].items()},
        coverage_radius={int(k): v for k, v in p["coverage_radius"].items()},
        waste_proportion={int(k): v for k, v in p["waste_proportion"].items()},
        collection_frequency={int(k): v for k, v in p["collection_frequency"].items()},
        lognormal_mu=p["lognormal_mu"],
        lognormal_sigma=p["lognormal_sigma"],
        overflow_threshold=p["overflow_threshold"],
    )

    # Reconstruir dict[int, BuildingData] desde data["sets"]["I"]
    I = {
        int(k): BuildingData(
            osm_id=v['osm_id'],
            latitude=v['latitude'],
            longitude=v['longitude'],
            h_i=v['h_i']
        ) for k, v in data['sets']['I'].items()
    }

    # Reconstruir dict[int, CandidateData] desde data["sets"]["J"]
    J = {
        int(k): CandidateData(
            osm_id=v['osm_id'],
            latitude=v['latitude'],
            longitude=v['longitude'],
            context=CandidateContext(v['context'])
        ) for k, v in data['sets']['J'].items()
    }

    # Reconstruir dij desde data["distances"]
    dij = {
        int(i): {int(j): d for j, d in j_dict.items()} for i, j_dict in data['distances'].items()
    }

    i_to_idx = {b.osm_id: idx for idx, b in I.items()}
    idx_to_i = {idx: b.osm_id for idx, b in I.items()}
    j_to_idx = {int(c.osm_id): idx for idx, c in J.items()}
    idx_to_j = {idx: int(c.osm_id) for idx, c in J.items()}

    return Instance(
        study_case=data['meta']['study_case'],
        osm_radius_m=data['meta']['osm_radius_m'],
        dijkstra_radius_m=data['meta']['dijkstra_radius_m'],
        generated_at=data['meta']['generated_at'],
        references=tuple(data['meta']['references']),
        n_buildings=data['meta']['n_buildings'],
        n_candidates=data['meta']['n_candidates'],
        n_waste_types=data['meta']['n_waste_types'],
        total_population=data['meta']['total_population'],
        n_dijkstra_connections=data['meta']['n_dijkstra_connections'],
        i_to_idx=i_to_idx,
        idx_to_i=idx_to_i,
        j_to_idx=j_to_idx,
        idx_to_j=idx_to_j,
        K=data['sets']['K'],
        I=I,
        J=J,
        dij=dij,
        params=params
    )