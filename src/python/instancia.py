"""
Instance generation for the waste collection point location problem.

Builds the HDM instance from OSMnx data (geographic graph + building
footprints) and serialises it to JSON. Designed to extend to HVM.

References:
    Li et al. (2026). Waste Management 209, 115211.
    Boeing (2025). Geographical Analysis 57, 567-577.
"""

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
    j_to_idx: dict[str, int]
    idx_to_j: dict[int, str]

    # --- Sets (with internal indices) ---
    K: list[int]               # [0, 1, 2, 3]
    I: dict[int, BuildingData] # idx → building data
    J: dict[int, CandidateData] # idx → candidate data

    # --- Distances (sparse, internal indices) ---
    dij: dict[int, dict[int, float]]  # dij[i_idx][j_idx] = distance in meters

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
    """Load a problem instance from a JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)

    # Convert nested dicts to dataclasses
    i_to_idx = data['i_to_idx']
    idx_to_i = {int(k): v for k, v in data['idx_to_i'].items()}
    j_to_idx = data['j_to_idx']
    idx_to_j = {int(k): v for k, v in data['idx_to_j'].items()}

    I = {int(k): BuildingData(**v) for k, v in data['I'].items()}
    J = {int(k): CandidateData(**v) for k, v in data['J'].items()}

    dij = {int(i): {int(j): d for j, d in j_dict.items()} for i, j_dict in data['dij'].items()}

    params = ModelParameters(**data['params'])

    return Instance(
        study_case=data['study_case'],
        osm_radius_m=data['osm_radius_m'],
        dijkstra_radius_m=data['dijkstra_radius_m'],
        generated_at=data['generated_at'],
        references=data['references'],
        n_buildings=data['n_buildings'],
        n_candidates=data['n_candidates'],
        n_waste_types=data['n_waste_types'],
        total_population=data['total_population'],
        n_dijkstra_connections=data['n_dijkstra_connections'],
        i_to_idx=i_to_idx,
        idx_to_i=idx_to_i,
        j_to_idx=j_to_idx,
        idx_to_j=idx_to_j,
        K=data['K'],
        I=I,
        J=J,
        dij=dij,
        params=params
    )