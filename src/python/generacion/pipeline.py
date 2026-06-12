"""
Ensamblado final y serialización de la instancia (módulo `pipeline` del paquete
`generacion`).

Pipeline completo (las etapas 1-6 viven en los módulos hermanos del paquete):
    1. download_graph / project_to_utm    (grafo)
    2. extract_candidates                  (extraccion)
    3. extract_buildings                   (extraccion)
    4. classify_candidate_context          (extraccion)
    5. consolidate_candidates              (extraccion)
    6. compute_distances                   (distancias)
   6b. ensure_coverage                     (distancias)
    7. build_instance + save_instance      (este módulo)

Este módulo solo contiene el ensamblado final (build_instance) y la
serialización a JSON (save_instance). El script de entrada que encadena todos
los pasos vive en `generar.py` (en `src/python/`). Toda la lógica de grafo,
extracción y distancias vive en los módulos hermanos del paquete.

Referencias:
    Li et al. (2026). Waste Management 209, 115211.
    Boeing (2025). Geographical Analysis 57, 567-577.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

from instancia import (
    BuildingData,
    CandidateData,
    GeographicConfig,
    Instance,
    ModelParameters,
)


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
    """Ensambla todos los componentes en un objeto Instance completo."""
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
                    "opening_cost": c.opening_cost,
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
