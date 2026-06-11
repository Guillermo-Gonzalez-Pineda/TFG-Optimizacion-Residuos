"""
Generador de instancias para el problema de localización de contenedores de
residuos (módulo 4/4: orquestación).

Pipeline completo:
    1. download_graph / project_to_utm    (generador_grafo)
    2. extract_candidates                  (generador_extraccion)
    3. extract_buildings                   (generador_extraccion)
    4. classify_candidate_context          (generador_extraccion)
    5. consolidate_candidates              (generador_extraccion)
    6. compute_distances                   (generador_distancias)
   6b. ensure_coverage                     (generador_distancias)
    7. build_instance + save_instance      (este módulo)

Este módulo solo contiene el ensamblado final (build_instance), la
serialización a JSON (save_instance) y el __main__ que encadena los pasos. Toda
la lógica de grafo, extracción y distancias vive en los tres módulos hermanos.

Uso:
    python3 generador.py [radio_en_metros]   # por defecto 500

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

# ── Fachada del paquete: re-exporta la API pública de los submódulos ──────
# build_instance / save_instance viven aquí; el resto del pipeline vive en los
# módulos hermanos. Re-exportarlos desde `generador` permite seguir haciendo
# `from generador import download_graph, ...` (notebooks) sin acoplar a la
# estructura interna de módulos.
from generador_grafo import (
    download_graph,
    project_to_utm,
    insert_point_on_edge,
)
from generador_extraccion import (
    cost_by_degree,
    extract_candidates,
    classify_candidate_context,
    consolidate_candidates,
    extract_buildings,
)
from generador_distancias import (
    compute_distances,
    evaluate_coverage,
    ensure_coverage,
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


if __name__ == "__main__":
    import sys

    import osmnx as ox
    from shapely.geometry.base import BaseGeometry

    # Las funciones del pipeline ya están importadas como fachada arriba.

    # Radio como argumento: python3 generador.py 750
    radius = int(sys.argv[1]) if len(sys.argv) > 1 else 500

    config = GeographicConfig(
        place="Plaza del Cristo, San Cristóbal de La Laguna, España",
        radius=radius,
        network_type="walk",
        cutoff_dijkstra=350,
        min_node_degree=2,
        graph_margin=100,
        base_opening_cost=4000.0,
    )

    params = ModelParameters(
        opening_cost=4000.0,
        max_bins=8,
        nimby_distance=0.0,
        waste_per_capita=1.32,
        overflow_penalty=500.0,
        bin_cost={0: 350.0, 1: 300.0, 2: 250.0, 3: 500.0},
        bin_capacity={0: 120.0, 1: 120.0, 2: 120.0, 3: 120.0},
        coverage_radius={0: 100.0, 1: 100.0, 2: 100.0, 3: 250.0},
        waste_proportion={0: 0.5012, 1: 0.0791, 2: 0.3885, 3: 0.0312},
        collection_frequency={0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0},
        lognormal_mu=0.0,
        lognormal_sigma=0.25,
        overflow_threshold=0.05,
    )

    print("Step 1/7 — Downloading graph...")
    graph = download_graph(config)
    print(f"          Nodes: {graph.number_of_nodes()}  Edges: {graph.number_of_edges()}")

    # Grafo UTM SOLO para compute_distances / ensure_coverage.
    graph_utm = project_to_utm(graph)

    # Guardar el grafo lat/lon para mapas.
    graph_path = f"data/processed/graph_{radius}m.graphml"
    ox.save_graphml(graph, graph_path)          # ← graph, NO graph_utm
    print(f"          Graph saved: {graph_path}")

    print("Step 2/7 — Extracting candidates...")
    candidates, idx_to_j, j_to_idx = extract_candidates(graph, config)
    print(f"          Candidates: {len(candidates)}")

    print("Step 3/7 — Extracting buildings...")
    buildings, idx_to_i, i_to_idx, gdf_buildings = extract_buildings(config)
    print(f"          Buildings: {len(buildings)}")

    # Limpiar columnas problemáticas para GeoJSON (listas/dicts/geometrías → str).
    for col in gdf_buildings.columns:
        if col != "geometry":
            if gdf_buildings[col].apply(
                lambda x: isinstance(x, (list, dict, BaseGeometry))
            ).any():
                gdf_buildings[col] = gdf_buildings[col].astype(str)

    buildings_path = f"data/processed/buildings_{radius}m.geojson"
    gdf_buildings.to_file(buildings_path, driver="GeoJSON")
    print(f"          Buildings GeoJSON saved: {buildings_path}")

    print("Step 4/7 — Classifying candidate context...")
    candidates = classify_candidate_context(config, candidates, graph, idx_to_j)

    print("Step 5/7 — Consolidating candidates...")
    candidates, idx_to_j, j_to_idx = consolidate_candidates(candidates, graph, idx_to_j)

    print("Step 6/7 — Computing distances...")
    dij = compute_distances(graph_utm, buildings, candidates, idx_to_j, config.cutoff_dijkstra)
    print(f"          Connections: {sum(len(v) for v in dij.values())}")

    # ── Step 6b: Garantizar cobertura total ──────────────────
    buildings, candidates, dij, idx_to_j, j_to_idx = ensure_coverage(
        graph_utm,      # UTM → compute_distances / proyecciones
        buildings, candidates, dij, params,
        idx_to_j, j_to_idx, config.cutoff_dijkstra,
    )
    # Actualizar mapas de edificios por si se eliminaron edificios inalcanzables.
    idx_to_i = {idx: osm for idx, osm in idx_to_i.items() if idx in buildings}
    i_to_idx = {osm: idx for idx, osm in idx_to_i.items()}

    print("Step 7/7 — Building and saving instance...")
    uncovered = evaluate_coverage(buildings, dij, params)
    if uncovered:
        print(f"          ⚠ {len(uncovered)} buildings without full coverage")
    else:
        print("          ✓ Full coverage")

    instance = build_instance(
        config, params, buildings, candidates,
        i_to_idx, idx_to_i, j_to_idx, idx_to_j, dij,
    )

    output_path = f"data/processed/instancia_laguna_{radius}m.json"
    save_instance(instance, output_path)
    print(f"\n✓ Instance saved to {output_path}")
    print(f"  Buildings:   {instance.n_buildings}")
    print(f"  Candidates:  {instance.n_candidates}")
    print(f"  Connections: {instance.n_dijkstra_connections}")
