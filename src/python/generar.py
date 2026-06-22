"""
Script de entrada del generador de instancias para el problema de localización de
contenedores de residuos.

Encadena las 7 etapas del pipeline (definidas en el paquete `generacion`) sobre
un caso de estudio concreto y guarda la instancia resultante en JSON, junto con
el grafo (GraphML) y los edificios (GeoJSON) para los mapas.

    1. download_graph / project_to_utm    (generacion.grafo)
    2. extract_candidates                  (generacion.extraccion)
    3. extract_buildings                   (generacion.extraccion)
    4. classify_candidate_context          (generacion.extraccion)
    5. consolidate_candidates              (generacion.extraccion)
    6. compute_distances                   (generacion.distancias)
   6b. ensure_coverage                     (generacion.distancias)
    7. build_instance + save_instance      (generacion.pipeline)

Uso:
    python3 generar.py [radio_en_metros]   # por defecto 500

Referencias:
    Li et al. (2026). Waste Management 209, 115211.
    Boeing (2025). Geographical Analysis 57, 567-577.
"""

from __future__ import annotations

from instancia import (
    GeographicConfig,
    ModelParameters,
)
from generacion import (
    download_graph,
    project_to_utm,
    extract_candidates,
    extract_buildings,
    classify_candidate_context,
    consolidate_candidates,
    compute_distances,
    ensure_coverage,
    evaluate_coverage,
    build_instance,
    save_instance,
)


if __name__ == "__main__":
    import sys

    import osmnx as ox
    from shapely.geometry.base import BaseGeometry

    # Las funciones del pipeline ya están importadas desde `generacion` arriba.

    # Radio como argumento: python3 generar.py 750
    radius = int(sys.argv[1]) if len(sys.argv) > 1 else 500

    config = GeographicConfig(
        place="Plaza del Cristo, San Cristóbal de La Laguna, España",
        radius=radius,
        network_type="walk",
        cutoff_dijkstra=275,
        min_node_degree=2,
        graph_margin=100,
        base_opening_cost=4000.0,
    )

    params = ModelParameters(
        opening_cost=4000.0,
        max_bins=8,
        nimby_distance=0.0,
        waste_per_capita=1.50,
        overflow_penalty=500.0,
        bin_cost={0: 350.0, 1: 300.0, 2: 250.0, 3: 500.0},
        bin_capacity={0: 120.0, 1: 120.0, 2: 120.0, 3: 120.0},
        coverage_radius={0: 100.0, 1: 100.0, 2: 150.0, 3: 275.0},
        waste_proportion={0: 0.40, 1: 0.22, 2: 0.37, 3: 0.01},
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
