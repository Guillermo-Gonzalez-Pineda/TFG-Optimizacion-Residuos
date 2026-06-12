"""
Generación de instancias para el problema de localización de contenedores de
residuos.

Este subpaquete agrupa todo el pipeline que, a partir de OpenStreetMap, produce
una `Instance` lista para optimizar. Está dividido por responsabilidad:

    - grafo       : descarga la red peatonal de OSMnx, la filtra, la proyecta a
                    UTM e inserta puntos sobre aristas (download_graph,
                    project_to_utm, insert_point_on_edge).
    - extraccion  : extrae candidatos (nodos) y edificios (huellas) de OSM, los
                    clasifica y consolida (extract_candidates, extract_buildings,
                    classify_candidate_context, consolidate_candidates,
                    cost_by_degree).
    - distancias  : calcula distancias edificio-candidato con Dijkstra y garantiza
                    la cobertura insertando contenedores artificiales
                    (compute_distances, evaluate_coverage, ensure_coverage).
    - pipeline    : ensambla los componentes en una `Instance` y la serializa a
                    JSON (build_instance, save_instance).

El script de entrada que encadena las etapas vive en `generar.py` (en
`src/python/`). Este `__init__` re-exporta la API pública de los cuatro módulos
para que los consumidores externos importen sin acoplarse a la estructura interna
(p. ej. `from generacion import build_instance, download_graph`).

Referencias:
    Li et al. (2026). Waste Management 209, 115211.
    Boeing (2025). Geographical Analysis 57, 567-577.
"""

from __future__ import annotations

from .grafo import (
    download_graph,
    project_to_utm,
    insert_point_on_edge,
)
from .extraccion import (
    cost_by_degree,
    extract_candidates,
    classify_candidate_context,
    consolidate_candidates,
    extract_buildings,
)
from .distancias import (
    compute_distances,
    evaluate_coverage,
    ensure_coverage,
)
from .pipeline import (
    build_instance,
    save_instance,
)

__all__ = [
    # grafo
    "download_graph",
    "project_to_utm",
    "insert_point_on_edge",
    # extraccion
    "cost_by_degree",
    "extract_candidates",
    "classify_candidate_context",
    "consolidate_candidates",
    "extract_buildings",
    # distancias
    "compute_distances",
    "evaluate_coverage",
    "ensure_coverage",
    # pipeline
    "build_instance",
    "save_instance",
]
