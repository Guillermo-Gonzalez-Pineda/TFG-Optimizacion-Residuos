#pragma once

#include <string>

#include "tabu/instancia.hpp"
#include "tabu/solution.hpp"

/**
 * Guarda una solución en un fichero JSON legible (indentado).
 *
 * El JSON contiene dos niveles de información:
 *
 *  1) RESUMEN (mismo formato que el resumen del exacto):
 *     - "cost":          coste total de la solución.
 *     - "runtime":       tiempo de ejecución en segundos.
 *     - "radius":        radio OSM (m) de la instancia, para trazabilidad.
 *     - "n_violations":  nº de puntos que superan max_bins.
 *     - "n_points_open": nº de puntos abiertos.
 *     - "total_bins":    suma de todos los contenedores.
 *     - "bins_per_type": {"k": Σ contenedores del tipo k}.
 *     - "open_points":   índices de los puntos abiertos, en orden ascendente.
 *
 *  2) DETALLE COMPLETO (información que NO está en el resumen del exacto):
 *     - "assignment":    {"i": {"k": punto}} solo para los pares con asignación.
 *     - "bins_detail":   {"j": {"k": contenedores}} por cada punto abierto.
 *     - "demand_detail": {"j": {"k": demanda}}     por cada punto abierto.
 *
 * Crea el directorio de salida si no existe.
 *
 * @param solution    Solución a guardar.
 * @param instance    Instancia asociada (dimensiones, costes, etc.).
 * @param output_path Ruta del fichero JSON de salida.
 * @param runtime     Tiempo de ejecución en segundos.
 */
void save_solution(const SolutionState& solution, const Instance& instance,
                   const std::string& output_path, double runtime);
