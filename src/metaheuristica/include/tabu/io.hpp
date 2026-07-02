#pragma once

#include <string>

#include "tabu/instancia.hpp"
#include "tabu/solution.hpp"

/**
 * Guarda una solución de la metaheurística en JSON, en el MISMO formato que
 * consume el pipeline de análisis en Python (analisis.carga.cargar_solucion),
 * espejo del exacto y la lagrangiana. Emite exactamente estos campos (claves
 * string, anidado {"j":{"k":v}} para las estructuras de doble índice):
 *
 *   - "metodo":   "tabu".
 *   - "z":        {"j": 0|1}          TODOS los candidatos (= is_open(j)).
 *   - "x":        {"j": {"k": bins}}  TODOS los candidatos (= bins[j][k]).
 *   - "w":        {"j": {"k": 0|1}}   TODOS los candidatos (= active[j][k]).
 *   - "y_assign": {"i": {"k": punto}} TODOS los edificios y tipos
 *                 (assignment[i][k], -1 si el par (i,k) quedó sin asignar).
 *   - "cost":     coste total de la solución.
 *   - "runtime":  segundos de la búsqueda (SOLO tabu_search, sin carga ni I/O).
 *   - "modo":     "per-tipo" | "entero" (vecindario activo del flag).
 *
 * z/x/w cubren TODOS los candidatos (los cerrados con 0), igual que el exacto,
 * para que las comparativas entre métodos sean uniformes. Crea el directorio de
 * salida si no existe.
 *
 * @param solution    Solución a guardar.
 * @param instance    Instancia asociada (dimensiones).
 * @param output_path Ruta del fichero JSON de salida.
 * @param runtime     Segundos de la llamada a tabu_search.
 * @param mode        "per-tipo" | "entero" (para el campo "modo").
 */
void save_solution(const SolutionState& solution, const Instance& instance,
                   const std::string& output_path, double runtime,
                   const std::string& mode);
