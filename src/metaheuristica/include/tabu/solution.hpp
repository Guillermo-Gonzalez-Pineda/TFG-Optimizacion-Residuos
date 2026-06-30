#pragma once

#include <vector>
#include <utility>
#include "tabu/instancia.hpp"

/**
 * Representa UNA solución concreta del problema: qué puntos están abiertos,
 * a dónde va cada edificio, cuántos contenedores hay, y los agregados de coste.
 * Es lo que la búsqueda tabú muta (movimientos) y evalúa (deltas).
 */
struct SolutionState {
  std::vector<bool> open;                                      // open[j] = ¿abierto?
  std::vector<std::vector<int>>    assignment;                 // assignment[i][k] = punto (-1 sin asignar)
  std::vector<std::vector<double>> demand_at;                  // demand_at[j][k] = demanda acumulada
  std::vector<std::vector<int>>    bins;                       // bins[j][k] = nº contenedores
  std::vector<std::vector<std::pair<int,int>>> buildings_at;   // buildings_at[j] = (i,k) reales en j
  double total_cost;                                           // apertura + contenedores + penalización
  int    n_violations;                                         // nº de puntos con Σ bins > max_bins
};

/**
 * Inicializa una solución vacía (ningún punto abierto) para la instancia dada.
 * Deja todos los vectores dimensionados y en su valor "vacío": nada abierto,
 * nadie asignado, sin demanda ni contenedores.
 *
 * @param solution Referencia a la solución a inicializar.
 * @param instance Instancia para la que se inicializa la solución.
 */
void init_empty(SolutionState& solution, const Instance& instance);