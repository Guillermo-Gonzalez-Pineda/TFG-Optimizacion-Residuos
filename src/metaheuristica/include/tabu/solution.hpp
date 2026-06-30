#pragma once

#include <vector>
#include <utility>
#include <limits>
#include <algorithm>
#include <set>
#include <cmath>
#include "tabu/instancia.hpp"

/**
 * Representa UNA solución concreta del problema: qué puntos están abiertos,
 * a dónde va cada edificio, cuántos contenedores hay, y los agregados de coste.
 * Es lo que la búsqueda tabú muta (movimientos) y evalúa (deltas).
 */
struct SolutionState {
  std::vector<bool> open;                                      // open[j] = ¿abierto?
  std::vector<std::vector<int>>    assignment;                 // assignment[i][k] = punto (-1 sin asignar)
  std::vector<std::vector<double>> assigned_dist;              // assigned_dist[i][k] = distancia del edificio i a su punto asignado
  std::vector<std::vector<double>> demand_at;                  // demand_at[j][k] = demanda acumulada
  std::vector<std::vector<int>>    bins;                       // bins[j][k] = nº contenedores
  std::vector<std::vector<std::pair<int,int>>> buildings_at;   // buildings_at[j] = (i,k) reales en j

  double total_cost;                                           // apertura + contenedores + penalización
  int n_violations_capacity;                                   // puntos con Σ bins > max_bins
  int n_violations_coverage;                                   // pares (i,k) con assignment == -1 (descubiertos)
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


/**
 * Abre el punto `candidate` en la solución y propaga sus consecuencias:
 * reasigna a `candidate` los edificios para los que pasa a ser el punto abierto
 * más cercano, actualizando demanda, contenedores y agregados de forma incremental.
 *
 * @param solution Solución a modificar.
 * @param instance Instancia (geometría y estructuras derivadas).
 * @param candidate Índice del punto a abrir.
 */
void apply_open(SolutionState& solution, const Instance& instance, int candidate);


/**
 * Cierra el punto `candidate` en la solución y reasigna sus edificios huérfanos.
 * Cada huérfano se reasigna a su siguiente punto abierto más cercano; si no existe
 * ninguno, queda sin asignar (assignment = -1), generando una violación de cobertura.
 * Actualiza demanda y contenedores de forma incremental. No modifica el coste
 * total ni los contadores de violación (eso lo hace compute_cost).
 *
 * @param solution Solución a modificar.
 * @param instance Instancia (geometría y estructuras derivadas).
 * @param candidate Índice del punto a cerrar.
 */
void apply_close(SolutionState& solution, const Instance& instance, int candidate);


/**
 * Calcula desde cero el coste total de la solución y su número de violaciones,
 * y los guarda en solution.total_cost y solution.n_violations.
 *
 * Coste = Σ apertura(j abiertos) + Σ contenedores·coste + ρ·violaciones.
 *
 * @param solution Solución cuyo coste se calcula (se modifican sus agregados).
 * @param instance Instancia (costes de apertura, de contenedor, ρ, max_bins).
 */
void compute_cost(SolutionState& solution, const Instance& instance, double rho);
