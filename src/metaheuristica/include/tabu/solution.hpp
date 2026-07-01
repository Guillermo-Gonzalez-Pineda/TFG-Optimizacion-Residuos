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
  std::vector<std::vector<bool>>   active;                     // active[j][k] = ¿el punto j sirve el tipo k? (= w[j,k] del exacto)
  std::vector<std::vector<int>>    assignment;                 // assignment[i][k] = punto (-1 sin asignar)
  std::vector<std::vector<double>> assigned_dist;              // assigned_dist[i][k] = distancia del edificio i a su punto asignado
  std::vector<std::vector<double>> demand_at;                  // demand_at[j][k] = demanda acumulada
  std::vector<std::vector<int>>    bins;                       // bins[j][k] = nº contenedores
  std::vector<std::vector<std::vector<int>>> buildings_at;     // buildings_at[j][k] = edificios i cuyo tipo k se sirve en j

  double total_cost;                                           // apertura + contenedores + penalización
  int n_violations_capacity;                                   // puntos con Σ bins > max_bins
  int n_violations_coverage;                                   // pares (i,k) con assignment == -1 (descubiertos)

  /**
   * ¿Está ABIERTO el punto j? En el modelo de activación por tipo un punto está
   * abierto (paga apertura C_j, "z[j]") si y solo si sirve ALGÚN tipo:
   *   is_open(j) = OR sobre k de active[j][k].
   * Es el detector del flip z[j] 0↔1 que usarán los deltas: la primera
   * activación abre el punto (paga C_j); la última desactivación lo cierra.
   */
  bool is_open(int j) const {
    for (bool a : active[j]) {
      if (a) return true;
    }
    return false;
  }
};

/**
 * Calcula el número de contenedores necesarios para una demanda dada.
 * Función pura: techo(demanda / capacidad). No toca estado.
 */
int bins_for_demand(double demand, double capacity);

/**
 * Encuentra el punto más cercano que tiene ACTIVO el tipo k (active[j][k]=1) y
 * por tanto puede servir al par (edificio i, tipo k), excluyendo `exclude`.
 * Traduce la restricción (9) del exacto: nearest a la instalación que tiene el
 * contenedor de ese tipo (w[j,k]=1). En régimen colapsado (punto entero o entero
 * apagado) coincide con "nearest abierto", pero se generaliza al modelo per-tipo.
 */
int find_nearest_active(const SolutionState& solution, const Instance& instance,
                        int i, int k, int exclude);

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
 * Activa el tipo `type` en el punto `candidate` (active[candidate][type]=true).
 * Es apply_open pero para UN SOLO tipo: atrae los edificios de ese tipo para los
 * que 'candidate' pasa a ser el punto ACTIVO más cercano, actualizando demanda,
 * contenedores (del tipo) y asignaciones de forma incremental.
 *
 * Contabiliza el coste de apertura z[j]: suma C_j a total_cost SOLO si esta
 * activación ABRE el punto (era el primer tipo; is_open pasa de false a true).
 * A diferencia de apply_open, esta primitiva SÍ toca total_cost (solo el término
 * de apertura; bins y penalizaciones siguen siendo de compute_cost). No-op si el
 * tipo ya estaba activo.
 */
void apply_activate(SolutionState& solution, const Instance& instance,
                    int candidate, int type);


/**
 * Desactiva el tipo `type` en el punto `candidate` (active[candidate][type]=false).
 * Es apply_close pero para UN SOLO tipo: los huérfanos de ese tipo se reasignan a
 * su siguiente punto ACTIVO más cercano (find_nearest_active); si no hay, quedan
 * descubiertos (assignment = -1).
 *
 * Contabiliza z[j]: resta C_j de total_cost SOLO si esta desactivación CIERRA el
 * punto (era el último tipo; is_open pasa de true a false). No-op si el tipo ya
 * estaba inactivo.
 */
void apply_deactivate(SolutionState& solution, const Instance& instance,
                      int candidate, int type);


/**
 * Intercambia: cierra j_out y abre j_in. Se apoya en las primitivas existentes
 * aplicadas en secuencia (cerrar y luego abrir), de modo que apply_open ve el
 * estado ya actualizado por apply_close y la interacción se resuelve sola.
 */
void apply_swap(SolutionState& solution, const Instance& instance,
                int j_out, int j_in);


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
