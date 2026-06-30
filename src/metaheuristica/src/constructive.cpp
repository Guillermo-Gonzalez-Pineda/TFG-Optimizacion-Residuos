#include "tabu/constructive.hpp"

#include <limits>

// Cuenta cuántos pares (edificio, tipo) DESCUBIERTOS cubriría el candidato j.
static int count_newly_covered(const SolutionState& solution,
                               const Instance& instance, int candidate) {
  int count = 0;
  for (const BuildingType& bt : instance.buildings_of[candidate]) {
    if (solution.assignment[bt.i][bt.k] == -1) {
      count++;
    }
  }
  return count;
}

// Devuelve el total de contenedores de un punto (suma sobre todos los tipos).
static int total_bins_at(const SolutionState& solution, const Instance& instance,
                         int point) {
  int total = 0;
  for (int k = 0; k < instance.n_waste_types; ++k) {
    total += solution.bins[point][k];
  }
  return total;
}

/**
 * Busca un punto saturado (Σ bins > max_bins) en la solución.
 * Devuelve el índice del primer punto saturado que encuentre, o -1 si no hay ninguno.
 */
static int find_saturated_point(const SolutionState& solution, const Instance& instance) {
  const int max_bins = instance.params.max_bins;
  for (int j = 0; j < instance.n_candidates; ++j) {
    if (!solution.open[j]) continue;
    if (total_bins_at(solution, instance, j) > max_bins) {
      return j;
    }
  }
  return -1;
}

/**
 * Busca un candidato cerrado que pueda aliviar la saturación de un punto dado.
 * Recorre los edificios asignados al punto saturado y sus candidatos válidos.
 * Devuelve el índice del primer candidato cerrado que pueda atraer a alguno de
 * esos edificios, o -1 si no hay ninguno.
 */
static int find_reliever(const SolutionState& solution, const Instance& instance,
                         int saturated_point) {
  // Recorrer los edificios asignados al punto saturado.
  for (const auto& [i, k] : solution.buildings_at[saturated_point]) {
    // Mirar sus candidatos válidos (ordenados por cercanía).
    for (const ValidCandidate& vc : instance.valid_candidates[i][k]) {
      // Un candidato cerrado y distinto del saturado puede atraer a este edificio.
      if (!solution.open[vc.j] && vc.j != saturated_point) {
        return vc.j;
      }
    }
  }
  return -1;   // ningún candidato cerrado puede aliviar este punto
}

void construct_initial(SolutionState& solution, const Instance& instance, double rho) {
  // Partir de una solución vacía.
  init_empty(solution, instance);

  const int n_candidates = instance.n_candidates;

  // --- FASE 1: abrir puntos hasta cubrir todos los pares ---
  while (true) {
    int best_candidate = -1;
    int best_coverage  = 0;

    // Buscar el candidato cerrado que cubre más pares (i,k) descubiertos.
    for (int j = 0; j < n_candidates; ++j) {
      if (solution.open[j]) continue;            // ya abierto, saltar
      const int coverage = count_newly_covered(solution, instance, j);
      if (coverage > best_coverage) {
        best_coverage = coverage;
        best_candidate = j;
      }
    }

    if (best_candidate == -1 || best_coverage == 0) {
      break;
    }

    apply_open(solution, instance, best_candidate);
  }

  // --- FASE 2: aliviar saturación abriendo puntos cercanos ---
  while (true) {
    const int saturated = find_saturated_point(solution, instance);
    if (saturated == -1) {
      break;
    }

    const int reliever = find_reliever(solution, instance, saturated);
    if (reliever == -1) {
      break;   // este punto no se puede aliviar abriendo nada: rendición honesta
    }

    // Abrir el candidato que alivia. apply_open reasigna los edificios que
    // ahora le quedan más cerca, reduciendo la demanda (y bins) del saturado.
    apply_open(solution, instance, reliever);
  }

  // Calcular el coste final de la solución construida.
  compute_cost(solution, instance, rho);
}