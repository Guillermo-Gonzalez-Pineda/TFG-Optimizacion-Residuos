#include "tabu/movimientos.hpp"
#include <limits>
#include <cmath>
#include <unordered_map>

double delta_close(const SolutionState& solution, const Instance& instance,
                   int candidate, double rho) {
  const int n_types  = instance.n_waste_types;
  const int max_bins = instance.params.max_bins;

  double delta = 0.0;

  // --- 1) Lo que ahorramos al cerrar 'candidate' ---
  delta -= instance.candidates[candidate].opening_cost;

  // Contenedores actuales de 'candidate': dejan de costar.
  int total_bins_candidate = 0;
  for (int k = 0; k < n_types; ++k) {
    delta -= solution.bins[candidate][k] * instance.params.bin_cost[k];
    total_bins_candidate += solution.bins[candidate][k];
  }

  // Si 'candidate' estaba saturado, esa violación de capacidad desaparece.
  if (total_bins_candidate > max_bins) {
    delta -= rho;   // quitamos una penalización de capacidad
  }

  // --- 2) Reubicar los huérfanos: acumular su demanda extra por destino ---
  // Mapa local: cuánta demanda extra recibe cada (destino, tipo). NO toca el estado.
  std::unordered_map<int, std::vector<double>> extra_demand;
  // (clave = punto destino; valor = vector de n_types con la demanda extra por tipo)

  int new_coverage_violations = 0;   // huérfanos que no encuentran destino

  // Bloque 2: buscar destino con la helper compartida.
  for (const auto& [i, k] : solution.buildings_at[candidate]) {
    int destination = find_nearest_open(solution, instance, i, k, candidate);

    if (destination == -1) {
      ++new_coverage_violations;
      
    } else {
      auto it = extra_demand.find(destination);
      if (it == extra_demand.end()) {
        it = extra_demand.emplace(destination, std::vector<double>(n_types, 0.0)).first;
      }
      it->second[k] += instance.demand[i][k];
    }
  }

  delta += rho * new_coverage_violations;

  // Bloque 3: usar bins_for_demand en vez del ceil inline.
  for (const auto& [destination, extra] : extra_demand) {
    int bins_before_total = 0, bins_after_total = 0;
    for (int k = 0; k < n_types; ++k) {
      const int    bins_now     = solution.bins[destination][k];
      const double demand_after = solution.demand_at[destination][k] + extra[k];
      const int    bins_after   = bins_for_demand(demand_after,
                                       instance.params.bin_capacity[k]);
      delta += (bins_after - bins_now) * instance.params.bin_cost[k];
      bins_before_total += bins_now;
      bins_after_total  += bins_after;
    }
    bool sat_before = (bins_before_total > max_bins);
    bool sat_after  = (bins_after_total  > max_bins);
    if (!sat_before && sat_after) delta += rho;
    if (sat_before && !sat_after) delta -= rho;
  }
  return delta;
}



double delta_open(const SolutionState& solution, const Instance& instance,
                  int candidate, double rho) {
  const int n_types  = instance.n_waste_types;
  const int max_bins = instance.params.max_bins;

  double delta = 0.0;

  // 1) Pagar la apertura del punto.
  delta += instance.candidates[candidate].opening_cost;

  std::vector<double> gained(n_types, 0.0);
  std::unordered_map<int, std::vector<double>> lost;   // lost[punto_antiguo][k]

  // 2) Recorrer los pares que PODRÍAN usar 'candidate' (la inversa).
  for (const BuildingType& bt : instance.buildings_of[candidate]) {
    const int i = bt.i;
    const int k = bt.k;

    if (bt.distance < solution.assigned_dist[i][k]) {
      gained[k] += instance.demand[i][k];

      const int old_point = solution.assignment[i][k];
      if (old_point == -1) {
        delta -= rho;

      } else {
        // Tenía punto antiguo: ese punto pierde su demanda.
        auto it = lost.find(old_point);
        if (it == lost.end()) {
          it = lost.emplace(old_point, std::vector<double>(n_types, 0.0)).first;
        }
        it->second[k] += instance.demand[i][k];
      }
    }
  }

  // 3) Coste de los contenedores que 'candidate' necesita para lo que gana.
  //    'candidate' parte de demanda 0 (estaba cerrado), así que sus bins nuevos
  //    salen directamente de 'gained'.
  int bins_candidate_total = 0;
  for (int k = 0; k < n_types; ++k) {
    const int bins_new = bins_for_demand(gained[k], instance.params.bin_capacity[k]);
    delta += bins_new * instance.params.bin_cost[k];
    bins_candidate_total += bins_new;
  }

  if (bins_candidate_total > max_bins) delta += rho;

  // 4) Ahorro en los puntos antiguos que pierden demanda.
  for (const auto& [old_point, lost_k] : lost) {
    int bins_before_total = 0, bins_after_total = 0;
    for (int k = 0; k < n_types; ++k) {
      const int    bins_now     = solution.bins[old_point][k];
      const double demand_after = solution.demand_at[old_point][k] - lost_k[k];
      const int    bins_after   = bins_for_demand(demand_after,
                                       instance.params.bin_capacity[k]);
      delta += (bins_after - bins_now) * instance.params.bin_cost[k];  // negativo: ahorro
      bins_before_total += bins_now;
      bins_after_total  += bins_after;
    }
    bool sat_before = (bins_before_total > max_bins);
    bool sat_after  = (bins_after_total  > max_bins);
    if (sat_before && !sat_after) delta -= rho;   // deja de saturarse
    if (!sat_before && sat_after) delta += rho;    // (raro al quitar demanda, por completitud)
  }

  return delta;
}


double delta_swap(const SolutionState& solution, const Instance& instance,
                  int j_out, int j_in, double rho) {
  SolutionState copy = solution;          // copia para no mutar el original
  apply_swap(copy, instance, j_out, j_in);
  compute_cost(copy, instance, rho);
  return copy.total_cost - solution.total_cost;
}