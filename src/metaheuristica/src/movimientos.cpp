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
  for (int k = 0; k < n_types; ++k) {
    for (int i : solution.buildings_at[candidate][k]) {
      int destination = find_nearest_active(solution, instance, i, k, candidate);

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

// Devuelve (creándolo si no existe) el vector de cambios de demanda del punto p
// dentro del mapa net_demand. Si p es nuevo, lo inicializa con n_types ceros.
static std::vector<double>& demand_change_of(
    std::unordered_map<int, std::vector<double>>& net_demand,
    int p, int n_types) {
  auto it = net_demand.find(p);
  if (it == net_demand.end()) {
    it = net_demand.emplace(p, std::vector<double>(n_types, 0.0)).first;
  }
  return it->second;
}


double delta_swap(const SolutionState& solution, const Instance& instance,
                  int j_out, int j_in, double rho) {
  const int n_types  = instance.n_waste_types;
  const int max_bins = instance.params.max_bins;

  double delta = 0.0;

  // Cambio de demanda por punto afectado. Local: no toca el estado.
  std::unordered_map<int, std::vector<double>> net_demand;

  int new_coverage = 0;   // cambio neto en violaciones de cobertura

  // --- 1) Cerrar j_out: ahorrar su apertura y reubicar sus edificios ---
  delta -= instance.candidates[j_out].opening_cost;

  for (int k = 0; k < n_types; ++k) {
    for (int i : solution.buildings_at[j_out][k]) {
      // Buscar destino: el más cercano que tenga ACTIVO el tipo k (tratando j_in
      // como activo, porque el swap lo abre) y que no sea j_out (que se cierra).
      int dest = -1;
      for (const ValidCandidate& vc : instance.valid_candidates[i][k]) {
        if (vc.j == j_out) continue;
        if (vc.j == j_in || solution.active[vc.j][k]) {
          dest = vc.j;
          break;
        }
      }
      if (dest == -1) {
        new_coverage++;   // ningún destino: queda descubierto
      } else {
        demand_change_of(net_demand, dest, n_types)[k] += instance.demand[i][k];
      }
    }
  }

  // --- 2) Abrir j_in: pagar su apertura y atraer los edificios que le queden más cerca ---
  delta += instance.candidates[j_in].opening_cost;

  for (const BuildingType& bt : instance.buildings_of[j_in]) {
    const int i = bt.i;
    const int k = bt.k;

    // Los edificios que estaban en j_out ya los reasigna la Sección 1 (a su
    // nearest-open, que puede ser j_in u otro punto abierto). NO volver a
    // tocarlos aquí: hacerlo los sumaría por segunda vez a j_in (doble conteo).
    if (solution.assignment[i][k] == j_out) continue;

    // Distancia a la asignación actual del edificio.
    const double current_dist = solution.assigned_dist[i][k];

    if (bt.distance < current_dist) {
      // Este edificio migra a j_in.
      demand_change_of(net_demand, j_in, n_types)[k] += instance.demand[i][k];

      const int old = solution.assignment[i][k];
      if (old == -1) {
        new_coverage--;   // estaba descubierto → j_in lo cubre
      } else if (old != j_out) {
        // Su punto antiguo pierde esta demanda. (Si venía de j_out, ya se contó arriba.)
        demand_change_of(net_demand, old, n_types)[k] -= instance.demand[i][k];
      }
    }
  }

  delta += rho * new_coverage;


  // --- 3) Traducir cada cambio de demanda a cambio de coste de contenedores ---
  for (const auto& [point, change] : net_demand) {
    int bins_before_total = 0;
    int bins_after_total  = 0;

    for (int k = 0; k < n_types; ++k) {
      // Para j_out la base efectiva es 0 (se cierra); para el resto, su demanda actual.
      const double base      = (point == j_out) ? 0.0 : solution.demand_at[point][k];
      const int    bins_now  = (point == j_out) ? 0   : solution.bins[point][k];
      const int    bins_after = bins_for_demand(base + change[k],
                                    instance.params.bin_capacity[k]);

      delta += (bins_after - bins_now) * instance.params.bin_cost[k];
      bins_before_total += bins_now;
      bins_after_total  += bins_after;
    }

    // Cambio de saturación (j_out se cierra, no puede saturar).
    if (point != j_out) {
      bool sat_before = (bins_before_total > max_bins);
      bool sat_after  = (bins_after_total  > max_bins);
      if (!sat_before && sat_after) delta += rho;
      if (sat_before && !sat_after) delta -= rho;
    }
  }


  // --- 4) Quitar el coste de los contenedores actuales de j_out (se cierra) ---
  int bins_out_total = 0;
  for (int k = 0; k < n_types; ++k) {
    delta -= solution.bins[j_out][k] * instance.params.bin_cost[k];
    bins_out_total += solution.bins[j_out][k];
  }
  if (bins_out_total > max_bins) delta -= rho;   // dejaba de estar saturado

  return delta;
}