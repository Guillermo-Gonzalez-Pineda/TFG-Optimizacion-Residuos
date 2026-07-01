#include "tabu/solution.hpp"


int bins_for_demand(double demand, double capacity) {
  if (demand <= 1e-9) return 0;   // residuo FP de un punto vaciado (~1e-14): no es demanda real
  return static_cast<int>(std::ceil(demand / capacity));
}

int find_nearest_open(const SolutionState& solution, const Instance& instance,
                      int i, int k, int exclude) {
  for (const ValidCandidate& vc : instance.valid_candidates[i][k]) {
    if (vc.j != exclude && solution.open[vc.j]) {
      return vc.j;   // el primero abierto es el más cercano (lista ordenada)
    }
  }
  return -1;
}

void init_empty(SolutionState& solution, const Instance& instance) {
  const int n_buildings  = instance.n_buildings;
  const int n_candidates = instance.n_candidates;
  const int n_types      = instance.n_waste_types;

  solution.open.assign(n_candidates, false);
  solution.assignment.assign(n_buildings, std::vector<int>(n_types, -1));
  solution.assigned_dist.assign(n_buildings,
      std::vector<double>(n_types, std::numeric_limits<double>::infinity()));
  solution.demand_at.assign(n_candidates, std::vector<double>(n_types, 0.0));
  solution.bins.assign(n_candidates, std::vector<int>(n_types, 0));
  solution.buildings_at.assign(n_candidates, std::vector<std::pair<int,int>>{});

  solution.total_cost   = 0.0;
  solution.n_violations_capacity = 0;
  solution.n_violations_coverage = 0;
}


/**
 * Recalcula el número de contenedores necesarios para un punto y tipo de residuo
 * dado, usando la demanda acumulada y la capacidad del contenedor.
 * Actualiza el vector `solution.bins` en consecuencia.
 * 
 * @param solution Referencia a la solución a modificar.
 * @param instance Instancia para acceder a la capacidad del contenedor.
 * @param point Índice del punto (candidato) a recalcular.
 * @param k Índice del tipo de residuo.
 */
static void recompute_bins(SolutionState& solution, const Instance& instance,
                           int point, int k) {
  solution.bins[point][k] = bins_for_demand(solution.demand_at[point][k], 
                                            instance.params.bin_capacity[k]);
}


void apply_open(SolutionState& solution, const Instance& instance, int candidate) {
  if (solution.open[candidate]) return;

  solution.open[candidate] = true;
  
  std::set<int> touched; // Conjunto de puntos cuya demanda cambia.
  touched.insert(candidate);


  for (const BuildingType& bt : instance.buildings_of[candidate]) {
    const int i = bt.i;         
    const int k = bt.k;     
    const double new_dist = bt.distance; 

    if (new_dist < solution.assigned_dist[i][k]) {
      const int old_point = solution.assignment[i][k];

      if (old_point != -1) {
        solution.demand_at[old_point][k] -= instance.demand[i][k];
        auto& old_list = solution.buildings_at[old_point];
        old_list.erase(std::remove(old_list.begin(), old_list.end(),
                                   std::make_pair(i, k)), old_list.end());
        touched.insert(old_point);
      }

      solution.assignment[i][k]    = candidate;
      solution.assigned_dist[i][k] = new_dist;
      solution.demand_at[candidate][k] += instance.demand[i][k];
      solution.buildings_at[candidate].push_back(std::make_pair(i, k));
    }
  }

  // Recalcular los contenedores de todos los puntos tocados.
  for (int point : touched) {
    for (int k = 0; k < instance.n_waste_types; ++k) {
      recompute_bins(solution, instance, point, k);
    }
  }
}


void apply_close(SolutionState& solution, const Instance& instance, int candidate) {
  // Si ya estaba cerrado, no hay nada que hacer.
  if (!solution.open[candidate]) return;

  // 1) Marcar el punto como cerrado.
  solution.open[candidate] = false;

  std::set<int> touched;

  // 2) Tomar los huérfanos: los pares (i,k) que REALMENTE estaban en 'candidate'.
  //    Copiamos la lista porque vamos a modificar buildings_at mientras iteramos.
  const std::vector<std::pair<int,int>> orphans = solution.buildings_at[candidate];

  // 3) Vaciar el punto cerrado: ya no sirve a nadie, sin demanda ni contenedores.
  solution.buildings_at[candidate].clear();
  for (int k = 0; k < instance.n_waste_types; ++k) {
    solution.demand_at[candidate][k] = 0.0;
    solution.bins[candidate][k]      = 0;
  }

  // 4) Reasignar cada huérfano a su siguiente punto abierto más cercano.
  for (const auto& [i, k] : orphans) {
    // Buscar destino con la helper compartida (excluyendo el punto cerrado).
    int new_point = find_nearest_open(solution, instance, i, k, candidate);

    if (new_point != -1) {
      // Necesitamos la distancia al nuevo punto: la tomamos de dist.
      solution.assignment[i][k]    = new_point;
      solution.assigned_dist[i][k] = instance.dist[new_point].at(i);
      solution.demand_at[new_point][k] += instance.demand[i][k];
      solution.buildings_at[new_point].push_back(std::make_pair(i, k));
      touched.insert(new_point);
    } else {
      solution.assignment[i][k]    = -1;
      solution.assigned_dist[i][k] = std::numeric_limits<double>::infinity();
    }
  }

  // 5) Recalcular los contenedores de los puntos que recibieron huérfanos.
  for (int point : touched) {
    for (int k = 0; k < instance.n_waste_types; ++k) {
      recompute_bins(solution, instance, point, k);
    }
  }

  // (No se toca total_cost ni los contadores de violación: eso es de compute_cost.)
}


void compute_cost(SolutionState& solution, const Instance& instance, double rho) {
  const int n_candidates = instance.n_candidates;
  const int n_types      = instance.n_waste_types;
  const int max_bins     = instance.params.max_bins;

  const double rho_capacity = rho;
  const double rho_coverage = rho;

  double cost = 0.0;
  int n_cap = 0;
  int n_cov = 0;

  // Recorremos cada punto candidato.
  for (int j = 0; j < n_candidates; ++j) {
    if (!solution.open[j]) continue;

    // Coste de apertura del punto.
    cost += instance.candidates[j].opening_cost;

    // Coste de los contenedores, y de paso contar el total para la violación.
    int total_bins = 0;
    for (int k = 0; k < n_types; ++k) {
      const int bins_jk = solution.bins[j][k];
      cost += bins_jk * instance.params.bin_cost[k];
      total_bins += bins_jk;
    }

    //  Comprobacion de violaciones de capacidad
    if (total_bins > max_bins) {
      ++n_cap;
    }
  }
  
  // Comprobación de violaciones de cobertura
  for (int i = 0; i < instance.n_buildings; ++i) {
    for (int k = 0; k < n_types; ++k) {
      if (solution.assignment[i][k] == -1) {
        ++n_cov;
      }
    }
  }

  // Penalización por violaciones.
  cost += rho_capacity * n_cap + rho_coverage * n_cov;

  // Guardar los agregados en la solución.
  solution.total_cost   = cost;
  solution.n_violations_capacity = n_cap;
  solution.n_violations_coverage = n_cov;
}


void apply_swap(SolutionState& solution, const Instance& instance,
                int j_out, int j_in) {
  apply_close(solution, instance, j_out);
  apply_open(solution, instance, j_in);
}