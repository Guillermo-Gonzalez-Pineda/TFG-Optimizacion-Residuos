#include "tabu/solution.hpp"


int bins_for_demand(double demand, double capacity) {
  if (demand <= 1e-9) return 0;   // residuo FP de un punto vaciado (~1e-14): no es demanda real
  return static_cast<int>(std::ceil(demand / capacity));
}

int find_nearest_active(const SolutionState& solution, const Instance& instance,
                        int i, int k, int exclude) {
  for (const ValidCandidate& vc : instance.valid_candidates[i][k]) {
    if (vc.j != exclude && solution.active[vc.j][k]) {
      return vc.j;   // el primero con el tipo k activo es el más cercano (lista ordenada)
    }
  }
  return -1;
}

void init_empty(SolutionState& solution, const Instance& instance) {
  const int n_buildings  = instance.n_buildings;
  const int n_candidates = instance.n_candidates;
  const int n_types      = instance.n_waste_types;

  solution.active.assign(n_candidates, std::vector<bool>(n_types, false));
  solution.assignment.assign(n_buildings, std::vector<int>(n_types, -1));
  solution.assigned_dist.assign(n_buildings,
      std::vector<double>(n_types, std::numeric_limits<double>::infinity()));
  solution.demand_at.assign(n_candidates, std::vector<double>(n_types, 0.0));
  solution.bins.assign(n_candidates, std::vector<int>(n_types, 0));
  solution.buildings_at.assign(n_candidates,
      std::vector<std::vector<int>>(n_types));

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
  if (solution.is_open(candidate)) return;

  // Andamiaje Paso 1: abrir el punto ENTERO = activar todos los tipos a la vez
  // (reproduce el open[j] viejo; los deltas/primitivas per-tipo llegan luego).
  for (int k = 0; k < instance.n_waste_types; ++k) {
    solution.active[candidate][k] = true;
  }

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
        auto& old_list = solution.buildings_at[old_point][k];
        old_list.erase(std::remove(old_list.begin(), old_list.end(), i),
                       old_list.end());
        touched.insert(old_point);
      }

      solution.assignment[i][k]    = candidate;
      solution.assigned_dist[i][k] = new_dist;
      solution.demand_at[candidate][k] += instance.demand[i][k];
      solution.buildings_at[candidate][k].push_back(i);
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
  if (!solution.is_open(candidate)) return;

  // 1) Marcar el punto como cerrado: desactivar todos sus tipos (andamiaje Paso 1).
  for (int k = 0; k < instance.n_waste_types; ++k) {
    solution.active[candidate][k] = false;
  }

  std::set<int> touched;

  // 2) Tomar los huérfanos: los pares (i,k) que REALMENTE estaban en 'candidate'.
  //    Recorremos las listas por tipo y de paso vaciamos el punto cerrado
  //    (ya no sirve a nadie: sin buildings, sin demanda, sin contenedores).
  std::vector<std::pair<int,int>> orphans;
  for (int k = 0; k < instance.n_waste_types; ++k) {
    for (int i : solution.buildings_at[candidate][k]) {
      orphans.emplace_back(i, k);
    }
    solution.buildings_at[candidate][k].clear();
    solution.demand_at[candidate][k] = 0.0;
    solution.bins[candidate][k]      = 0;
  }

  // 3) Reasignar cada huérfano a su siguiente punto abierto más cercano.
  for (const auto& [i, k] : orphans) {
    // Buscar destino con la helper compartida (excluyendo el punto cerrado).
    int new_point = find_nearest_active(solution, instance, i, k, candidate);

    if (new_point != -1) {
      // Necesitamos la distancia al nuevo punto: la tomamos de dist.
      solution.assignment[i][k]    = new_point;
      solution.assigned_dist[i][k] = instance.dist[new_point].at(i);
      solution.demand_at[new_point][k] += instance.demand[i][k];
      solution.buildings_at[new_point][k].push_back(i);
      touched.insert(new_point);
    } else {
      solution.assignment[i][k]    = -1;
      solution.assigned_dist[i][k] = std::numeric_limits<double>::infinity();
    }
  }

  // 4) Recalcular los contenedores de los puntos que recibieron huérfanos.
  for (int point : touched) {
    for (int k = 0; k < instance.n_waste_types; ++k) {
      recompute_bins(solution, instance, point, k);
    }
  }

  // (No se toca total_cost ni los contadores de violación: eso es de compute_cost.)
}


void apply_activate(SolutionState& solution, const Instance& instance,
                    int candidate, int type) {
  // No-op si el tipo ya estaba activo: así el flip z[j] solo se evalúa con cambio real.
  if (solution.active[candidate][type]) return;

  // --- Acoplamiento z[j]: cobrar apertura SOLO si esta activación abre el punto ---
  // (mirar is_open antes y después del flip; no contar tipos a mano).
  const bool was_open = solution.is_open(candidate);
  solution.active[candidate][type] = true;
  const bool now_open = solution.is_open(candidate);
  if (!was_open && now_open) {
    solution.total_cost += instance.candidates[candidate].opening_cost;
  }

  std::set<int> touched;   // puntos cuya demanda del tipo 'type' cambia
  touched.insert(candidate);

  // Atraer los edificios de tipo 'type' para los que 'candidate' es ahora el punto
  // ACTIVO más cercano (misma lógica que apply_open, restringida a un solo tipo).
  for (const BuildingType& bt : instance.buildings_of[candidate]) {
    if (bt.k != type) continue;               // solo el tipo que activamos
    const int    i        = bt.i;
    const double new_dist = bt.distance;

    if (new_dist < solution.assigned_dist[i][type]) {
      const int old_point = solution.assignment[i][type];
      if (old_point != -1) {
        solution.demand_at[old_point][type] -= instance.demand[i][type];
        auto& old_list = solution.buildings_at[old_point][type];
        old_list.erase(std::remove(old_list.begin(), old_list.end(), i),
                       old_list.end());
        touched.insert(old_point);
      }
      solution.assignment[i][type]    = candidate;
      solution.assigned_dist[i][type] = new_dist;
      solution.demand_at[candidate][type] += instance.demand[i][type];
      solution.buildings_at[candidate][type].push_back(i);
    }
  }

  // Solo cambió la demanda del tipo 'type': recalcular SUS bins en los tocados.
  for (int point : touched) {
    recompute_bins(solution, instance, point, type);
  }
}


void apply_deactivate(SolutionState& solution, const Instance& instance,
                      int candidate, int type) {
  // No-op si el tipo ya estaba inactivo.
  if (!solution.active[candidate][type]) return;

  // --- Acoplamiento z[j]: reembolsar apertura SOLO si esta desactivación cierra el punto ---
  const bool was_open = solution.is_open(candidate);
  solution.active[candidate][type] = false;
  const bool now_open = solution.is_open(candidate);
  if (was_open && !now_open) {
    solution.total_cost -= instance.candidates[candidate].opening_cost;
  }

  // Huérfanos del tipo 'type' en 'candidate'; de paso vaciar ese (punto, tipo).
  const std::vector<int> orphans = solution.buildings_at[candidate][type];
  solution.buildings_at[candidate][type].clear();
  solution.demand_at[candidate][type] = 0.0;
  solution.bins[candidate][type]      = 0;

  std::set<int> touched;

  // Reasignar cada huérfano a su siguiente punto con el tipo 'type' ACTIVO.
  for (int i : orphans) {
    const int new_point = find_nearest_active(solution, instance, i, type, candidate);
    if (new_point != -1) {
      solution.assignment[i][type]    = new_point;
      solution.assigned_dist[i][type] = instance.dist[new_point].at(i);
      solution.demand_at[new_point][type] += instance.demand[i][type];
      solution.buildings_at[new_point][type].push_back(i);
      touched.insert(new_point);
    } else {
      solution.assignment[i][type]    = -1;
      solution.assigned_dist[i][type] = std::numeric_limits<double>::infinity();
    }
  }

  for (int point : touched) {
    recompute_bins(solution, instance, point, type);
  }
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
    if (!solution.is_open(j)) continue;

    // Coste de apertura del punto (z[j]: se paga una vez por punto abierto).
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