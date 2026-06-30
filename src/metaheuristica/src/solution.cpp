#include "tabu/solution.hpp"

void init_empty(SolutionState& solution, const Instance& instance) {
  const int n_buildings  = instance.n_buildings;
  const int n_candidates = instance.n_candidates;
  const int n_types      = instance.n_waste_types;

  solution.open.assign(n_candidates, false);
  solution.assignment.assign(n_buildings, std::vector<int>(n_types, -1));
  solution.demand_at.assign(n_candidates, std::vector<double>(n_types, 0.0));
  solution.bins.assign(n_candidates, std::vector<int>(n_types, 0));
  solution.buildings_at.assign(n_candidates, std::vector<std::pair<int,int>>{});

  solution.total_cost   = 0.0;
  solution.n_violations = 0;
}