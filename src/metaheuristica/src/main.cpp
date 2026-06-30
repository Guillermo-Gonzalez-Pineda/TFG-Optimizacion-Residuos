#include <iostream>
#include <algorithm>
#include <chrono>
#include <string>
#include "tabu/instancia.hpp"
#include "tabu/solution.hpp"
#include "tabu/constructive.hpp"
#include "tabu/io.hpp"
#include "tabu/movimientos.hpp"
#include "tabu/tabu.hpp"

int main(int argc, char** argv) {
  // Ruta a la instancia de 500m (se puede sobrescribir por argumento).
  const std::string path = (argc > 1)
      ? argv[1]
      : "../../data/processed/instancia_laguna_500m.json";

  // Cronometramos el cómputo (carga + preproceso + construcción).
  const auto t0 = std::chrono::steady_clock::now();

  Instance instance = load_instance(path);
  preprocess(instance);

  // Test apply_close: abrir y cerrar debe dejar el coste como al principio.
  SolutionState solution;

  construct_initial(solution, instance, 100000.0);
  double cost_greedy = solution.total_cost;

  TabuParams params;   // valores por defecto
  SolutionState result = tabu_search(solution, instance, params);

  std::cout << "\n=== Tabu search ===\n";
  std::cout << "  Coste greedy:  " << cost_greedy << "\n";
  std::cout << "  Coste tabu:    " << result.total_cost << "\n";
  std::cout << "  Mejora:        " << (cost_greedy - result.total_cost)
            << " (" << 100.0*(cost_greedy - result.total_cost)/cost_greedy << "%)\n";
  std::cout << "  Factible: " << (result.n_violations_capacity==0 &&
                                  result.n_violations_coverage==0 ? "SI" : "NO") << "\n";

  return 0;
}