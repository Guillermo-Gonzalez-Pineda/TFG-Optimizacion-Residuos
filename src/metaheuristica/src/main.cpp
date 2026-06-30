#include <iostream>
#include <algorithm>
#include "tabu/instancia.hpp"
#include "tabu/solution.hpp"

int main(int argc, char** argv) {
  // Ruta a la instancia de 500m (se puede sobrescribir por argumento).
  const std::string path = (argc > 1)
      ? argv[1]
      : "../../data/processed/instancia_laguna_500m.json";

  Instance instance = load_instance(path);
  preprocess(instance);

  SolutionState solution;
  init_empty(solution, instance);

  std::cout << "Estado inicializado:\n";
  std::cout << "  open.size():        " << solution.open.size()
            << " (esperado " << instance.n_candidates << ")\n";
  std::cout << "  assignment.size():  " << solution.assignment.size()
            << " (esperado " << instance.n_buildings << ")\n";
  std::cout << "  assignment[0][0]:   " << solution.assignment[0][0]
            << " (esperado -1, sin asignar)\n";
  std::cout << "  total_cost:         " << solution.total_cost << " (esperado 0)\n";
  std::cout << "  ¿algun punto abierto? "
            << (std::any_of(solution.open.begin(), solution.open.end(),
                            [](bool b){ return b; }) ? "SI" : "NO")
            << " (esperado NO)\n";

  return 0;
}