#include <iostream>
#include <algorithm>
#include <chrono>
#include <string>
#include "tabu/instancia.hpp"
#include "tabu/solution.hpp"
#include "tabu/constructive.hpp"
#include "tabu/io.hpp"

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
  SolutionState test;
  init_empty(test, instance);
  compute_cost(test, instance, 100000.0);
  double cost_empty = test.total_cost;

  apply_open(test, instance, 42);
  compute_cost(test, instance, 100000.0);
  double cost_open = test.total_cost;

  apply_close(test, instance, 42);
  compute_cost(test, instance, 100000.0);
  double cost_closed = test.total_cost;

  std::cout << "\n=== Test apply_close (abrir/cerrar punto 42) ===\n";
  std::cout << "  Coste vacio:           " << cost_empty << "\n";
  std::cout << "  Coste tras abrir 42:   " << cost_open << "\n";
  std::cout << "  Coste tras cerrar 42:  " << cost_closed << "\n";
  std::cout << "  ¿Vuelve al vacio? " << (cost_closed == cost_empty ? "SI" : "NO") << "\n";


  // Test 4a: reasignación con éxito. Necesitamos dos puntos que sirvan al edificio 0.
  // Sabemos que el 42 sirve al edificio 0 (tipo 0). Busquemos otro de su lista.
  SolutionState test2;
  init_empty(test2, instance);
  // Abrir el 42 y el segundo candidato válido del edificio 0, tipo 0:
  int primero = instance.valid_candidates[0][0][0].j;   // el más cercano
  int segundo = instance.valid_candidates[0][0][1].j;   // el 2º más cercano
  std::cout << primero << " " << segundo << "\n";
  apply_open(test2, instance, primero);
  apply_open(test2, instance, segundo);
  std::cout << "\n=== Test 4a (reasignacion) ===\n";
  std::cout << "  Edificio 0 asignado a: " << test2.assignment[0][0]
            << " (el mas cercano de los dos abiertos)\n";
  apply_close(test2, instance, test2.assignment[0][0]);   // cerrar su punto actual
  std::cout << "  Tras cerrar su punto, reasignado a: " << test2.assignment[0][0]
            << " (esperado: el otro punto, NO -1)\n";

  return 0;
}